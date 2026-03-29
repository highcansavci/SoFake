import csv
import logging
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so model/ and utils/ are importable
# when this file is run as  python evaluation/fusion_evaluator.py
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from model.physnet_model import PhysNet          # 3D-CNN rPPG extractor
from utils.utils_sig import butter_bandpass

logging.basicConfig(level=logging.INFO, format="%(message)s")


class FusionEvaluator:
    """
    Two-stage rPPG-based deepfake evaluator.

    Stage 1 – PhysNet (3D CNN):
        Input  : (1, 3, T, 128, 128)  RGB clip, pixel values in [0, 1]
        Output : (1, N, T)            ST-rPPG block  (N = S*S+1 = 5 for S=2)

    Stage 2 – Signal analysis:
        rppg_score  = cardiac-band power ratio from the current 30-frame window,
                      in [0, 1].  High -> strong cardiac signal (likely real).
        fft_score   = cardiac-band power fraction from a rolling ~10-second
                      accumulator, in [0, 1].  Computed with rfft + argmax
                      (robust; never returns 0 due to "no peaks found").
                      Also logs the dominant HR in bpm.
    """

    _CLIP_LEN      = 30    # frames per PhysNet inference window
    _STRIDE        = 15    # inference cadence (frames)
    _FACE_SZ       = 128   # spatial resolution expected by PhysNet
    _FFT_ACCUM_LEN = 300   # rolling rPPG accumulator length (~10 s at 30 fps)
    _FFT_MIN_LEN   = 60    # minimum samples before attempting FFT (~2 s)

    def __init__(self, video_path: str, weight_path: str):
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_path = Path(video_path).resolve()

        # ── Fix #1: load model_weights.pt into PhysNet with strict=True ────────
        self.physnet = PhysNet(S=2).to(self.device)
        state_dict   = torch.load(weight_path, map_location=self.device)
        self.physnet.load_state_dict(state_dict)   # strict=True (default)
        self.physnet.eval()
        logging.info(f"PhysNet loaded from {weight_path} on {self.device}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR uint8 -> RGB float32 (128, 128, 3) in [0, 1]."""
        # Fix #2: BGR -> RGB before feeding into PhysNet
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._FACE_SZ, self._FACE_SZ))
        return resized.astype(np.float32) / 255.0

    def _to_tensor(self, frames: list) -> torch.Tensor:
        """List of T x (H, W, 3) float32 arrays -> (1, 3, T, H, W) tensor."""
        # Fix #3: correct 5-D input shape for PhysNet
        arr = np.stack(frames, axis=0)       # (T, H, W, 3)
        arr = arr.transpose(3, 0, 1, 2)      # (3, T, H, W)
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)  # (1, 3, T, H, W)

    @staticmethod
    def _znorm(x: np.ndarray) -> np.ndarray:
        """Zero-mean / unit-std normalisation; safe against flat signals."""
        std = x.std()
        return (x - x.mean()) / (std if std > 1e-8 else 1.0)

    def _rppg_score(self, signal: np.ndarray, fps: float) -> float:
        """Cardiac-band power ratio for the current 30-frame window, in [0, 1]."""
        try:
            filtered    = butter_bandpass(signal, lowcut=0.6, highcut=4.0, fs=fps)
            total_power = float(np.var(signal))
            if total_power < 1e-8:
                return 0.0
            return float(np.clip(np.var(filtered) / total_power, 0.0, 1.0))
        except Exception:
            return 0.0

    def _fft_score(self, accum_signal: np.ndarray, fps: float) -> tuple:
        """
        Cardiac-band power fraction and dominant HR from the accumulated signal.
        Returns (score in [0,1], hr_bpm).

        Uses rfft + argmax instead of find_peaks so it always returns a result
        once >= _FFT_MIN_LEN samples are available, even with sparse spectra.
        """
        if len(accum_signal) < self._FFT_MIN_LEN:
            return 0.0, 0.0
        try:
            # 1. Remove slow inter-chunk drift
            sig = accum_signal - np.linspace(
                accum_signal[0], accum_signal[-1], len(accum_signal)
            )
            # 2. Bandpass to cardiac band
            filtered = butter_bandpass(sig, lowcut=0.6, highcut=4.0, fs=fps)
            # 3. FFT with Hann window
            windowed = filtered * np.hanning(len(filtered))
            spectrum = np.abs(np.fft.rfft(windowed))
            freqs    = np.fft.rfftfreq(len(windowed), d=1.0 / fps)
            # 4. Restrict to cardiac band [0.6, 4] Hz
            cardiac_mask = (freqs >= 0.6) & (freqs <= 4.0)
            if not cardiac_mask.any():
                return 0.0, 0.0
            # 5. Dominant HR via argmax (never fails unlike find_peaks)
            cardiac_spectrum = spectrum.copy()
            cardiac_spectrum[~cardiac_mask] = 0.0
            peak_bin = int(np.argmax(cardiac_spectrum))
            hr_bpm   = float(freqs[peak_bin] * 60.0)
            # 6. Score = cardiac-band energy fraction
            total_power   = float(np.sum(spectrum))   + 1e-8
            cardiac_power = float(np.sum(spectrum[cardiac_mask]))
            score = float(np.clip(cardiac_power / total_power, 0.0, 1.0))
            return score, hr_bpm
        except Exception:
            return 0.0, 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> list:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_buf = deque(maxlen=self._CLIP_LEN)
        # Each appended entry is a z-normalised 15-sample chunk to eliminate
        # inter-window DC steps before accumulation.
        rppg_accum = deque(maxlen=self._FFT_ACCUM_LEN)
        results    = []
        frame_cnt  = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_buf.append(self._preprocess(frame))
            frame_cnt += 1

            if len(frame_buf) == self._CLIP_LEN and frame_cnt % self._STRIDE == 0:
                # ── Stage 1: PhysNet 3D-CNN inference ─────────────────────
                clip = self._to_tensor(list(frame_buf))    # (1, 3, 30, 128, 128)
                with torch.no_grad():
                    rppg_block = self.physnet(clip)        # (1, 5, 30)

                # Spatially-averaged rPPG signal (last slot in the N-dim block)
                signal = rppg_block[0, -1, :].cpu().numpy()   # (30,)

                # z-normalise the newest _STRIDE samples before appending so
                # consecutive chunks share the same scale and zero-mean.
                chunk = self._znorm(signal[-self._STRIDE:])
                rppg_accum.extend(chunk.tolist())

                # ── Stage 2: Signal-based scores ──────────────────────────
                r_score         = self._rppg_score(signal, fps)
                accum_arr       = np.array(rppg_accum)
                f_score, hr_bpm = self._fft_score(accum_arr, fps)

                results.append({
                    "window":     frame_cnt,
                    "rppg_score": round(r_score, 4),
                    "fft_score":  round(f_score, 4),
                })
                logging.info(
                    f"[window={frame_cnt:>4d}]  "
                    f"rppg_score={r_score:.4f}  "
                    f"fft_score={f_score:.4f}  "
                    f"HR={hr_bpm:.1f} bpm  "
                    f"(accum={len(rppg_accum)} samples)"
                )

        cap.release()
        return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FusionEvaluator – rPPG deepfake scorer")
    parser.add_argument(
        "--video",
        default="../datasets/FaceForensics/original_sequences/youtube/c23/videos/183.mp4",
    )
    parser.add_argument("--output",  default="results/183_final.csv")
    parser.add_argument("--weights", default="inference/model_weights.pt")
    args = parser.parse_args()

    evaluator = FusionEvaluator(args.video, args.weights)
    data      = evaluator.evaluate()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["window", "rppg_score", "fft_score"])
        writer.writeheader()
        writer.writerows(data)

    logging.info(f"\nResults saved to {args.output}  ({len(data)} windows)")
