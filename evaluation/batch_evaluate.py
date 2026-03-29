#!/usr/bin/env python3
"""
batch_evaluate.py

Scans both original_sequences and manipulated_sequences under a FaceForensics
dataset root and runs FusionEvaluator on every .mp4 found.  Results are saved
as CSV files mirroring the original directory tree under an output root.

Usage (from ~/im_lab/SoFake):
    python3 evaluation/batch_evaluate.py \
        --dataset  ../datasets/FaceForensics \
        --weights  inference/model_weights.pt \
        --output   results/batch
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.fusion_evaluator import FusionEvaluator

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def find_videos(dataset_root: Path) -> list[tuple[Path, str]]:
    """
    Walk the dataset tree and return (video_path, label) pairs.
    label = "real" for original_sequences, "fake" for manipulated_sequences.
    """
    entries = []
    for video in sorted(dataset_root.rglob("*.mp4")):
        if "original_sequences" in video.parts:
            label = "real"
        elif "manipulated_sequences" in video.parts:
            label = "fake"
        else:
            label = "unknown"
        entries.append((video, label))
    return entries


def relative_csv_path(video: Path, dataset_root: Path, output_root: Path) -> Path:
    """
    Mirror the video's relative path under output_root, replacing .mp4 with .csv.
    e.g.  original_sequences/youtube/c23/videos/183.mp4
          -> output_root/original_sequences/youtube/c23/videos/183.csv
    """
    rel = video.relative_to(dataset_root)
    return output_root / rel.with_suffix(".csv")


def run_batch(dataset_root: Path, weight_path: str, output_root: Path) -> None:
    videos = find_videos(dataset_root)
    if not videos:
        log.error(f"No .mp4 files found under {dataset_root}")
        sys.exit(1)

    log.info(f"Found {len(videos)} video(s) under {dataset_root}")
    log.info(f"Results will be written to {output_root}\n")

    summary_rows = []
    t0_total = time.time()

    for i, (video, label) in enumerate(videos, 1):
        rel = video.relative_to(dataset_root)
        log.info(f"[{i}/{len(videos)}] {rel}  (label={label})")

        csv_out = relative_csv_path(video, dataset_root, output_root)
        csv_out.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        try:
            evaluator = FusionEvaluator(str(video), weight_path)
            data = evaluator.evaluate()
        except Exception as exc:
            log.error(f"  ERROR: {exc}")
            summary_rows.append({"video": str(rel), "label": label,
                                  "windows": 0, "mean_rppg": "", "mean_fft": "",
                                  "status": f"ERROR: {exc}"})
            continue

        elapsed = time.time() - t0

        # Write per-video CSV
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["window", "rppg_score", "fft_score"])
            writer.writeheader()
            writer.writerows(data)

        if data:
            mean_rppg = round(sum(r["rppg_score"] for r in data) / len(data), 4)
            mean_fft  = round(sum(r["fft_score"]  for r in data) / len(data), 4)
        else:
            mean_rppg = mean_fft = 0.0

        log.info(f"  -> {len(data)} windows  mean_rppg={mean_rppg}  "
                 f"mean_fft={mean_fft}  ({elapsed:.1f}s)")
        summary_rows.append({"video": str(rel), "label": label,
                              "windows": len(data),
                              "mean_rppg": mean_rppg, "mean_fft": mean_fft,
                              "status": "ok"})

    # Write summary CSV
    summary_path = output_root / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video", "label", "windows", "mean_rppg", "mean_fft", "status"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    total_elapsed = time.time() - t0_total
    log.info(f"\nDone. {len(videos)} videos in {total_elapsed:.1f}s")
    log.info(f"Summary -> {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch rPPG deepfake evaluator")
    parser.add_argument("--dataset",  default="../datasets/FaceForensics",
                        help="Path to FaceForensics root")
    parser.add_argument("--weights",  default="inference/model_weights.pt")
    parser.add_argument("--output",   default="results/batch",
                        help="Root directory for CSV output")
    args = parser.parse_args()

    run_batch(
        dataset_root=Path(args.dataset).resolve(),
        weight_path=args.weights,
        output_root=Path(args.output),
    )
