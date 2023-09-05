from face_detection.face_detection import face_detection
import glob
import torch
import numpy as np
from model.physnet_model import PhysNet
from utils.utils_sig import butter_bandpass


def convert_data_to_tsv(device):
    model = PhysNet(S=2).to(device).eval()
    model.load_state_dict(torch.load('./model_weights.pt', map_location=device))
    spoof_video_path = r"dataset/data/spoof/*.mp4"
    no_spoof_video_path = r"dataset/data/no-spoof/*.mp4"

    spoof_videos = glob.glob(spoof_video_path)
    no_spoof_videos = glob.glob(no_spoof_video_path)

    with open("rppg_data.tsv", "w", encoding="utf-8") as rp:
        rp.write("rpgg_data" + "\t" + "class")
        for spoof_video in spoof_videos:
            face_list, fps = face_detection(video_path=spoof_video)
            print('\nrPPG estimation')

            with torch.no_grad():
                face_list = torch.tensor(face_list.astype('float32'), device=device)
                rppg = model(face_list)[:, -1, :]
                rppg = rppg[0].detach().cpu().numpy()
                rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fps)
                if rppg.shape[0] < 1500:
                    rppg = np.pad(rppg, (0, 1500 - rppg.shape[0]), constant_values=0)
                else:
                    rppg = rppg[:1500]

                rp.write(rppg + "\t" + "0")

        for no_spoof_video in no_spoof_videos:
            face_list, fps = face_detection(video_path=no_spoof_video)
            print('\nrPPG estimation')

            with torch.no_grad():
                face_list = torch.tensor(face_list.astype('float32'), device=device)
                rppg = model(face_list)[:, -1, :]
                rppg = rppg[0].detach().cpu().numpy()
                rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fps)
                if rppg.shape[0] < 1500:
                    rppg = np.pad(rppg, (0, 1500 - rppg.shape[0]), constant_values=0)
                else:
                    rppg = rppg[:1500]

                rp.write(rppg + "\t" + "0")
