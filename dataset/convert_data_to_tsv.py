from face_detection.face_detection import face_detection
import glob
import torch
import pandas as pd
import numpy as np
from model.physnet_model import PhysNet
from utils.utils_sig import butter_bandpass
import msgpack


def convert_data_to_tsv(device):
    model = PhysNet(S=2).to(device).eval()
    model.load_state_dict(torch.load('../inference/model_weights.pt', map_location=device))
    spoof_video_path = r"../dataset/data/spoof/*.mp4"
    no_spoof_video_path = r"../dataset/data/no-spoof/*.MOV"

    spoof_videos = glob.glob(spoof_video_path)
    no_spoof_videos = glob.glob(no_spoof_video_path)

    rppg_data = []
    labels = []

    for spoof_video in spoof_videos:
        try:
            face_list, fps = face_detection(video_path=spoof_video)
        except TypeError:
            continue

        with torch.no_grad():
            face_list = torch.tensor(face_list.astype('float32'), device=device)
            rppg = model(face_list)[:, -1, :]
            rppg = rppg[0].detach().cpu().numpy()
            rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fps)
            if rppg.shape[0] < 500:
                rppg = np.pad(rppg, (0, 1500 - rppg.shape[0]), constant_values=0)
            else:
                rppg = rppg[:500]

            rppg_data.append(rppg)
            labels.append(1)

    for no_spoof_video in no_spoof_videos:
        try:
            face_list, fps = face_detection(video_path=no_spoof_video)
        except TypeError:
            continue

        with torch.no_grad():
            face_list = torch.tensor(face_list.astype('float32'), device=device)
            rppg = model(face_list)[:, -1, :]
            rppg = rppg[0].detach().cpu().numpy()
            rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fps)
            if rppg.shape[0] < 1500:
                rppg = np.pad(rppg, (0, 1500 - rppg.shape[0]), constant_values=0)
            else:
                rppg = rppg[:1500]

            rppg_data.append(rppg)
            labels.append(0)

    data = list(zip(rppg_data, labels))
    df = pd.DataFrame(data, columns=['rppg_data', 'label'])

    with open("df.msgpack", "wb") as f:
        f.write(df.to_msgpack())

