import cv2
import numpy as np
import torch
import sys

sys.path.append('..')
from model.physnet_model import PhysNet
import matplotlib.pyplot as plt
from face_detection.face_detection import face_detection
import time

from utils.utils_sig import butter_bandpass, hr_fft

device = torch.device('cpu')
model = PhysNet(S=2).to(device).eval()
model.load_state_dict(torch.load('./model_weights.pt', map_location=device))

def live_demo():
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('fake.mp4', fourcc, 20.0, size)
    frame_count = 0

    while True:
        _, frame = cap.read()
        frame_count += 1

        cv2.imshow('Recording...', frame)
        out.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            out.release()
            face_list, fps = face_detection(video_path="fake.mp4")
            print('\nrPPG estimation')

            with torch.no_grad():
                face_list = torch.tensor(face_list.astype('float32')).to(device)
                rppg = model(face_list)[:, -1, :]
                rppg = rppg[0].detach().cpu().numpy()
                rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fps)

            hr, psd_y, psd_x = hr_fft(rppg, fs=fps)

            fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))

            ax1.plot(np.arange(len(rppg)) / fps, rppg)
            ax1.set_xlabel('time (sec)')
            ax1.grid('on')
            ax1.set_title('rPPG waveform')

            ax2.plot(psd_x, psd_y)
            ax2.set_xlabel('heart rate (bpm)')
            ax2.set_xlim([40, 200])
            ax2.grid('on')
            ax2.set_title('PSD')

            plt.savefig('./results.png')

            print('heart rate: %.2f bpm' % hr)

    cap.release()
    cv2.destroyAllWindows()


# Load your pre-trained PhysNet model here
device = torch.device('cpu')
model = PhysNet(S=2).to(device).eval()
model.load_state_dict(torch.load('./model_weights.pt', map_location=device))

prev_time = time.time()

# Define the number of frames to stack as the temporal dimension (T)
NUM_FRAMES = 1000

if __name__ == "__main__":
    # Start the live demo
    live_demo()
