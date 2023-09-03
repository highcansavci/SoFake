import cv2
import numpy as np
import torch
import sys
sys.path.append('..')
from model.physnet_model import PhysNet
import matplotlib.pyplot as plt
from face_detection.face_detection import face_detection
import tkinter as tk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

from utils.utils_sig import butter_bandpass, hr_fft

# Initialize camera capture
cap = cv2.VideoCapture(0)  # Use the default camera (you can specify a different camera if needed)

# Create GUI window
root = tk.Tk()
root.title("Live rPPG Estimation")
root.geometry("800x1100")

# Create Canvas for displaying camera feed
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# Create Matplotlib figure for live graphs
fig, (ax1, ax2) = plt.subplots(1, ncols=2, figsize=(8, 6))
ax1.set_xlabel('time (sec)')
ax1.set_title('rPPG waveform')
ax2.set_xlabel('heart rate (bpm)')
ax2.set_xlim([40, 200])
ax2.set_title('PSD')

# Create FigureCanvasTkAgg to embed the Matplotlib figure in the Tkinter GUI
canvas_mpl = FigureCanvasTkAgg(fig, master=root)
canvas_mpl_widget = canvas_mpl.get_tk_widget()
canvas_mpl_widget.pack()

# Load your pre-trained PhysNet model here
device = torch.device('cpu')
model = PhysNet(S=2).to(device).eval()
model.load_state_dict(torch.load('./model_weights.pt', map_location=device))

prev_time = time.time()

# Define the number of frames to stack as the temporal dimension (T)
NUM_FRAMES = 30

# Initialize an empty list to store the frames
frame_buffer = []


def close_win(e):
    cap.release()
    cv2.destroyAllWindows()
    root.quit()
    root.destroy()


# Bind the ESC key with the callback function
root.bind('<Escape>', lambda e: close_win(e))


def update_gui():
    global prev_time

    while True:
        ret, frame = cap.read()

        if ret is True:
            frame = cv2.resize(frame, (128, 128))  # Resize to match model input size
            if frame.shape[2] > 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB if more than 3 channels

            frame = frame.transpose((2, 0, 1))  # Change the dimension order to (C, H, W)
            # ...
            face_list, fps = face_detection(frame, prev_time)
            prev_time = time.time()  # Update previous time for the next frame

            # Ensure frame has 3 color channels (RGB)
            if frame.shape[0] != 3:
                frame = np.stack([frame] * 3, axis=0)

            frame_buffer.append(frame)

            if len(frame_buffer) < NUM_FRAMES:
                continue

            # Keep only the last NUM_FRAMES frames in the buffer
            if len(frame_buffer) > NUM_FRAMES:
                frame_buffer.pop(0)

            # Check if we have enough frames to form the temporal dimension
            if len(frame_buffer) == NUM_FRAMES:
                # Stack the frames along the temporal dimension
                frame_sequence = np.stack(frame_buffer, axis=0)  # Shape: (T, C, H, W)
                frame_sequence = frame_sequence[np.newaxis, :]  # Add batch dimension

                # Convert frame sequence to a tensor and send it to the model
                frame_tensor = torch.tensor(frame_sequence.astype('float32')).to(device)

                # Perform face detection and rPPG estimation here on 'frame'

                with torch.no_grad():
                    # face_list = torch.tensor(face_list.astype('float32')).to(device)
                    rppg = model(frame_tensor.permute(0, 2, 1, 3, 4))[:, -1, :]
                    rppg = rppg[0].detach().cpu().numpy()
                    try:
                        rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fps)
                    except:
                        continue

            # Compute PSD
            hr, psd_y, psd_x = hr_fft(rppg, fs=fps)

            # Update live graphs using Matplotlib
            ax1.clear()
            ax1.plot(np.arange(len(rppg)) / fps, rppg)
            ax1.set_xlabel('time (sec)')
            ax1.set_title('rPPG waveform')

            ax2.clear()
            ax2.plot(psd_x, psd_y)
            ax2.set_xlabel('heart rate (bpm)')
            ax2.set_xlim([40, 200])
            ax2.set_title('PSD')

            canvas_mpl.draw()

            # Convert 'frame' to a format compatible with Tkinter
            frame = frame.transpose(1, 2, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)

            # Update the Canvas with the new frame
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.img = img  # Store the image reference to prevent garbage collection

            # Schedule the next update
            root.update()
        else:
            break


if __name__ == "__main__":
    # Start the GUI update loop
    update_gui()
