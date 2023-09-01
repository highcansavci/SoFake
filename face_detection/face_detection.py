import numpy as np
from facenet_pytorch import MTCNN
import torch
import cv2
import time


def face_detection(frame, prev_time):
    # Initialize MTCNN and device
    device = torch.device('cpu')
    mtcnn = MTCNN(device=device)

    # Perform face detection on the input frame
    frame = frame.transpose((1, 2, 0))  # Change the dimension order to (H, W, C)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))  # Change the dimension order to (C, W, H)
    try:
        boxes, _, = mtcnn.detect(torch.tensor(frame))
    except:
        boxes = None

    # If no face is detected, return an empty array and FPS
    if boxes is None:
        return np.array([]), prev_time

    # Calculate frame rate (fps)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Extract and preprocess the detected face
    box_len = np.max([boxes[0, 2] - boxes[0, 0], boxes[0, 3] - boxes[0, 1]])
    box_half_len = np.round(box_len / 2 * 1.1).astype('int')
    box_mid_y = np.round((boxes[0, 3] + boxes[0, 1]) / 2).astype('int')
    box_mid_x = np.round((boxes[0, 2] + boxes[0, 0]) / 2).astype('int')
    cropped_face = frame[box_mid_y - box_half_len:box_mid_y + box_half_len,
                   box_mid_x - box_half_len:box_mid_x + box_half_len]
    cropped_face = cv2.resize(cropped_face, (128, 128))

    return cropped_face, fps
