import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch
from config.config import Config

config_ = Config().config

def face_detection(video_path):
    device = torch.device("cpu")
    mtcnn = MTCNN(device=device)

    cap = cv2.VideoCapture(video_path)

    video_list = []
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_list.append(frame)
        else:
            break

    cap.release()

    face_list = []
    for t, frame in enumerate(video_list):
        boxes, _, = mtcnn.detect(frame)
        box_len = np.max([boxes[0, 2] - boxes[0, 0], boxes[0, 3] - boxes[0, 1]])
        box_half_len = np.round(box_len / 2 * 1.1).astype('int')
        box_mid_y = np.round((boxes[0, 3] + boxes[0, 1]) / 2).astype('int')
        box_mid_x = np.round((boxes[0, 2] + boxes[0, 0]) / 2).astype('int')
        cropped_face = frame[box_mid_y - box_half_len:box_mid_y + box_half_len,
                       box_mid_x - box_half_len:box_mid_x + box_half_len]
        if cropped_face is None or np.prod(cropped_face.shape) == 0:
            continue
        cropped_face = cv2.resize(cropped_face, (128, 128))
        face_list.append(cropped_face)

    face_list = np.array(face_list)  # (T, H, W, C)
    print(face_list.shape)
    face_list = np.transpose(face_list, (3, 0, 1, 2))  # (C, T, H, W)
    face_list = np.array(face_list)[np.newaxis]

    return face_list, 30
