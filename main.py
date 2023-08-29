import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.optical_flow import Raft_Small_Weights
from torchvision.utils import flow_to_image
from utils import BasicUtils

def frame_preprocess(frame, device):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    return frame

model = torchvision.models.optical_flow.raft_small(weights=Raft_Small_Weights)
device = BasicUtils().device_chooser

model.eval()
video_path = "/content/woman.mp4"
cap = cv2.VideoCapture(video_path)
with torch.no_grad():
    while True:
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()
        frame1 = frame_preprocess(frame1, device)
        frame2 = frame_preprocess(frame2, device)
        # predict the flow
        flows = model(frame1.to(device), frame2.to(device))
        # transpose the flow output and convert it into numpy array
        flows = flows[-1].detach().cpu()
        print(f"dtype = {flows.dtype}")
        print(f"shape = {flows.shape} = (N, 2, H, W)")
        print(f"min = {flows.min()}, max = {flows.max()}")
        flow_imgs = flow_to_image(flows)
        flow_imgs = flow_imgs[0].permute(1, 2, 0)
        flow_imgs = flow_imgs.numpy()
        flow_imgs = cv2.cvtColor(flow_imgs, cv2.COLOR_RGB2BGR)
        cv2.imshow(flow_imgs)

        if not ret2:
            cap.release()
            cv2.destroyAllWindows()