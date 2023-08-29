import cv2
import torch
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image
from utils import BasicUtils

def frame_preprocess(frame, device):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    frame = frame.unsqueeze(0)
    return frame

model = raft_small(weights=Raft_Small_Weights.C_T_V2)
device = BasicUtils().device_chooser()

model.eval()
video_path = "woman.mp4"
cap = cv2.VideoCapture(video_path)
with torch.no_grad():
    while True:
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()
        frame1 = frame_preprocess(frame1, device)
        frame2 = frame_preprocess(frame2, device)
        flows = model(frame1.to(device), frame2.to(device))
        flows = flows[-1].detach().cpu()
        print(f"dtype = {flows.dtype}")
        print(f"shape = {flows.shape} = (N, 2, H, W)")
        print(f"min = {flows.min()}, max = {flows.max()}")
        flow_imgs = flow_to_image(flows)
        flow_imgs = flow_imgs[0].permute(1, 2, 0)
        flow_imgs = flow_imgs.numpy()
        flow_imgs = cv2.cvtColor(flow_imgs, cv2.COLOR_RGB2BGR)
        cv2.imshow("OUT",flow_imgs)
        cv2.waitKey(1)

        if not ret2:
            cap.release()
            cv2.destroyAllWindows()