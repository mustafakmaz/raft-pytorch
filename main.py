import cv2
import torch
from torchvision.models.optical_flow import raft_small, raft_large
from torchvision.utils import flow_to_image
from utils import BasicUtils
from time import time

# Preprocessing frames for optical flow estimation
def frame_preprocess(frame):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    frame = frame.unsqueeze(0)
    return frame

# Frame counter initialization
frame_counter = 0

# Reading the video file
video_path = "woman.mp4"
cap = cv2.VideoCapture(video_path)

# raft_large weights
# C_T_V1, C_T_V2, C_T_SKHT_V1, C_T_SKHT_V2, C_T_SKHT_K_V1, C_T_SKHT_K_V2

# raft_small weights
# C_T_V1, C_T_V2
model_type = "raft_small"
weights_name = "Raft_Small_Weights.C_T_V1"

# Model-weight compatibility check
if "raft_small" == model_type:
    if "Raft_Small_Weights" in weights_name:
        model = raft_small(weights=weights_name)
    else:
        print("Chosen weights are not usable in raft_small! Please use the appropriate weight!")
        exit()

elif "raft_large" == model_type:
    if "Raft_Large_Weights" in weights_name:
        model = raft_large(weights=weights_name)
    else:
        print("Chosen weights are not usable in raft_large! Please use the appropriate weight!")
        exit()

device = BasicUtils().device_chooser()
model.eval()

with torch.no_grad():
    while True:
        start_time = time()
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()
        frame1 = frame_preprocess(frame1)
        frame2 = frame_preprocess(frame2)
        flows = model(frame1.to(device), frame2.to(device))
        flows = flows[-1].detach().cpu()
        flow_imgs = flow_to_image(flows)
        flow_imgs = flow_imgs[0].permute(1, 2, 0)
        flow_imgs = flow_imgs.numpy()
        flow_imgs = cv2.cvtColor(flow_imgs, cv2.COLOR_RGB2BGR)
        end_time = time()
        cv2.imshow("Optical Flow Output", flow_imgs)
        frame_counter += 1
        print("Frame: {} -- FPS: {:.2f}".format(frame_counter, 1/(end_time - start_time)))
        cv2.waitKey(1)

        if not ret2:
            cap.release()
            cv2.destroyAllWindows()