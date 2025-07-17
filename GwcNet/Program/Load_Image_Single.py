import torch
import cv2
import numpy as np
from Program.SceneFlow_Dataset import load_pfm
from Program.Preprocessing import Preprocessing

def Load_Image_Single(left_img_path, right_img_path, disp_map_path, split="validating", dataset="Tramvaj"):

    # Načtení obrázků
    left_img = cv2.cvtColor(cv2.imread(left_img_path), cv2.COLOR_BGR2RGB)
    right_img = cv2.cvtColor(cv2.imread(right_img_path), cv2.COLOR_BGR2RGB)

    # Načtení disparity mapy
    if dataset == "Tramvaj" or dataset == "KITTI" or dataset == "DrivingStereo":
        disp_map = cv2.imread(disp_map_path, cv2.IMREAD_UNCHANGED).astype('float32') #KITTI nebo Driving Stereo nebo Tramvaj
    else:
        disp_map = load_pfm(disp_map_path).astype(np.float32) # SceneFlow


    # Převod na PyTorch tensor
    left_img = torch.tensor(left_img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
    right_img = torch.tensor(right_img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
    if dataset == "Tramvaj" or dataset == "SceneFlow":
        disp_map = torch.tensor(disp_map, dtype=torch.float32) # SceneFlow nebo Tramvaj
    else:
        disp_map = torch.tensor(disp_map, dtype=torch.float32) / 256.0 # KITTI nebo Driving Stereo

    left_img = left_img[ :, :, :right_img.size(2)] # Pro Tramvaj dataset
    disp_map = disp_map[ :, :right_img.size(2)]

    left_img = left_img.unsqueeze(0)
    right_img = right_img.unsqueeze(0)

    left_img_orig = left_img

    transform = Preprocessing()

    left_img, right_img, disp_map = transform(left_img, right_img, disp_map, split)

    # print(f"Left Image Shape: {left_img.shape}")
    # print(f"Right Image Shape: {right_img.shape}")
    # print(f"Disparity Map Shape: {disp_map.shape}")

    return left_img, right_img, disp_map, left_img_orig