import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Program.PSMNet import PSMNet
import numpy as np
from Program.Load_Image_Single import Load_Image_Single
from Program.Visualize_Disparity import visualize_disparity

if __name__ == "__main__":
    # Nastavení

    dataset = "Tramvaj" # Tramvaj, KITTI, SceneFlow, DrivingStereo

    left_img_path = "Cesta_K_Levemu_Obrazu/1720687472555864693.png"
    right_img_path = "Cesta_K_Pravemu_Obrazu/1720687472555864693.png"
    disp_map_path = "Cesta_K_Mape_Disparity/1720687472555864693.png"

    checkpoint_pth = "Cesta_K_Natrenovanemu_Modelu/model.pth"

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Načti model a checkpoint
    model = PSMNet(max_disparity=192)
    checkpoint = torch.load(checkpoint_pth, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    with torch.no_grad():
        # Načtení jednoho stereo páru
        left_img, right_img, gt_disp, left_img_orig = Load_Image_Single(left_img_path, right_img_path, disp_map_path, dataset=dataset)

        # Vizualizace disparity na prvním vzorku
        left_img, right_img, gt_disp = left_img.to(device), right_img.to(device), gt_disp.to(device)
        pred_disp = model(left_img, right_img)  # Predikovaná disparity mapa
        visualize_disparity(left_img[0], pred_disp[-1], left_img_orig[0], gt_disp, dataset=dataset)  # Vykreslit výsledky