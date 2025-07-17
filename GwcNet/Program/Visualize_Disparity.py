import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Program.GwcNet import GwcNet
import numpy as np
from Program.Load_Image_Single import Load_Image_Single

def visualize_disparity(left_img, pred_disp, left_img_orig, gt_disp=None, dataset="Tramvaj"):

    mask = None

    if dataset == "SceneFlow":
        pred_disp = pred_disp.squeeze(0,1)
        mask = (gt_disp >= 0) & (gt_disp < 192)
    elif dataset == 'KITTI':
        mask = (gt_disp > 0)
    else:
        pred_disp = pred_disp.squeeze(0,1)
        mask = (gt_disp > 0)

    epe = torch.abs(pred_disp[mask] - gt_disp[mask]).mean().item()
    rmse = torch.sqrt(torch.mean((pred_disp[mask] - gt_disp[mask]) ** 2)).item()

    error = torch.abs(pred_disp - gt_disp)  # Absolutní chyba 
    threshold = torch.max(torch.tensor(3.0, device=gt_disp.device), 0.05 * gt_disp)  # 3 px nebo 5 % disparity
    error_mask = (error > threshold)  # Pixely s chybou větší než threshold
    three_pe_all = (error_mask & mask).float().sum() / mask.float().sum() * 100

    print(f"Validation Results: EPE = {epe:.4f}, RMSE = {rmse:.4f}")
    print(f"Three-Pixel Error (ALL): {three_pe_all.item():.2f}%")

    # Přesuneme na CPU a upravíme formát
    left_img_orig = left_img_orig.permute(1, 2, 0)  
    left_img = left_img.permute(1, 2, 0).cpu().numpy()  
    pred_disp = pred_disp.squeeze(0,1).detach().cpu().numpy()  

    gt_disp = gt_disp.detach().cpu().numpy()  # Skutečná disparity mapa
    # disp_values = gt_disp[gt_disp > 0].flatten()  # Odstraníme nuly (neznámé disparity)
    # plt.hist(disp_values, bins=50, color='blue', alpha=0.7)
    # plt.axvline(x=35, color='red', linestyle='dashed', label="Threshold (35)")
    # plt.xlabel("Disparity Value")
    # plt.ylabel("Frequency")
    # plt.title("Disparity Value Histogram")
    # plt.legend()
    # plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    mask = gt_disp > 0
    # print(f"Mask shape: {mask.shape}")
    # print(f"Pred Disp shape: {pred_disp.shape}")
    # print(f"GT Disp shape: {gt_disp.shape}")
    error_map = np.zeros_like(gt_disp)
    error_map[mask] = np.abs(pred_disp[mask] - gt_disp[mask])

    axs[0, 0].imshow(left_img_orig)  # Levý obrázek
    axs[0, 0].set_title("Levý obraz", fontsize=34)
    axs[0, 0].axis("off")

    im1 = axs[0, 1].imshow(pred_disp, cmap='jet')  # Predikovaná disparity mapa
    axs[0, 1].set_title("Predikovaná disparita", fontsize=34)
    axs[0, 1].axis("off")

    im2 = axs[1, 0].imshow(gt_disp, cmap='jet')  # Skutečná disparity mapa
    axs[1, 0].set_title("Ground Truth Disparita", fontsize=34)
    axs[1, 0].axis("off")

    # Error mapa
    im = axs[1, 1].imshow(error_map, cmap='hot', vmin=0, vmax=np.percentile(error_map, 99))  
    axs[1, 1].set_title("Mapa Chyb (Abs Rozdíl)", fontsize=34)
    axs[1, 1].axis("off")

    fig.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    plt.show()