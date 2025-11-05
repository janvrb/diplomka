import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Program.ACVNet import ACVNet
import numpy as np
from Program.Load_Image_Single import Load_Image_Single
from Program.Visualize_Disparity import visualize_disparity
import json
import os
import glob

def load_config(config_path='config.json'):
    """Načte konfigurační soubor JSON."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"CHYBA: Konfigurační soubor '{config_path}' nebyl nalezen.")
        exit(1)
    except json.JSONDecodeError:
        print(f"CHYBA: Konfigurační soubor '{config_path}' má neplatný formát JSON.")
        exit(1)

if __name__ == "__main__":
    # 1. Načtení konfigurace
    config = load_config('config.json')

    dataset = config.get("dataset", "Tramvaj")
    checkpoint_pth = config.get("checkpoint_pth")
    
    # Načtení cest
    paths = config.get("paths", {})
    left_folder = paths.get("left_folder")
    right_folder = paths.get("right_folder")
    disp_folder = paths.get("disp_folder")

    # Načtení nastavení modelu
    model_settings = config.get("model_settings", {})

    # Kontrola, zda byly zadány klíčové cesty
    if not all([checkpoint_pth, left_folder, right_folder, disp_folder]):
        print("CHYBA: V 'config.json' chybí některé klíčové cesty (checkpoint_pth, left_folder, atd.).")
        exit(1)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Používám zařízení: {device}")

    # Načti model a checkpoint
    model = ACVNet(max_disparity=192, freeze_att_weights=False, att_weights_only=False)
    checkpoint = torch.load(checkpoint_pth, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Najdeme všechny levé obrázky (png a jpg)
    left_image_paths = sorted(
        glob.glob(os.path.join(left_folder, "*.png")) + 
        glob.glob(os.path.join(left_folder, "*.jpg"))
    )

    if not left_image_paths:
        print(f"CHYBA: Ve složce '{left_folder}' nebyly nalezeny žádné obrázky (.png, .jpg).")
        exit(1)
        
    print(f"Nalezeno {len(left_image_paths)} obrázků ke zpracování.")

    with torch.no_grad():
        for left_img_path in left_image_paths:
            # Název vstupu
            filename = os.path.basename(left_img_path)
            print(f"--- Zpracovávám: {filename} ---")

            right_img_path = os.path.join(right_folder, filename)
            disp_map_path = os.path.join(disp_folder, filename)

            # Kontrola, zda existují všechny potřebné soubory
            if not os.path.exists(right_img_path):
                print(f"Varování: Chybí pravý obrázek pro {filename}. Přeskakuji.")
                continue
            if not os.path.exists(disp_map_path):
                print(f"Varování: Chybí GT disparita pro {filename}. Přeskakuji.")
                continue

            # Načtení jednoho stereo páru
            try:
                left_img, right_img, gt_disp, left_img_orig = Load_Image_Single(
                    left_img_path, right_img_path, disp_map_path, dataset='Tramvaj'
                )
            except Exception as e:
                print(f"Chyba při načítání obrázku {filename}: {e}. Přeskakuji.")
                continue
            # Vizualizace disparity na prvním vzorku
            left_img, right_img, gt_disp = left_img.to(device), right_img.to(device), gt_disp.to(device)
            pred_disp = model(left_img, right_img)  # Predikovaná disparity mapa
            visualize_disparity(left_img[0], pred_disp[-1], left_img_orig[0], gt_disp, dataset=dataset)  # Vykreslit výsledky