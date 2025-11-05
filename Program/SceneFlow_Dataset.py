import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from Program.Preprocessing import Preprocessing

def load_pfm(file):
    """ Funkce pro načtení disparity z PFM souboru """
    with open(file, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        if header != "PF" and header != "Pf":
            raise ValueError("Nesprávný formát PFM souboru.")

        dim_line = f.readline().decode("utf-8").rstrip()
        width, height = map(int, dim_line.split())

        scale = float(f.readline().decode("utf-8").rstrip())
        if scale < 0:  
            endian = "<"
        else:
            endian = ">"

        data = np.fromfile(f, endian + "f").reshape(height, width)
        data = np.flipud(data)  # Otočení obrazu na správnou orientaci
        return data

class SceneFlowDataset(Dataset):
    def __init__(self, root_dir, split='TRAIN'):

        self.root_dir = root_dir
        self.split = split
        self.image_paths = []
        self.disparity_paths = []
        self.transform = Preprocessing()

        # Cesty k hlavním složkám
        if split == 'VAL':
            image_root = os.path.join(root_dir, "frames_cleanpass/Skola/Diplomka/Datasets/Scene_Flow/frames_cleanpass", split)
            disparity_root = os.path.join(root_dir, "disparity/Skola/Diplomka/Datasets/Scene_Flow/disparity", split)
        else:
            image_root = os.path.join(root_dir, "frames_cleanpass", split)
            disparity_root = os.path.join(root_dir, "disparity", split)

        if split == 'TRAIN':
            parts = ["A", "B", "C", "Driving", "Monkaa"]
        elif split == 'TEST':
            parts = ["B", "C"]
        elif split == 'VAL':
            parts = ["A"]
        for part in parts:
            part_img_dir = os.path.join(image_root, part)
            part_disp_dir = os.path.join(disparity_root, part)

            for scene in sorted(os.listdir(part_img_dir)):  
                scene_img_dir = os.path.join(part_img_dir, scene)
                scene_disp_dir = os.path.join(part_disp_dir, scene)

                if part == "Driving":
                    for scene2 in sorted(os.listdir(scene_img_dir)):
                        scene2_img_dir = os.path.join(scene_img_dir, scene2, "slow")
                        scene2_disp_dir = os.path.join(scene_disp_dir, scene2, "slow")

                        left_img_dir = os.path.join(scene2_img_dir, "left")
                        right_img_dir = os.path.join(scene2_img_dir, "right")
                        left_disp_dir = os.path.join(scene2_disp_dir, "left")

                        # Načítání všech souborů ve složce
                        for img_name in sorted(os.listdir(left_img_dir)):
                            if img_name.endswith(".png"):  
                                left_img_path = os.path.join(left_img_dir, img_name)
                                right_img_path = os.path.join(right_img_dir, img_name)
                                
                                disparity_path = os.path.join(left_disp_dir, img_name.replace(".png", ".pfm"))
                                if os.path.exists(disparity_path):
                                    self.image_paths.append((left_img_path, right_img_path))
                                    self.disparity_paths.append(disparity_path)

                else:
                    left_img_dir = os.path.join(scene_img_dir, "left")
                    right_img_dir = os.path.join(scene_img_dir, "right")
                    left_disp_dir = os.path.join(scene_disp_dir, "left")

                    # Načítání všech souborů ve složce
                    for img_name in sorted(os.listdir(left_img_dir)):
                        if img_name.endswith(".png"): 
                            left_img_path = os.path.join(left_img_dir, img_name)
                            right_img_path = os.path.join(right_img_dir, img_name)
                            
                            disparity_path = os.path.join(left_disp_dir, img_name.replace(".png", ".pfm"))
                            if os.path.exists(disparity_path):
                                self.image_paths.append((left_img_path, right_img_path))
                                self.disparity_paths.append(disparity_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Načtení levého a pravého obrázku
        left_img = cv2.imread(self.image_paths[idx][0], cv2.IMREAD_COLOR)
        right_img = cv2.imread(self.image_paths[idx][1], cv2.IMREAD_COLOR)

        print(f"Název obrazku: {self.image_paths[idx][0]}")

        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        # Načtení disparity mapy
        disp_map = load_pfm(self.disparity_paths[idx]).astype(np.float32)

        left_img = torch.tensor(left_img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        right_img = torch.tensor(right_img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        disp_map = torch.tensor(disp_map, dtype=torch.float32)

        if self.split == "TEST":
            left_img_orig = left_img
            obj_map = left_img
            left_img, right_img, disp_map = self.transform(left_img, right_img, disp_map, split=self.split)
            return left_img, right_img, obj_map, disp_map, left_img_orig

        left_img, right_img, disp_map = self.transform(left_img, right_img, disp_map, split=self.split)
        return left_img, right_img, disp_map

def get_dataloader(root_dir, split='TRAIN', batch_size=1, shuffle=True, num_workers=0):
    """ Vytvoření dataloaderu pro SceneFlow """
    dataset = SceneFlowDataset(root_dir, split)
    if len(dataset) == 0:
        print("Error: No data found in SceneFlow dataset.")
        return None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


