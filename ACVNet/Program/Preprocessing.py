import torch
import torchvision.transforms as transforms

class Preprocessing:
    """ Předzpracování dat - normalizace a náhodný ořez """
    
    def __init__(self, crop_size=(256, 512)):

        self.crop_size = crop_size
        
        # Normalizace barev
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, left_img, right_img, disp_map=None, split='TRAIN'):

        # if split == 'TRAIN' or split == 'training' or split == 'VAL' or split == 'validating':
        if split == 'TRAIN' or split == 'training':
            i, j, h, w = transforms.RandomCrop.get_params(left_img, output_size=self.crop_size)
            left_img = transforms.functional.crop(left_img, i, j, h, w)
            right_img = transforms.functional.crop(right_img, i, j, h, w)

        # Normalizace barev
        left_img = self.normalize(left_img)
        right_img = self.normalize(right_img)

        if disp_map is not None:
            if split == 'TRAIN' or split == 'training':
                disp_map = transforms.functional.crop(disp_map, i, j, h, w)
            return left_img, right_img, disp_map
        return left_img, right_img
