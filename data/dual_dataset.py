import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import yaml
import pandas as pd
from safetensors.torch import load_file
from model.model_loader import load_model
from data.rgb_dataset import RGBDataset


class DualPhaseDataset(RGBDataset):
    """
    Handles combinations of two phases (e.g., Pre+Post1 or Post1+Post2)
    """
    def __init__(self, phase_indices, *args, **kwargs):
        # phase_indices: e.g., [0, 1] for Pre+Post1
        super().__init__(*args, **kwargs)
        self.phase_indices = phase_indices

    def __getitem__(self, idx):
        img_paths, label = self.samples[idx]

        # 1. Load the two target images
        img1 = Image.open(img_paths[self.phase_indices[0]]).convert("L")
        img2 = Image.open(img_paths[self.phase_indices[1]]).convert("L")

        transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
        ])
        
        t1 = transform(img1) # [1, 192, 192]
        t2 = transform(img2) # [1, 192, 192]

        # 2. Combine into 3 channels [3, 192, 192]
        # Option: Phase 1, Phase 2, and their Average as the 3rd channel
        # avg = (t1 + t2) / 2
        combined_rgb = torch.cat([t1, t2], dim=0)

        # 3. Prepare for 3D Vision Model [1, 3, 16, 192, 192]
        # This matches your model_config "in_chans": 3
        input_tensor = combined_rgb.unsqueeze(1).repeat(1, 16, 1, 1)
        input_tensor = input_tensor.unsqueeze(0)

        # 4. Feature Extraction
        with torch.no_grad():
            device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(device)
            feat = self.model({"breast_mr": input_tensor})
            vector = feat.squeeze()

        return vector, torch.tensor(label, dtype=torch.long)



# Screenshot Combination 1: Pre-contrast + Post-contrast 1
class PrePost1Dataset(DualPhaseDataset):
    def __init__(self, **kwargs):
        super().__init__(phase_indices=[0, 1], **kwargs)

# Screenshot Combination 2: Post-contrast 1 + Post-contrast 2
class Post1Post2Dataset(DualPhaseDataset):
    def __init__(self, **kwargs):
        super().__init__(phase_indices=[1, 2], **kwargs)