import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import yaml
import pandas as pd
from safetensors.torch import load_file
from model.model_loader import load_model
from data.rgb_dataset import RGBConcatDataset


# --- Base Class to handle the heavy lifting ---
class SinglePhaseDataset(RGBConcatDataset):
    """
    Inherits from your existing RGBConcatDataset to reuse the
    folder filtering and model loading logic.
    """

    def __init__(self, phase_index, *args, **kwargs):
        # phase_index: 0 for PRE, 1 for PO1, 2 for PO2
        super().__init__(*args, **kwargs)
        self.phase_index = phase_index

    def __getitem__(self, idx):
        img_paths, label = self.samples[idx]

        # 1. Load ONLY the target phase image
        target_path = img_paths[self.phase_index]
        img = Image.open(target_path).convert("L")

        # 2. Transform to Tensor
        transform = transforms.Compose(
            [
                transforms.Resize((192, 192)),
                transforms.ToTensor(),
            ]
        )
        img_t = transform(img)  # [1, 192, 192]

        # 3. Create a 3-channel version of the SAME image
        # Stacking the same phase 3 times [3, 192, 192]
        combined_rgb = torch.cat([img_t, img_t, img_t], dim=0)

        # 4. Prepare for 3D Vision Model [1, 3, 16, 192, 192]
        input_tensor = combined_rgb.unsqueeze(1).repeat(1, 16, 1, 1)
        input_tensor = input_tensor.unsqueeze(0)

        # 5. Feature Extraction
        with torch.no_grad():
            device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(device)
            feat = self.model({"breast_mr": input_tensor})
            vector = feat.squeeze()

        return vector, torch.tensor(label, dtype=torch.long)


# --- Specific Dataset Classes ---


class PreOnlyDataset(SinglePhaseDataset):
    def __init__(self, **kwargs):
        super().__init__(phase_index=0, **kwargs)


class Po1OnlyDataset(SinglePhaseDataset):
    def __init__(self, **kwargs):
        super().__init__(phase_index=1, **kwargs)


class Po2OnlyDataset(SinglePhaseDataset):
    def __init__(self, **kwargs):
        super().__init__(phase_index=2, **kwargs)
