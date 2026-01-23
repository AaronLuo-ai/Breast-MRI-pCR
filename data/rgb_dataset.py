import torch
import sys
import os
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from safetensors.torch import load_file
external_path = "/home/aaron.l/Pillar/pillar-pretrain"
if external_path not in sys.path:
    sys.path.insert(0, external_path)
from src.miniclip.multimodal_atlas import MultiModalAtlas



IMAGE_SIZE = 192

def get_pcr_label(raw_label, patient_id):
    # 1. Check for TRULY blank cells (now that 'None' is preserved as text)
    if pd.isna(raw_label) or raw_label == "":
        print(f"Excluding {patient_id}: Cell is actually empty")
        return None

    # 2. Manual ID Exclusions
    exclusions = ["M83641802_20231010175132", "M44963301_20221129183544", "M46018541_20210421154917", "M16589234_20230809172111", "M31136020_20211228182422"]
    if patient_id in exclusions:
        return None

    response_clean = str(raw_label).strip().lower()

    # 3. Priority logic: Check Incomplete/None first
    if "incomplete" in response_clean or response_clean == "none" or patient_id == "M19653719_20200313081324":
        return 0
    
    # 4. Check Complete/DCIS second
    elif "complete" in response_clean or "dcis" in response_clean:
        return 1
    
    else:
        return None

        
class RGBDataset(Dataset):
    def __init__(self, feature_path, response_path, model_config_path, checkpoint_path, transform=None):
        self.feature_path = feature_path
        self.transform = transform
        
        # 1. Initialize the Model (Feature Extractor)
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
            
        self.model = MultiModalAtlas(args=None, model_config=model_config, embed_dim=384, multiscale_feats=True)
        
        # Load weights
        state_dict = load_file(checkpoint_path)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        # 2. Load Excel & Map Labels
        df = pd.read_excel(response_path, keep_default_na=False)
        self.response_data = df[df['MIRRIR_DCE-MRI'].str.contains(r'[a-zA-Z]', na=False)]
        # Dictionary to map 'Complete' -> 1, else -> 0
        label_lookup = self.response_data.set_index("MIRRIR_DCE-MRI")["pCR-Breast"].to_dict()
        # 3. Filter for folders that have the required 3 timings
        self.samples = []
        unique_elements = sorted(self.response_data["MIRRIR_DCE-MRI"].dropna().unique())
        print(f"Total unique patients in response data: {len(unique_elements)} ")
        all_folders = sorted([f for f in os.listdir(self.feature_path) if os.path.isdir(os.path.join(self.feature_path, f))])        # print(f"Total unique patients in response data: {len(unique_elements)} ")
        for element in unique_elements:
            subject_id = str(element).split('_')[0]
            actual_folder = next((f for f in all_folders if f.startswith(subject_id)), None)

            if actual_folder:
                subdir_path = os.path.join(self.feature_path, actual_folder) 
                
                if os.path.isdir(subdir_path):
                    files = sorted(os.listdir(subdir_path))    
                    # Find the specific files for each timing
                    # Using 'lower()' and 'in' to handle cases like 'PRE', 'pre', or 'PRE_reversedStack'
                    pre_file = next((f for f in files if "pre" in f.lower() and f.endswith('.png')), None)
                    po1_file = next((f for f in files if "po1" in f.lower() and f.endswith('.png')), None)
                    po2_file = next((f for f in files if "po2" in f.lower() and f.endswith('.png')), None)

                    if pre_file and po1_file and po2_file:
                        raw_label = label_lookup.get(element)
                        if raw_label is None:
                            print(f"Warning: No label found for patient {element}")
                            continue
                        
                        pcr_label = get_pcr_label(raw_label, element)
                        if pcr_label is None:
                            continue
                        paths = [
                            os.path.join(subdir_path, pre_file),
                            os.path.join(subdir_path, po1_file),
                            os.path.join(subdir_path, po2_file)
                        ]
                        # print(f"Added patient {element} with label {pcr_label} and files: {paths}\n")
                        self.samples.append((paths, pcr_label))
                    else:
                        print(f"Skipping patient {element}: Missing required phases. Found - PRE: {pre_file}, PO1: {po1_file}, PO2: {po2_file}\n")
                else:
                    print(f"Directory for patient {element} does not exist at path: {subdir_path}\n")
            else:
                print(f"No matching folder found for patient {element} in feature path.\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # This forces the Child classes to implement their own version
        raise NotImplementedError("Subclasses must implement __getitem__")











class RGBConcatDataset(RGBDataset):
    def __init__(self, feature_path, response_path, model_config_path, checkpoint_path, transform=None):
        super().__init__(feature_path, response_path, model_config_path, checkpoint_path, transform)
    
    def __getitem__(self, idx):
        img_paths, label = self.samples[idx]
        
        # 1. Load each image as grayscale ("L")
        # We load them as 1-channel so we can stack them into 3 channels
        pre_img = Image.open(img_paths[0]).convert("L")
        po1_img = Image.open(img_paths[1]).convert("L")
        po2_img = Image.open(img_paths[2]).convert("L")

        # 2. Transform each to IMAGE_SIZE x IMAGE_SIZE Tensor
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(), # Result: [1, IMAGE_SIZE, IMAGE_SIZE]
        ])
        
        pre_t = transform(pre_img)
        po1_t = transform(po1_img)
        po2_t = transform(po2_img)

        # 3. Combine into a single RGB tensor [3, IMAGE_SIZE, IMAGE_SIZE]
        # Channel 0 (R) = Pre, Channel 1 (G) = Po1, Channel 2 (B) = Po2
        combined_rgb = torch.cat([pre_t, po1_t, po2_t], dim=0)

        # 4. Prepare for 3D Vision Model [1, 3, 16, IMAGE_SIZE, IMAGE_SIZE]
        # Result: [3, 16, IMAGE_SIZE, IMAGE_SIZE] (Stretching the 2D image into 16 slices)
        input_tensor = combined_rgb.unsqueeze(1).repeat(1, 16, 1, 1)
        # Result: [1, 3, 16, IMAGE_SIZE, IMAGE_SIZE] (Batch dimension)
        input_tensor = input_tensor.unsqueeze(0)

        # 5. Single Forward Pass
        with torch.no_grad():
            device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(device)
            print(f"Input tensor shape for combined RGB: {input_tensor.shape}")
            # The model now sees temporal changes as colors
            feat = self.model({"breast_mr": input_tensor})
            # Squeeze results in a single vector (likely 384 dimensions)
            vector = feat.squeeze() 

        return vector, torch.tensor(label, dtype=torch.long)










class RGBStackDataset(RGBDataset):

    def __init__(self, feature_path, response_path, model_config_path, checkpoint_path, transform=None):
        super().__init__(feature_path, response_path, model_config_path, checkpoint_path, transform)
    
    def __getitem__(self, idx):
        img_paths, label = self.samples[idx]
        phase_vectors = []

        # 1. Define the transform once
        # Note: We convert to RGB because the feature extractor expects 3-channel input
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(), # [3, IMAGE_SIZE, IMAGE_SIZE] if RGB, [1, IMAGE_SIZE, IMAGE_SIZE] if L
        ])

        # 2. Process each phase individually
        for path in img_paths:
            # Convert to RGB to ensure 3 channels for the vision model
            img = Image.open(path).convert("RGB")
            img_t = transform(img) # Shape: [3, IMAGE_SIZE, IMAGE_SIZE]

            # 3. Prepare for 3D Vision Model [1, 3, 16, IMAGE_SIZE, IMAGE_SIZE]
            # Unsqueeze depth dim and repeat 16 times, then add batch dim
            input_tensor = img_t.unsqueeze(1).repeat(1, 16, 1, 1).unsqueeze(0)

            with torch.no_grad():
                device = next(self.model.parameters()).device
                input_tensor = input_tensor.to(device)
                print(f"Input tensor shape for phase at: {input_tensor.shape}")
                # Extract features for this specific phase
                feat = self.model({"breast_mr": input_tensor})
                phase_vectors.append(feat.squeeze()) 

        # 4. 3456-dim vector
        combined_vector = torch.cat(phase_vectors, dim=0)

        return combined_vector, torch.tensor(label, dtype=torch.long)