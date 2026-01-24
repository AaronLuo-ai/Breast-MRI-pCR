import torch
from torch.utils.data import Dataset
import nibabel as nib
import os
from rgb_dataset import get_pcr_label, MultiModalAtlas, load_file
import pandas as pd
import yaml
import re
from torchvision import transforms
from PIL import Image


# Regex looks for 'z', then '+' or '-', then digits
def extract_depth(filename):
    match = re.search(r'z([+-]?\d+)', filename)
    return int(match.group(1)) if match else 0


class ThreeDimDataset(Dataset):   
    def __init__(self, feature_path, response_path, model_config_path, checkpoint_path, IMAGE_SIZE = 192, EMBED_DIM = 384, transform = None):
        self.feature_path = feature_path
        self.transform = transform
        
        # 1. Initialize the Model (Feature Extractor)
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
            
        self.model = MultiModalAtlas(args=None, model_config=model_config, embed_dim=EMBED_DIM, multiscale_feats=True)
        state_dict = load_file(checkpoint_path)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 2. Load Excel & Map Labels
        df = pd.read_excel(response_path, keep_default_na=False)
        self.response_data = df[df['MIRRIR_DCE-MRI'].str.contains(r'[a-zA-Z]', na=False)]
        # Dictionary to map 'Complete' -> 1, else -> 0
        label_lookup = self.response_data.set_index("MIRRIR_DCE-MRI")["pCR-Breast"].to_dict()
        all_folders = sorted([f for f in os.listdir(self.feature_path) 
                            if os.path.isdir(os.path.join(self.feature_path, f))])
        unique_elements = sorted(self.response_data["MIRRIR_DCE-MRI"].dropna().unique())
        print(f"Total unique patients in response data: {len(unique_elements)} ")
        self.samples = []

        for element in unique_elements:
            subject_id = str(element).split('_')[0]
            actual_folder = next((f for f in all_folders if f.startswith(subject_id)), None)

            if not actual_folder:
                continue

            subdir_path = os.path.join(self.feature_path, actual_folder)
            phase_folders = sorted([f for f in os.listdir(subdir_path) 
                                if os.path.isdir(os.path.join(subdir_path, f))])

            # 1. Initialize a container for this specific patient
            # We store PATHS here, not actual image arrays, to save memory.
            patient_entry = {"label": get_pcr_label(label_lookup.get(element), element)}
            is_valid_patient = True

            # The 3 required phases
            required_phases = ["PRE__reversedStack", "PO1__reversedStack", "PO2__reversedStack"]
            if not all(phase in phase_folders for phase in required_phases):
                print(f"Error: {actual_folder} missing required phases. Skipping.")
                is_valid_patient = False

            for phase in required_phases:
                phase_path = os.path.join(subdir_path, phase)
                # Use your extract_depth logic to ensure Z-axis order
                files = sorted([f for f in os.listdir(phase_path) if f.endswith('.png')], 
                            key=extract_depth)

                if len(files) != 16:
                    print(f"Error: {actual_folder} Phase {phase} has {len(files)} images. Skipping.")
                    is_valid_patient = False
                    break 

                # Store the list of 16 image paths under the phase name (e.g., "PRE__reversedStack")
                patient_entry[phase] = [os.path.join(phase_path, f) for f in files]

            # 2. Only add to samples if all phases were present and correct
            required_keys = ["label", "PRE__reversedStack", "PO1__reversedStack", "PO2__reversedStack"]
            if is_valid_patient and all(k in patient_entry for k in required_keys):
                self.samples.append(patient_entry)



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError("This is a base class. Use derived classes for specific phase handling.")




class ThreeDimRGBConcat(ThreeDimDataset):
    """
    Handles concatenation of multiple phases as channels for 3D Vision Models
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        # 1. Get the entry for the specific patient
        # Each entry is { "PRE...": [path1, path2...], "PO1...": [...], "label": x }
        patient_entry = self.samples[idx]
        label = patient_entry["label"]

        # We want to extract features for these specific phases in order
        # You can adjust this list based on which phases you want to include
        target_phases = ["PRE__reversedStack", "PO1__reversedStack", "PO2__reversedStack"]
        all_phase_features = []

        # 2. Setup the image transformation
        transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
        ])

        for phase_name in target_phases:
            image_paths = patient_entry[phase_name]
            slice_tensors = []

            # 3. Load and transform each of the 16 slices
            for path in image_paths:
                img = Image.open(path).convert("L")
                img_t = transform(img)  # Shape: [1, 192, 192]
                
                # Replicate to get 3 channels: [3, 192, 192]
                img_rgb = torch.cat([img_t, img_t, img_t], dim=0)
                slice_tensors.append(img_rgb)

            # 4. Stack slices to form a 3D volume [3, 16, 192, 192]
            # (C, D, H, W) format
            phase_volume = torch.stack(slice_tensors, dim=1) 
            
            # Add batch dimension for the model: [1, 3, 16, 192, 192]
            input_tensor = phase_volume.unsqueeze(0)

            # 5. Extract Features using the model (Pillar/Atlas)
            with torch.no_grad():
                device = next(self.model.parameters()).device
                input_tensor = input_tensor.to(device)
                
                # Extract features (adjust key "breast_mr" if your config differs)
                feat = self.model({"breast_mr": input_tensor})
                
                # Squeeze to get a flat vector (e.g., [384])
                all_phase_features.append(feat.squeeze())

        # 6. Concatenate features from all phases
        # If each phase is 384, the final vector is 384 * 3 = 1152
        final_vector = torch.cat(all_phase_features, dim=0)

        return final_vector, torch.tensor(label, dtype=torch.long)





class ThreeDimRGBStack(ThreeDimDataset):
    """
    Handles stacking of multiple phases as channels for 3D Vision Models
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        # 1. Access the paths for the patient
        patient_entry = self.samples[idx]
        label = patient_entry["label"]

        # We map phases to RGB channels: R=PRE, G=PO1, B=PO2
        phase_keys = ["PRE__reversedStack", "PO1__reversedStack", "PO2__reversedStack"]
        
        transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
        ])

        # This list will hold 3 volumes, each [16, 192, 192]
        phase_volumes = []

        for phase_name in phase_keys:
            image_paths = patient_entry[phase_name]
            slice_tensors = []

            # 2. Load and stack the 16 slices for this phase
            for path in image_paths:
                img = Image.open(path).convert("L")
                img_t = transform(img)  # Shape: [1, 192, 192]
                slice_tensors.append(img_t)

            # Create a single-channel volume: [16, 192, 192]
            # Dim 0 is depth (Z-axis) based on your physical order sorting
            vol = torch.cat(slice_tensors, dim=0) 
            phase_volumes.append(vol)

        # 3. Stack phases into RGB Channels
        # phase_volumes[0] -> Red (PRE)
        # phase_volumes[1] -> Green (PO1)
        # phase_volumes[2] -> Blue (PO2)
        # Resulting shape: [3, 16, 192, 192] (C, D, H, W)
        combined_input = torch.stack(phase_volumes, dim=0)

        # 4. Add batch dimension and run through Pillar model
        input_tensor = combined_input.unsqueeze(0) # [1, 3, 16, 192, 192]

        with torch.no_grad():
            device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(device)
            
            # Single forward pass because phases are stacked in channels
            feat = self.model({"breast_mr": input_tensor})
            final_vector = feat.squeeze()

        return final_vector, torch.tensor(label, dtype=torch.long)