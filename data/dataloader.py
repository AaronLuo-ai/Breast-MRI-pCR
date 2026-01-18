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

class MriFeatureDataset(Dataset):

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
        self.response_data = pd.read_excel(response_path)
        # Dictionary to map 'Complete' -> 1, else -> 0
        label_lookup = self.response_data.set_index("MIRRIR_DCE-MRI")["pCR-Breast"].to_dict()
        # 3. Filter for folders that have the required 3 timings
        self.samples = []
        unique_elements = self.response_data["MIRRIR_DCE-MRI"].dropna().unique()
        all_folders = [f for f in os.listdir(self.feature_path) if os.path.isdir(os.path.join(self.feature_path, f))]
        # print(f"Total unique patients in response data: {len(unique_elements)} ")
        for element in unique_elements:
            subject_id = str(element).split('_')[0]
            actual_folder = next((f for f in all_folders if f.startswith(subject_id)), None)

            if actual_folder:
                subdir_path = os.path.join(self.feature_path, actual_folder) 
                
                if os.path.isdir(subdir_path):
                    files = os.listdir(subdir_path)    
                    # Find the specific files for each timing
                    # Using 'lower()' and 'in' to handle cases like 'PRE', 'pre', or 'PRE_reversedStack'
                    pre_file = next((f for f in files if "pre" in f.lower() and f.endswith('.png')), None)
                    po1_file = next((f for f in files if "po1" in f.lower() and f.endswith('.png')), None)
                    po2_file = next((f for f in files if "po2" in f.lower() and f.endswith('.png')), None)

                    if pre_file and po1_file and po2_file:
                        raw_label = label_lookup.get(element)
                        pcr_label = 1 if str(raw_label).strip() == "Complete" else 0
                        
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
        img_paths, label = self.samples[idx]
        
        # 1. Load each image as grayscale ("L")
        # We load them as 1-channel so we can stack them into 3 channels
        pre_img = Image.open(img_paths[0]).convert("L")
        po1_img = Image.open(img_paths[1]).convert("L")
        po2_img = Image.open(img_paths[2]).convert("L")

        # 2. Transform each to 192x192 Tensor
        transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(), # Result: [1, 192, 192]
        ])
        
        pre_t = transform(pre_img)
        po1_t = transform(po1_img)
        po2_t = transform(po2_img)

        # 3. Combine into a single RGB tensor [3, 192, 192]
        # Channel 0 (R) = Pre, Channel 1 (G) = Po1, Channel 2 (B) = Po2
        combined_rgb = torch.cat([pre_t, po1_t, po2_t], dim=0)

        # 4. Prepare for 3D Vision Model [1, 3, 16, 192, 192]
        # Result: [3, 16, 192, 192] (Stretching the 2D image into 16 slices)
        input_tensor = combined_rgb.unsqueeze(1).repeat(1, 16, 1, 1)
        # Result: [1, 3, 16, 192, 192] (Batch dimension)
        input_tensor = input_tensor.unsqueeze(0)

        # 5. Single Forward Pass
        with torch.no_grad():
            device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(device)
            
            # The model now sees temporal changes as colors
            feat = self.model({"breast_mr": input_tensor})
            # Squeeze results in a single vector (likely 384 dimensions)
            vector = feat.squeeze() 

        return vector, torch.tensor(label, dtype=torch.long)



class MriFeatureDataset2(Dataset):

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
        self.response_data = pd.read_excel(response_path)
        # Dictionary to map 'Complete' -> 1, else -> 0
        label_lookup = self.response_data.set_index("MIRRIR_DCE-MRI")["pCR-Breast"].to_dict()
        # 3. Filter for folders that have the required 3 timings
        self.samples = []
        unique_elements = self.response_data["MIRRIR_DCE-MRI"].dropna().unique()
        all_folders = [f for f in os.listdir(self.feature_path) if os.path.isdir(os.path.join(self.feature_path, f))]
        # print(f"Total unique patients in response data: {len(unique_elements)} ")
        for element in unique_elements:
            subject_id = str(element).split('_')[0]
            actual_folder = next((f for f in all_folders if f.startswith(subject_id)), None)

            if actual_folder:
                subdir_path = os.path.join(self.feature_path, actual_folder) 
                
                if os.path.isdir(subdir_path):
                    files = os.listdir(subdir_path)    
                    # Find the specific files for each timing
                    # Using 'lower()' and 'in' to handle cases like 'PRE', 'pre', or 'PRE_reversedStack'
                    pre_file = next((f for f in files if "pre" in f.lower() and f.endswith('.png')), None)
                    po1_file = next((f for f in files if "po1" in f.lower() and f.endswith('.png')), None)
                    po2_file = next((f for f in files if "po2" in f.lower() and f.endswith('.png')), None)

                    if pre_file and po1_file and po2_file:
                        raw_label = label_lookup.get(element)
                        pcr_label = 1 if str(raw_label).strip() == "Complete" else 0
                        
                        paths = [
                            os.path.join(subdir_path, pre_file),
                            os.path.join(subdir_path, po1_file),
                            os.path.join(subdir_path, po2_file)
                        ]
                        print(f"Added patient {element} with label {pcr_label} and files: {paths}\n")
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
        img_paths, label = self.samples[idx]
        phase_vectors = []

        # 1. Define the transform once
        # Note: We convert to RGB because the feature extractor expects 3-channel input
        transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(), # [3, 192, 192] if RGB, [1, 192, 192] if L
        ])

        # 2. Process each phase individually
        for path in img_paths:
            # Convert to RGB to ensure 3 channels for the vision model
            img = Image.open(path).convert("RGB")
            img_t = transform(img) # Shape: [3, 192, 192]

            # 3. Prepare for 3D Vision Model [1, 3, 16, 192, 192]
            # Unsqueeze depth dim and repeat 16 times, then add batch dim
            input_tensor = img_t.unsqueeze(1).repeat(1, 16, 1, 1).unsqueeze(0)

            with torch.no_grad():
                device = next(self.model.parameters()).device
                input_tensor = input_tensor.to(device)
                
                # Extract features for this specific phase
                feat = self.model({"breast_mr": input_tensor})
                phase_vectors.append(feat.squeeze()) 

        # 4. 3456-dim vector
        combined_vector = torch.cat(phase_vectors, dim=0)

        return combined_vector, torch.tensor(label, dtype=torch.long)

def visualize_mri_samples(dataset, num_samples=2):
    """
    Plots the PRE, PO1, and PO2 images for a set number of patients.
    """
    if len(dataset) == 0:
        print("Dataset is empty. Check your paths and filenames.")
        return

    # Set up the figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    plt.subplots_adjust(hspace=0.4)
    
    # If only 1 sample, axes will be a 1D array; make it 2D for consistency
    if num_samples == 1:
        axes = [axes]

    timing_labels = ["PRE", "Post-1 (PO1)", "Post-2 (PO2)"]

    for i in range(num_samples):
        # Get the paths and label from the dataset's internal list
        img_paths, label = dataset.samples[i]
        
        # Extract Patient ID from the folder name (parent of the first image)
        patient_folder = os.path.basename(os.path.dirname(img_paths[0]))
        patient_id = patient_folder.split('_')[0]

        for j in range(3):
            img = Image.open(img_paths[j]).convert("RGB")
            
            axes[i][j].imshow(img)
            axes[i][j].set_title(f"{timing_labels[j]}\n{os.path.basename(img_paths[j])}", fontsize=9)
            axes[i][j].axis('off')
        
        # Add a row-level label for the patient and pCR status
        label_text = "pCR: Complete (1)" if label == 1 else "pCR: Not Complete (0)"
        axes[i][0].text(-10, 100, f"Patient: {patient_id}\n{label_text}", 
                        va='center', ha='right', fontsize=12, fontweight='bold', rotation=90)

    plt.tight_layout()
    plt.savefig("mri_extraction_check.png")
    print(f"Visualization saved as 'mri_extraction_check.png'. You can download this to view it.")
    plt.show()

# --- TESTING SECTION ---
if __name__ == "__main__":
    # 1. Setup paths
    BASE_DIR = "/home/aaron.l"
    dataset = MriFeatureDataset(
        feature_path=f"{BASE_DIR}/FeatureExtraction/MriExtraction/mri_features",
        response_path=f"{BASE_DIR}/VisualizeDir/MultimodalPilotDataset_v1_DEID.xlsx",
        model_config_path=f"{BASE_DIR}/Pillar/model.config.yaml",
        checkpoint_path=f"{BASE_DIR}/Pillar/pillar-pretrain/model.safetensors",
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )
    number_one = 0
    number_zero = 0 
    for _, label in dataset.samples:
        if label == 1:
            number_one += 1
        else:
            number_zero += 1
    print(f"Total samples with label 1: {number_one}")
    print(f"Total samples with label 0: {number_zero}")

    if len(dataset) > 0:
        # 2. Extract one item
        feature_vector, label = dataset[0]

        # 3. Print Results
        print("\n" + "="*30)
        print("DATALOADER OUTPUT CHECK")
        print("="*30)
        print(f"Feature Vector Shape: {feature_vector.shape}")
        print(f"Label:                {label} (Type: {label.dtype})")
        print(f"Number of Samples:     {len(dataset)}")
        print("="*30)
    else:
        print("Dataset is empty. Check your directory paths.")