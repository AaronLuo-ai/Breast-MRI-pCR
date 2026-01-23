from rgb_dataset import RGBStackDataset, RGBDataset
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os



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

def debug_mri_dataloader(dataset, num_to_check=5):
    """
    Prints the file paths and labels stored in the dataset's sample list.
    """
    print(f"\n{'='*30} DATASET AUDIT {'='*30}")
    print(f"Total samples found: {len(dataset)}")
    print(f"{'Index':<5} | {'Label':<6} | {'Paths (Pre, Po1, Po2)'}")
    print("-" * 80)

    for i in range(min(num_to_check, len(dataset))):
        img_paths, label = dataset.samples[i]
        
        # Format the paths to show only the patient folder and filename for readability
        readable_paths = [os.path.join(p.split('/')[-2], p.split('/')[-1]) for p in img_paths]
        
        label_str = "1 (pCR)" if label == 1 else "0 (Non-pCR)"
        
        print(f"{i:<5} | {label_str:<10} | {readable_paths[0]}")
        print(f"{'':<5} | {'':<10} | {readable_paths[1]}")
        print(f"{'':<5} | {'':<10} | {readable_paths[2]}")
        print("-" * 80)


# --- TESTING SECTION ---
if __name__ == "__main__":
    # 1. Setup paths
    BASE_DIR = "/home/aaron.l"
    dataset = RGBStackDataset(
        feature_path=f"{BASE_DIR}/FeatureExtraction/MriExtraction/mri_features",
        response_path=f"{BASE_DIR}/FeatureExtraction/MriExtraction/MultimodalPilotDataset_v1_DEID_wImages_final.xlsx",
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
    
    debug_mri_dataloader(dataset, num_to_check=len(dataset))