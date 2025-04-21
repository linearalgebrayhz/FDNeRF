import os
import shutil
import argparse
from pathlib import Path

def restructure_fdnerf_dataset(source_dir, dest_dir):
    """
    Restructures FDNeRF dataset to match expected format, creating directories
    for each image set.
    
    Args:
        source_dir: Directory containing the original data files (E057_Cheeks_Puffed, etc.)
        dest_dir: Directory where the restructured data will be placed
    """
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Find all image files in the dataset
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.jpg') and not file.endswith('_mask.jpg'):
                # Extract base filename without extension
                base_name = os.path.splitext(file)[0]
                
                # For E057_all files (the prefix will be in the filename)
                if 'E057_all' in root:
                    # Create directory for this image set
                    sample_dir = os.path.join(dest_dir, base_name)
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # Copy files (image, mask, pose)
                    full_path = os.path.join(root, file)
                    mask_path = os.path.join(root, base_name + "_mask.png")
                    pose_path = os.path.join(root, base_name + "_pose.npy")
                    
                    # Copy files to new structure
                    shutil.copy(full_path, os.path.join(sample_dir, "image.jpg"))
                    if os.path.exists(mask_path):
                        shutil.copy(mask_path, os.path.join(sample_dir, "mask.png"))
                    if os.path.exists(pose_path):
                        shutil.copy(pose_path, os.path.join(sample_dir, "pose.npy"))
                
                # For other directories (like E057_Cheeks_Puffed)
                else:
                    expression = os.path.basename(root).split('_')[1:]
                    expression = '_'.join(expression)
                    
                    # Create directory name by combining expression and base filename
                    dir_name = f"{expression}_{base_name}"
                    sample_dir = os.path.join(dest_dir, dir_name)
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # Copy files (image, mask, pose)
                    full_path = os.path.join(root, file)
                    mask_path = os.path.join(root, base_name + "_mask.png")
                    pose_path = os.path.join(root, base_name + "_pose.npy")
                    
                    # Copy files to new structure
                    shutil.copy(full_path, os.path.join(sample_dir, "image.jpg"))
                    if os.path.exists(mask_path):
                        shutil.copy(mask_path, os.path.join(sample_dir, "mask.png"))
                    if os.path.exists(pose_path):
                        shutil.copy(pose_path, os.path.join(sample_dir, "pose.npy"))

    # Copy meta.json if it exists
    meta_paths = [
        os.path.join(source_dir, "E057_all", "meta.json"),
        os.path.join(source_dir, "E057_Cheeks_Puffed", "meta.json"),
        os.path.join(source_dir, "E061_Lips_Puffed", "meta.json")
    ]
    
    for meta_path in meta_paths:
        if os.path.exists(meta_path):
            shutil.copy(meta_path, os.path.join(dest_dir, "meta.json"))
            print(f"Copied meta.json from {meta_path}")
            break

    print(f"Dataset restructuring complete. New structure created in {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restructure FDNeRF dataset to match expected format")
    parser.add_argument("--source_dir", type=str, required=True, 
                      help="Path to original dataset directory")
    parser.add_argument("--dest_dir", type=str, required=True,
                      help="Path where restructured dataset will be created")
    
    args = parser.parse_args()
    restructure_fdnerf_dataset(args.source_dir, args.dest_dir)