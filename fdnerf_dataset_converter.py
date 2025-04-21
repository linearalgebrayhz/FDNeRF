#!/usr/bin/env python3
"""
FDNeRF Dataset Converter

This script converts a dataset with expression images, masks, and pose files
into the format expected by FDNeRF.

Usage:
    python fdnerf_dataset_converter.py --input_dir /scratch/network/hy4522/FDNeRF_data/multiface/m--20171024--0000--002757580--GHS --output_dir /scratch/network/hy4522/FDNeRF_data/converted

Requirements:
    - numpy
    - torch
    - PIL (Pillow)
"""

import os
import json
import argparse
import shutil
import numpy as np
import torch
from PIL import Image
import random
from pathlib import Path
import pickle

def setup_argparse():
    parser = argparse.ArgumentParser(description='Convert a dataset to FDNeRF format')
    parser.add_argument('--input_dir', required=True, help='Input directory containing the source dataset')
    parser.add_argument('--output_dir', required=True, help='Output directory for the FDNeRF dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of identities for training')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of identities for validation')
    parser.add_argument('--image_size', type=int, default=256, help='Size of images (assumed square)')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for splitting data')
    parser.add_argument('--list_prefix', type=str, default='mixwild', help='Prefix for train/val/test list files')
    parser.add_argument('--with_fake_tracking', action='store_true', help='Generate fake tracking parameters')
    
    return parser.parse_args()

def create_directory_structure(output_dir):
    """Create the base directory structure for the FDNeRF dataset"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_identities(input_dir):
    """Find all identity directories in the input directory"""
    identities = []
    for item in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, item)) and item.startswith('E'):
            identities.append(item)
    return identities

def load_pose_from_npy(pose_file):
    """Load camera pose from a .npy file"""
    if os.path.exists(pose_file):
        return np.load(pose_file)
    else:
        # Default pose (identity matrix)
        return np.eye(4)

def process_identity_directory(input_dir, identity, output_dir, with_fake_tracking=False, image_size=256):
    """Process a single identity directory"""
    # Create identity directory and subdirectories
    identity_dir = os.path.join(output_dir, identity)
    mixexp_dir = os.path.join(identity_dir, 'mixexp')
    images_masked_dir = os.path.join(mixexp_dir, 'images_masked')
    parsing_dir = os.path.join(mixexp_dir, 'parsing')
    images_3dmm_dir = os.path.join(mixexp_dir, 'images_3dmm')
    
    os.makedirs(images_masked_dir, exist_ok=True)
    os.makedirs(parsing_dir, exist_ok=True)
    os.makedirs(images_3dmm_dir, exist_ok=True)
    
    # Get all images and poses from the source directory
    source_dir = os.path.join(input_dir, identity)
    files = os.listdir(source_dir)
    
    # Filter for image files and organize by ID
    image_files = [f for f in files if f.endswith('.jpg')]
    mask_files = [f for f in files if f.endswith('_mask.png')]
    pose_files = [f for f in files if f.endswith('_pose.npy')]
    
    valid_img_ids = []
    transforms_data = {
        "focal_len": 1000,  # You may want to adjust these values
        "cx": image_size // 2,
        "cy": image_size // 2,
        "near": 8,
        "far": 26,
        "frames": []
    }
    
    # Process each image
    for img_file in sorted(image_files):
        img_id = img_file.split('.')[0]
        valid_img_ids.append(img_id)
        
        # Copy and convert image
        src_img_path = os.path.join(source_dir, img_file)
        dst_img_path = os.path.join(images_masked_dir, f"{img_id}.png")
        
        # Convert JPG to PNG if needed
        if img_file.endswith('.jpg'):
            img = Image.open(src_img_path)
            img.save(dst_img_path)
        else:
            shutil.copy(src_img_path, dst_img_path)
        
        # Copy mask if exists
        mask_file = f"{img_id}_mask.png"
        if mask_file in mask_files:
            src_mask_path = os.path.join(source_dir, mask_file)
            dst_mask_path = os.path.join(parsing_dir, f"{img_id}.png")
            shutil.copy(src_mask_path, dst_mask_path)
        
        # Process pose
        pose_file = f"{img_id}_pose.npy"
        if pose_file in pose_files:
            pose_matrix = load_pose_from_npy(os.path.join(source_dir, pose_file))
            transforms_data["frames"].append({
                "img_id": img_id,
                "transform_matrix": pose_matrix.tolist()
            })
    
    # Write valid_img_ids.txt
    with open(os.path.join(images_3dmm_dir, 'valid_img_ids.txt'), 'w') as f:
        for img_id in valid_img_ids:
            f.write(f"{img_id}\n")
    
    # Write face_transforms_pose.json
    with open(os.path.join(images_3dmm_dir, 'face_transforms_pose.json'), 'w') as f:
        json.dump(transforms_data, f, indent=2)
    
    # Generate tracking parameters
    if with_fake_tracking:
        create_fake_tracking_params(images_3dmm_dir, len(valid_img_ids))

    return identity_dir

def create_fake_tracking_params(output_dir, num_frames):
    """Create fake tracking parameters for testing"""
    # Creating simple tensors for demonstration
    # In a real scenario, these would contain meaningful expression parameters
    euler = torch.zeros((num_frames, 3))  # Rotation angles (pitch, yaw, roll)
    trans = torch.zeros((num_frames, 3))  # Translation (x, y, z)
    exp = torch.zeros((num_frames, 79))   # Expression parameters (can adjust dimension as needed)
    
    # Add some variation to make it look like different expressions
    for i in range(num_frames):
        # Random angles between -15 and 15 degrees
        euler[i] = torch.tensor([
            random.uniform(-0.2, 0.2),  # pitch
            random.uniform(-0.2, 0.2),  # yaw
            random.uniform(-0.1, 0.1)   # roll
        ])
        
        # Random translations between -1 and 1
        trans[i] = torch.tensor([
            random.uniform(-0.5, 0.5),  # x
            random.uniform(-0.5, 0.5),  # y
            random.uniform(-0.5, 0.5)   # z
        ])
        
        # Random expression parameters (simplified)
        exp[i, :] = torch.randn(79) * 0.1
    
    # Multiply by 10 as the code divides by 10 later
    trans = trans * 10.0
    
    # Create the dict and save as torch.pt file
    tracking_params = {
        'euler': euler,
        'trans': trans,
        'exp': exp
    }
    
    torch.save(tracking_params, os.path.join(output_dir, 'track_params.pt'))

    # Also create a dummy 3DMM parameters file
    params_3dmm = {
        'params': {}
    }
    
    for i, frame_idx in enumerate(range(num_frames)):
        # Create a parameter vector that has enough dimensions
        # Typical 3DMM parameters include identity, expression, texture, etc.
        param_vector = np.zeros(257)  # Arbitrary size based on code
        
        # Set different parts of the vector for different meanings
        # 80:144 - expression parameters
        # 224:227 - angle parameters
        # 254:257 - translation parameters
        param_vector[80:144] = np.random.randn(64) * 0.1  # Expression
        param_vector[224:227] = np.random.randn(3) * 0.1  # Angle
        param_vector[254:257] = np.random.randn(3) * 0.1  # Translation
        
        params_3dmm['params'][i] = param_vector
    
    # Save as pickle
    with open(os.path.join(output_dir, 'params_3dmm.pkl'), 'wb') as f:
        pickle.dump(params_3dmm, f)

def create_split_lists(output_dir, identities, train_ratio, val_ratio, list_prefix, random_seed=42):
    """Create train/val/test split lists"""
    random.seed(random_seed)
    random.shuffle(identities)
    
    n_identities = len(identities)
    n_train = int(n_identities * train_ratio)
    n_val = int(n_identities * val_ratio)
    
    train_identities = identities[:n_train]
    val_identities = identities[n_train:n_train+n_val]
    test_identities = identities[n_train+n_val:]
    
    # Write train list
    with open(os.path.join(output_dir, f"{list_prefix}_train.lst"), 'w') as f:
        for identity in train_identities:
            f.write(f"{identity}\n")
    
    # Write val list
    with open(os.path.join(output_dir, f"{list_prefix}_val.lst"), 'w') as f:
        for identity in val_identities:
            f.write(f"{identity}\n")
    
    # Write test list
    with open(os.path.join(output_dir, f"{list_prefix}_test.lst"), 'w') as f:
        for identity in test_identities:
            f.write(f"{identity}\n")

def main():
    args = setup_argparse()
    
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    
    # Create output directory
    output_dir = create_directory_structure(args.output_dir)
    
    # Get all identity directories from input
    identities = get_identities(args.input_dir)
    print(f"Found {len(identities)} identities in {args.input_dir}")
    
    # Process each identity
    processed_identities = []
    for identity in identities:
        print(f"Processing {identity}...")
        identity_dir = process_identity_directory(
            args.input_dir, 
            identity, 
            output_dir, 
            with_fake_tracking=args.with_fake_tracking,
            image_size=args.image_size
        )
        processed_identities.append(identity)
    
    # Create train/val/test split lists
    create_split_lists(
        output_dir, 
        processed_identities, 
        args.train_ratio, 
        args.val_ratio, 
        args.list_prefix,
        args.random_seed
    )
    
    print(f"Conversion complete. Output dataset at {output_dir}")

if __name__ == "__main__":
    main()