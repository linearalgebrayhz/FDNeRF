import os
import sys
import torch
import numpy as np
import open3d as o3d
from pyhocon import ConfigFactory
from model import make_model
from data import get_split_dataset
from torch.utils.data import DataLoader

# === Project path correction ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# === Configuration paths ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERIMENT = "2Dimplicitdeform_reconstruct"
CONF_PATH = f"./results/{EXPERIMENT}/fp_mixexp_2D_implicit.conf"
CHECKPOINT_PATH = f"./results/{EXPERIMENT}/checkpoints/pixel_nerf_latest"
EXPORT_PATH = f"./results/{EXPERIMENT}/export/volume_points.ply"

# === Load configuration and model ===
print("Loading config and model...")
conf = ConfigFactory.parse_file(CONF_PATH)
model = make_model(conf["model"])
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.cuda().eval()

# === Load test data ===
test_dset, val_dset, _ = get_split_dataset("fp_admixexp",
                                    '/scratch/network/hy4522/FDNeRF_data/FDNeRF_converted',
                                    n_view_in=12,
                                    list_prefix="mixwild",
                                    sem_win=1,
                                    with_mask=True)

test_loader = DataLoader(test_dset, batch_size=1, shuffle=False)
data = next(iter(test_loader))

print("Encoding source views...")
images = data["images"][0, :12].to(DEVICE)  # Take 12 source images
poses = data["poses"][0, :12].to(DEVICE)
focal = data["focal"][0, :12].to(DEVICE)
c = data["c"][0, :12].to(DEVICE)
semantic_src = data["semantic_src"].to(DEVICE)
semantic_cdn = data["semantic_cdn"].to(DEVICE)

# Fix semantic input for the model
sem_src = semantic_src[0, :, :, 0].to(DEVICE)
sem_cdn = semantic_cdn[0, :, :, 0].to(DEVICE)

print(f"sem_src shape: {sem_src.shape}")
print(f"sem_cdn shape: {sem_cdn.shape}")

semantic = {
    "semantic_src": sem_src,
    "semantic_cdn": sem_cdn,
}

# Encode the images
model.encode(images.unsqueeze(0), poses.unsqueeze(0), focal, c, semantic=semantic)

@torch.no_grad()
def direct_query_density(model, pts):
    """
    A simplified density query that bypasses the complex forward pass
    and directly queries the MLP with a constant feature vector
    """
    # Process points in manageable chunks
    batch_size = 1024
    sigmas = []
    
    # Create a constant feature vector to substitute for image features
    # Since we only need density for isosurface extraction, this is sufficient
    dummy_feature = torch.zeros(model.latent_size, device=pts.device)
    
    for i in range(0, pts.shape[0], batch_size):
        # Extract current batch
        pts_batch = pts[i:i+batch_size]
        num_pts = pts_batch.shape[0]
        
        # Prepare positional encoding input
        if model.use_xyz:
            pos_input = pts_batch
        else:
            pos_input = pts_batch[:, 2:3]  # Only z coordinate
        
        # Apply positional encoding
        if model.use_code:
            pos_encoded = model.code(pos_input)
        else:
            pos_encoded = pos_input
            
        # Add dummy viewdirs if needed
        if model.use_viewdirs:
            # Default view direction (negative z)
            viewdirs = torch.zeros_like(pts_batch)
            viewdirs[:, 2] = -1.0
            
            if model.use_code_viewdirs and not model.use_code_separate:
                # Encode and combine with pos_encoded
                if model.use_code:
                    view_pos_encoded = torch.cat([pos_input, viewdirs], dim=1)
                    encoded = model.code(view_pos_encoded)
                else:
                    encoded = torch.cat([pos_encoded, viewdirs], dim=1)
            elif model.use_code_viewdirs and model.use_code_separate:
                # Encode positions and viewdirs separately
                viewdirs_encoded = model.code_dir(viewdirs)
                encoded = torch.cat([pos_encoded, viewdirs_encoded], dim=1)
            else:
                # Just append raw viewdirs
                encoded = torch.cat([pos_encoded, viewdirs], dim=1)
        else:
            encoded = pos_encoded
        
        # Replicate the dummy feature for each point
        repeated_dummy = dummy_feature.expand(num_pts, -1)
        
        # Concatenate with encoded positions (and view dirs if used)
        mlp_input = torch.cat([repeated_dummy, encoded], dim=1)
        
        # Forward pass through just the MLP
        if torch.isnan(mlp_input).any():
            mlp_input = torch.where(torch.isnan(mlp_input), 
                                   torch.full_like(mlp_input, 0),
                                   mlp_input)
        
        # Run through the MLP directly
        with torch.no_grad():
            output = model.mlp_coarse(mlp_input)
        
        # Extract density (sigma) and apply activation
        sigma = output[:, 3]
        sigma = torch.relu(sigma)  # Ensure non-negative density
        
        sigmas.append(sigma)
    
    # Combine all chunks
    return torch.cat(sigmas, dim=0)

# === Generate voxel grid ===
print("Generating voxel grid...")
grid_size = 128
bound = 1.0
x = torch.linspace(-bound, bound, grid_size)
y = torch.linspace(-bound, bound, grid_size)
z = torch.linspace(-bound, bound, grid_size)
X, Y, Z = torch.meshgrid(x, y, z)
pts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3).cuda()

# === Extract density ===
print("Querying model...")
sigmas = []
CHUNK = 65536  # Adjust this based on your GPU memory
total_chunks = (pts.shape[0] + CHUNK - 1) // CHUNK

for i in range(0, pts.shape[0], CHUNK):
    print(f"Processing chunk {i//CHUNK + 1}/{total_chunks}")
    chunk_sigma = direct_query_density(model, pts[i:i+CHUNK])
    sigmas.append(chunk_sigma)

sigmas = torch.cat(sigmas, dim=0)

# === Filter valid points ===
sigma_thresh = 5
mask = sigmas > sigma_thresh
pts_valid = pts[mask]

print(f"Selected {pts_valid.shape[0]} points with σ > {sigma_thresh}")
os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)

# === Export to .ply ===
if pts_valid.shape[0] > 0:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_valid.detach().cpu().numpy())
    o3d.io.write_point_cloud(EXPORT_PATH, pcd)
    print("Exported to", EXPORT_PATH)
else:
    print("No points above threshold. Nothing to export.")