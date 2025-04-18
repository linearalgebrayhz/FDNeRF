import os
import sys
import torch
import numpy as np
import open3d as o3d
from pyhocon import ConfigFactory
from model import make_model
from data import get_split_dataset
from torch.utils.data import DataLoader
# from dataset import get_dataset  # 确保你有这个函数或替换为你的导入方式

# === 项目路径修正 ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# === 配置路径 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERIMENT = "2Dimplicitdeform_reconstruct"
CONF_PATH = f"./results/{EXPERIMENT}/fp_mixexp_2D_implicit.conf"
CHECKPOINT_PATH = f"./results/{EXPERIMENT}/checkpoints/pixel_nerf_latest"
EXPORT_PATH = f"./results/{EXPERIMENT}/export/volume_points.ply"

# === 加载配置和模型 ===
print("Loading config and model...")
conf = ConfigFactory.parse_file(CONF_PATH)
model = make_model(conf["model"])
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.cuda().eval()

# === 加载测试数据 ===
test_dset, val_dset, _ = get_split_dataset("fp_admixexp",
                                      '/scratch/network/hy4522/FDNeRF_data/FDNeRF_converted',
                                      n_view_in=12,
                                      list_prefix="mixwild",
                                      sem_win=1,
                                      with_mask=True)

test_loader = DataLoader(test_dset, batch_size=1, shuffle=False)
data = next(iter(test_loader))

print("Encoding source views...")
images = data["images"][0, :12].to(DEVICE)  # 取13张source图
poses = data["poses"][0, :12].to(DEVICE)
focal = data["focal"][0, :12].to(DEVICE)
c = data["c"][0, :12].to(DEVICE)
semantic_src = data["semantic_src"].to(DEVICE)
semantic_cdn = data["semantic_cdn"].to(DEVICE)

semantic = {
    "semantic_src": semantic_src[0][:,:,0],  # torch.Size([12,85,27])
    "semantic_cdn": semantic_cdn[0][:,:,0],  # torch.Size([12,85,27])
}
print(f"images: {images.shape}")
print(f"poses: {poses.shape}")
print(f"c: {c.shape}")
print(f"focal: {focal.shape}")
print(f"semantic_src shape: {semantic_src.shape}")
print(f"semantic_src[0][:,:,0] shape: {semantic_src[0][:,:,0].shape}")
print(f"semantic['semantic_src'] shape: {semantic['semantic_src'].shape}")

model.encode(images.unsqueeze(0), poses.unsqueeze(0), focal, c, semantic=semantic)

# === 生成体素网格 ===
grid_size = 128
bound = 1.0
x = torch.linspace(-bound, bound, grid_size)
y = torch.linspace(-bound, bound, grid_size)
z = torch.linspace(-bound, bound, grid_size)
X, Y, Z = torch.meshgrid(x, y, z)
pts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3).cuda()

def repeat_interleave(x, num_repeat):
    return x.repeat_interleave(num_repeat, dim=0)

# === 提取密度 ===
@torch.no_grad()
def query_density(model, pts):
    B = pts.shape[0]
    SB = 1
    NS = model.num_views_per_obj 
    
    pts = pts.view(SB, B, 3)
    
    viewdirs = torch.zeros_like(pts)
    viewdirs[..., 2] = -1.0
    
    all_sigmas = []
    
    for view_idx in range(NS):

        temp_poses = model.poses[view_idx:view_idx+1]
        temp_focal = model.focal[view_idx:view_idx+1]
        temp_c = model.c[view_idx:view_idx+1]
        
        view_pts = pts.clone()
        pts_rot = torch.matmul(temp_poses[:, None, :3, :3], view_pts.unsqueeze(-1))[..., 0]
        # Apply translation
        pts_cam = pts_rot + temp_poses[:, None, :3, 3]
        
        uv = -pts_cam[:, :, :2] / (pts_cam[:, :, 2:] + 1e-7)
        uv *= temp_focal.unsqueeze(1)
        uv += temp_c.unsqueeze(1)
    
        latent = model.encoder.index(
            uv,
            None,
            model.image_shape,
            freeze_enc=model.stop_encoder_grad
        ).transpose(1, 2)  # Transform to [SB, B, latent]
        
        if model.use_xyz:
            if model.normalize_z:
                pts_norm = pts_cam - temp_poses[:, None, :3, 3]
                z_feature = pts_norm.reshape(-1, 3)
            else:
                z_feature = pts_cam.reshape(-1, 3)
        else:
            if model.normalize_z:
                pts_norm = pts_cam - temp_poses[:, None, :3, 3]
                z_feature = -pts_norm[..., 2].reshape(-1, 1)
            else:
                z_feature = -pts_cam[..., 2].reshape(-1, 1)
        
        # Apply positional encoding if needed
        if model.use_code and not model.use_code_viewdirs:
            z_feature = model.code(z_feature)
            
        if model.use_viewdirs:
            view_dirs = viewdirs.reshape(SB, B, 3, 1)
            if not model.use_world_dirs:
                view_dirs = torch.matmul(temp_poses[:, None, :3, :3], view_dirs)
            view_dirs = view_dirs.reshape(-1, 3)
            
            if not model.use_code_separate:
                z_feature = torch.cat((z_feature, view_dirs), dim=1)

        # Apply second encoding if needed
        if model.use_code and model.use_code_viewdirs:
            z_feature = model.code(z_feature)
        if model.use_viewdirs and model.use_code_separate:
            dir_feature = model.code_dir(view_dirs)
            z_feature = torch.cat((z_feature, dir_feature), dim=1)
        
        # Combine latent and z_feature
        latent = latent.reshape(-1, model.latent_size)
        mlp_input = torch.cat((latent, z_feature), dim=-1) if model.d_in > 0 else latent
        
        # Run through MLP
        output = model.mlp_coarse(mlp_input)
        
        # Extract sigma and add to our collection
        sigma = output[..., 3:4]
        all_sigmas.append(sigma)

    # Aggregate sigmas from all views (e.g., mean, max)
    sigmas = torch.stack(all_sigmas, dim=0)
    # Can use mean, max, or other aggregation methods
    final_sigma = torch.mean(sigmas, dim=0)
    
    return final_sigma.view(B)  # 取 σ 通道

print("Querying model...")
sigmas = []
CHUNK = 65536
for i in range(0, pts.shape[0], CHUNK):
    chunk_sigma = query_density(model, pts[i:i+CHUNK])
    sigmas.append(chunk_sigma)
sigmas = torch.cat(sigmas, dim=0)

# === 过滤有效点 ===
sigma_thresh = 10.0
mask = sigmas > sigma_thresh
pts_valid = pts[mask]

print(f"Selected {pts_valid.shape[0]} points with σ > {sigma_thresh}")
os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)

# === 导出为 .ply ===
if pts_valid.shape[0] > 0:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_valid.detach().cpu().numpy())
    o3d.io.write_point_cloud(EXPORT_PATH, pcd)
    print("Exported to", EXPORT_PATH)
else:
    print("No points above threshold. Nothing to export.")