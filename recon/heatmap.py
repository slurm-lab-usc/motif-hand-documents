import sys
import os
current_file_dir = os.path.dirname(__file__)

import glob
import cv2
workspace_dir = os.path.dirname(current_file_dir)

sys.path.append(workspace_dir)

import torch
import pypose as pp

import open3d as o3d
from scipy.spatial.transform import Rotation as R
import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from vis.utils.colmap import Dataset, Parser
from vis.utils.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from vis.utils.misc import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from vis.utils.lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    fully_fused_projection_2dgs,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    rasterize_to_pixels_2dgs,
    spherical_harmonics,
)
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam


def rotate_quat(points, rot_vecs):
    """
    Rotate 3D points by a pypose SE3 (tx, ty, tz, qw, qx, qy, qz).
    `rot_vecs` can be shape (7,) for a single transform or (B,7) for batches.
    """
    transform = pp.SE3(rot_vecs)
    return transform.Act(points)


def project(points, camera_params, camera_model=None):
    """
    Projects 3D points in world frame into 2D pixel coordinates using
    the final camera_params = [tx, ty, tz, qw, qx, qy, qz, fx, cx, cy].

    This sample assumes:
       - A pinhole camera model with focal length fx = fy.
       - The principal point is at (cx, cy).
    """
    # 1) Transform points to camera coordinates (rotation + translation).
    points_cam = rotate_quat(points, camera_params[..., :7])  # shape: (N, 3)

    # 2) Divide the first two coords by the depth (3rd coord).
    xy = points_cam[..., :2]
    z = points_cam[..., 2].unsqueeze(-1)
    proj_2d = xy / z  # shape: (N, 2)

    # 3) Apply focal length and principal point shift.
    fx = camera_params[..., -3].unsqueeze(-1)  # shape: (1,)
    cx = camera_params[..., -2].unsqueeze(-1)  # shape: (1,)
    cy = camera_params[..., -1].unsqueeze(-1)  # shape: (1,)

    proj_2d = proj_2d * fx + torch.cat([cx, cy], dim=-1)  # shape: (N, 2)
    return proj_2d


def reproject_simple_pinhole(points, camera_params):
    points_proj = rotate_quat(points, camera_params[..., :7])
    points_proj = points_proj[..., :2] / points_proj[..., 2].unsqueeze(-1)
    f = camera_params[..., -3].unsqueeze(-1)
    pp_ = camera_params[..., -2:]
    points_proj = points_proj * f + pp_
    return points_proj


def assign_colors_to_points(
    points,       # (N, 3) in world coordinates
    intrinsics,   # (3, 3)
    extrinsic_raw,    # (4, 4) transform from world -> camera
    image,         # (H, W, 3), color image
    features_dc, features_extra, opacities, scales, rots
) -> torch.Tensor:
    """
    For each 3D point, project it into the image. If the resulting pixel is
    within the image bounds and its color is non-zero, assign that color
    to the point. Otherwise the point color is (0, 0, 0).

    Returns:
        color_array: (N, 3) float Tensor of RGB colors.
                     Points out-of-bounds, behind the camera,
                     or landing on a zero-pixel remain (0,0,0).
    """
    device = points.device
    H, W = image.shape[:2]
    points = points.to(torch.float64)
    N = points.shape[0]
    color_array = torch.zeros((N, 3), dtype=image.dtype, device=device)
    extrinsic = extrinsic_raw.to(torch.float32)
    intrinsics = intrinsics.to(torch.float32)
    extrinsic_numpy = extrinsic.cpu().detach().numpy()

    points = torch.tensor(points).to("cuda").to(torch.float32)
    rots = torch.tensor(rots).to("cuda").to(torch.float32)
    scales = torch.tensor(scales).to("cuda").to(torch.float32)
    covars = torch.ones((points.shape[0], 6)).to("cuda").to(torch.float32)

    proj_results = fully_fused_projection(
        points,
        covars,
        rots,
        scales,
        torch.linalg.inv(extrinsic),
        intrinsics,
        W,
        H,
    )

    radii, means, depth, _, _ = proj_results
    px = means[0, :, 0]
    py = means[0, :, 1]

    in_bounds = (
        (px >= 0) & (px < W) &
        (py >= 0) & (py < H)
    )
    px_in = px[in_bounds]
    py_in = py[in_bounds]
    in_bounds_indices = torch.where(in_bounds)[0]

    for i in range(px_in.shape[0]):
        x_i = int(px_in[i])
        y_i = int(py_in[i])
        pt_idx = in_bounds_indices[i]

        pixel_val = image[y_i, x_i]  # shape: (3,)
        color_array[pt_idx] = pixel_val

    return color_array


from plyfile import PlyData, PlyElement


def load_ply(path, max_sh_degree=3):
    plydata = PlyData.read(path)

    xyz = np.stack(
        (np.array(plydata.elements[0]["x"]),
         np.array(plydata.elements[0]["y"]),
         np.array(plydata.elements[0]["z"])),
        axis=1,
    )
    opacities = np.array(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
    features_dc[:, 0, 0] = np.array(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.array(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.array(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.array(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.array(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.array(plydata.elements[0][attr_name])

    return xyz, features_dc, features_extra, opacities, scales, rots


def load_heatmap(folder_path, save_path):
    """
    Load, rotate, convert to RGB, pad, and save images from 'folder_path'.
    Each image is rotated 90 degrees CCW and padded to 3840x1260 with blue background.
    Returns a list of NumPy arrays (RGB images).
    """
    exts = ("*.png", "*.jpg", "*.jpeg")
    heatmaps = []

    os.makedirs(save_path, exist_ok=True)

    for ext in exts:
        pattern = os.path.join(folder_path, ext)
        for file_path in sorted(glob.glob(pattern)):
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)

            if img is None:
                print(f"Warning: Could not read image: {file_path}")
                continue

            heatmaps.append(img)

    return heatmaps


def save_ply(points, colors, out_path):
    """
    Save a colored point cloud to a .ply file (ASCII format).
    """
    assert points.shape[0] == colors.shape[0], "Points and colors must have the same number of vertices."
    assert points.shape[1] == 3, "Points should be of shape (N, 3)."
    assert colors.shape[1] == 3, "Colors should be of shape (N, 3)."

    num_points = points.shape[0]

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]

    with open(out_path, 'w') as f:
        for line in header:
            f.write(line + "\n")

        for i in range(num_points):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

    print(f"Saved PLY: {out_path}")


def vote_color_exclude_zeros(global_colors_list):
    """
    Performs a majority vote on the colors for each pixel with special handling:
    - If a pixel has more than 2 votes with high red values (R channel > 200), it is set to pure red ([255, 0, 0]).
    - If all M votes for a pixel are zeros, the result is zero.
    - If there is at least one non-zero color, all zero votes are ignored, and the majority vote
      is computed from the non-zero colors.
    """
    M, N, _ = global_colors_list.shape
    voted_color = np.zeros((N, 3), dtype=global_colors_list.dtype)

    for n in range(N):
        votes = global_colors_list[:, n, :]

        red_votes = np.sum(votes[:, 0] > 200)
        if red_votes > 2:
            voted_color[n] = np.array([255, 0, 0], dtype=global_colors_list.dtype)
            continue

        non_zero_mask = ~np.all(votes == 0, axis=1)

        if np.any(non_zero_mask):
            votes = votes[non_zero_mask]

        if votes.shape[0] == 0:
            voted_color[n] = np.zeros(3, dtype=global_colors_list.dtype)
        else:
            unique_colors, counts = np.unique(votes, axis=0, return_counts=True)
            majority_index = np.argmax(counts)
            voted_color[n] = unique_colors[majority_index]

    return voted_color


def rescale_red_green(voted_color, stddev=10, final_color_variation=50):
    """
    Remaps colors:
    - Red pixels (red dominant) are changed to greenish-cyan with random variation.
    - All other pixels are changed to red with random variation.
    - Adds additional random variation to the final color output.
    """
    result = voted_color.copy().astype(np.int32)

    target_cyanish_green = np.array([0, 255, 200], dtype=np.int32)
    target_red = np.array([255, 0, 0], dtype=np.int32)

    red_mask = (voted_color[:, 0] > voted_color[:, 1]) & (voted_color[:, 0] > voted_color[:, 2])
    non_red_mask = ~red_mask

    num_red = np.count_nonzero(red_mask)
    if num_red > 0:
        noise = np.random.normal(0, stddev, (num_red, 3))
        new_green_colors = np.clip(target_cyanish_green + noise, 0, 255)
        result[red_mask] = new_green_colors.astype(np.int32)

    num_non_red = np.count_nonzero(non_red_mask)
    if num_non_red > 0:
        noise = np.random.normal(0, stddev, (num_non_red, 3))
        new_red_colors = np.clip(target_red + noise, 0, 255)
        result[non_red_mask] = new_red_colors.astype(np.int32)

    final_noise = np.random.normal(0, final_color_variation, result.shape)
    result = np.clip(result + final_noise, 0, 255)

    return result.astype(np.uint8)

# ----------------------
# CLI usage
# ----------------------
if __name__ == "__main__":
    import argparse
    import imageio.v2 as imageio
    import tqdm
    import pypose as pp

    arg_parser = argparse.ArgumentParser(
        description="Project thermal images onto 3D Gaussian points and recolor a point cloud."
    )

    # 3D Gaussian points (means) PLY
    arg_parser.add_argument(
        "--points_ply",
        type=str,
        required=True,
        help="Path to Gaussian point cloud PLY (e.g., shortcanpoints.ply).",
    )

    # COLMAP / nerf-style result folder
    arg_parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to COLMAP/NeRF result folder used by vis.utils.colmap.Parser.",
    )

    # Heatmap input/output dirs
    arg_parser.add_argument(
        "--heatmap_dir",
        type=str,
        required=True,
        help="Folder containing input heatmap images (e.g., output_images).",
    )
    arg_parser.add_argument(
        "--heatmap_save_dir",
        type=str,
        default=None,
        help="Folder to save processed heatmaps (default: same as --heatmap_dir).",
    )

    # Existing colored point cloud (before recoloring)
    arg_parser.add_argument(
        "--input_colored_pcd",
        type=str,
        required=True,
        help="Path to input colored point cloud PLY (e.g., colored_cloud_final_raw_K.ply).",
    )

    # Output location / name
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for the final recolored PLY (default: directory of --input_colored_pcd).",
    )
    arg_parser.add_argument(
        "--output_name",
        type=str,
        default="heat.ply",
        help="Output PLY file name (default: heat.ply).",
    )

    # Parser / trajectory configs
    arg_parser.add_argument(
        "--factor",
        type=int,
        default=1,
        help="Downsample factor for Parser (default: 1).",
    )
    arg_parser.add_argument(
        "--test_every",
        type=int,
        default=8,
        help="Parser test_every parameter (default: 8).",
    )

    # Color remapping hyperparams
    arg_parser.add_argument(
        "--stddev",
        type=float,
        default=10.0,
        help="Stddev of noise when mapping to red/green (default: 10.0).",
    )
    arg_parser.add_argument(
        "--final_color_variation",
        type=float,
        default=50.0,
        help="Stddev of final random color variation (default: 50.0).",
    )

    args = arg_parser.parse_args()

    # Default heatmap_save_dir = heatmap_dir if not provided
    if args.heatmap_save_dir is None:
        args.heatmap_save_dir = args.heatmap_dir

    # Default output_dir = directory of input_colored_pcd if not provided
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.input_colored_pcd))
    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------
    # 1) Load Gaussian point means
    # ----------------------
    points_world, features_dc, features_extra, opacities, scales, rots = load_ply(
        args.points_ply
    )

    # ----------------------
    # 2) Build dataset / loader
    # ----------------------
    colmap_parser = Parser(
        data_dir=args.data_dir,
        factor=args.factor,
        normalize=True,
        test_every=args.test_every,
    )
    valset = Dataset(colmap_parser, split="train")
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=1, shuffle=False, num_workers=1
    )

    # ----------------------
    # 3) Load heatmap images
    # ----------------------
    images = load_heatmap(args.heatmap_dir, save_path=args.heatmap_save_dir)

    # ----------------------
    # 4) Project heatmaps onto 3D points and accumulate per-view colors
    # ----------------------
    global_colors_list = []
    for i, data in enumerate(valloader):
        camtoworlds = data["camtoworld"]  # [1, 4, 4] or similar
        Ks = data["K"]                    # [1, 3, 3]
        image = images[i]                 # (H, W, 3) uint8

        points_world_torch = torch.from_numpy(points_world).to("cuda")
        image_torch = torch.from_numpy(image).to("cuda")

        point_colors = assign_colors_to_points(
            points_world_torch,
            Ks.to("cuda"),
            camtoworlds.to("cuda"),
            image_torch,
            features_dc,
            features_extra,
            opacities,
            scales,
            rots,
        )
        colors = point_colors.cpu().detach().numpy()
        global_colors_list.append(colors)

    # Optional: voted color (currently not used downstream, but kept for completeness)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    voted_color = vote_color_exclude_zeros(np.asarray(global_colors_list))

    # ----------------------
    # 5) Load existing colored cloud and recolor it
    # ----------------------
    pcd = o3d.io.read_point_cloud(args.input_colored_pcd)

    # Ensure color data is scaled to [0, 255] uint8 for processing
    original_colors = np.asarray(pcd.colors) * 255.0
    original_colors = original_colors.astype(np.uint8)

    # Apply color rescaling function
    final_colors = rescale_red_green(
        original_colors,
        stddev=args.stddev,
        final_color_variation=args.final_color_variation,
    )

    # Rescale colors back to [0, 1] for Open3D
    final_colors_normalized = final_colors.astype(np.float64) / 255.0

    # Assign new colors back to point cloud
    pcd.colors = o3d.utility.Vector3dVector(final_colors_normalized)

    # ----------------------
    # 6) Save final PLY
    # ----------------------
    output_path = os.path.join(args.output_dir, args.output_name)
    o3d.io.write_point_cloud(output_path, pcd)

    print("Final assigned colors for each 3D point saved to:", output_path)
