#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from scene.cameras import Camera
import cv2

from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.hyper_loader import Load_hyper_data, format_hyper_data
import torchvision.transforms as transforms
import copy
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB, RGB2SH
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def _normalize_resolution_mode(half_res):
    """
    Normalize the resolution mode while preserving backward compatibility.
    
    Args:
        half_res: can be 'normal', 'half', 'quarter', or True/False for backward compatibility
    
    Returns:
        'normal', 'half', or 'quarter'
    """
    if isinstance(half_res, bool):
        return 'half' if half_res else 'normal'
    elif isinstance(half_res, str):
        half_res_lower = half_res.lower()
        if half_res_lower in ['normal', 'half', 'quarter']:
            return half_res_lower
        else:
            print(f"Warning: Unknown half_res value '{half_res}', defaulting to 'normal'")
            return 'normal'
    else:
        print(f"Warning: Invalid half_res type '{type(half_res)}', defaulting to 'normal'")
        return 'normal'

def _get_resize_scale(resolution_mode):
    """
    Return the resize scale for a resolution mode.
    
    Args:
        resolution_mode: 'normal', 'half', or 'quarter'
    
    Returns:
        resize scale (1, 2,  4)
    """
    if resolution_mode == 'normal':
        return 1
    elif resolution_mode == 'half':
        return 2
    elif resolution_mode == 'quarter':
        return 4
    else:
        return 1

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array
   
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    # breakpoint()
    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image = PILtoTorch(image,None)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              time = float(idx/len(cam_extrinsics)), mask=None) # default by monocular settings.
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    # breakpoint()
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    # breakpoint()
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=train_cam_infos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
def generateCamerasFromTransforms(path, template_transformsfile, extension, maxtime, half_res='normal'):
    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()
    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w
    cam_infos = []
    # generate render poses and times
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)
    render_times = torch.linspace(0,maxtime,render_poses.shape[0])
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file)
        try:
            fovx = template_json["camera_angle_x"]
        except:
            # Account for the configured resolution mode.
            resolution_mode = _normalize_resolution_mode(half_res)
            original_w = template_json['w']
            scale = _get_resize_scale(resolution_mode)
            effective_w = original_w // scale
            fovx = focal2fov(template_json["fl_x"], effective_w)
    # load a single image to get image info.
    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        
        # Read with cv2 to avoid repeated PIL conversions.
        im_data_bgra = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        # Normalize channel layouts.
        if im_data_bgra is None:
            # Fall back to PIL if cv2 cannot read this image.
            image = Image.open(image_path)
            im_data = np.array(image.convert("RGBA"))
            if len(im_data.shape) == 3 and im_data.shape[2] == 3:
                im_data = cv2.cvtColor(im_data, cv2.COLOR_RGB2RGBA)
        else:
            # cv2 reads BGR(A), convert to RGB(A).
            if len(im_data_bgra.shape) == 2:
                im_data = cv2.cvtColor(im_data_bgra, cv2.COLOR_GRAY2RGBA)
            elif im_data_bgra.shape[2] == 3:
                im_data = cv2.cvtColor(im_data_bgra, cv2.COLOR_BGR2RGBA)
            elif im_data_bgra.shape[2] == 4:
                im_data = cv2.cvtColor(im_data_bgra, cv2.COLOR_BGRA2RGBA)
            else:
                im_data = im_data_bgra
        
        # Resize according to the configured resolution mode.
        resolution_mode = _normalize_resolution_mode(half_res)
        if resolution_mode != 'normal':
            original_height, original_width = im_data.shape[:2]
            scale = _get_resize_scale(resolution_mode)
            new_width = original_width // scale
            new_height = original_height // scale
            # cv2.resize expects (width, height).
            im_data = cv2.resize(im_data, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Convert directly to a CHW torch tensor.
        # Normalize to [0, 1].
        im_data_float = im_data.astype(np.float32) / 255.0
        image = torch.from_numpy(im_data_float).permute(2, 0, 1)  # HWC -> CHW
        break
    # format information
    # Read the coordinate-system convention from the template file.
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_contents = json.load(json_file)
        coordinate_system = template_contents.get("coordinate_system", "blender").lower()
        if coordinate_system not in ["blender", "nerf", "colmap"]:
            coordinate_system = "blender"
    
    for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
        time = time/maxtime
        # Extract R and T according to the coordinate-system convention.
        if coordinate_system == "colmap":
            # COLMAP-style poses are world-to-camera transforms.
            c2w = np.array(poses.cpu())
            # change from OpenGL camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3].T
            T = w2c[:3, 3]
        else:
            # Blender/NeRF-style poses are camera-to-world transforms.
            matrix = np.linalg.inv(np.array(poses))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        FovY = fovy 
        FovX = fovx
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
    return cam_infos
def _process_single_frame(args_tuple):
    """Load and process one frame image for the parallel camera reader."""
    idx, frame, path, extension, mapper, white_background, coordinate_system, default_fovx, half_res = args_tuple
    
    cam_name = os.path.join(path, frame["file_path"] + extension)
    time = mapper[frame["time"]]
    
    # Extract R and T according to the coordinate-system convention.
    if coordinate_system == "colmap":
        # COLMAP transform_matrix is world-to-camera.
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].T
        T = w2c[:3, 3]
    else:
        # Blender/NeRF transform_matrix is camera-to-world.
        matrix = np.linalg.inv(np.array(frame["transform_matrix"]))  # c2w -> w2c
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]

    image_path = os.path.join(path, cam_name)
    image_name = Path(cam_name).stem
    
    im_data_bgra = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if im_data_bgra is None:
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        if len(im_data.shape) == 3 and im_data.shape[2] == 3:
            im_data = cv2.cvtColor(im_data, cv2.COLOR_RGB2RGBA)
    else:
        if len(im_data_bgra.shape) == 2:
                # Grayscale image.
            im_data = cv2.cvtColor(im_data_bgra, cv2.COLOR_GRAY2RGBA)
        elif im_data_bgra.shape[2] == 3:
            # BGR image.
            im_data = cv2.cvtColor(im_data_bgra, cv2.COLOR_BGR2RGBA)
        elif im_data_bgra.shape[2] == 4:
            # BGRA image.
            im_data = cv2.cvtColor(im_data_bgra, cv2.COLOR_BGRA2RGBA)
        else:
            im_data = im_data_bgra
    
    # Normalize to [0, 1] and composite the alpha channel.
    norm_data = im_data.astype(np.float32) / 255.0
    bg = np.array([1.0, 1.0, 1.0], dtype=np.float32) if white_background else np.array([0.0, 0.0, 0.0], dtype=np.float32)
    arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    
    resolution_mode = _normalize_resolution_mode(half_res)
    if resolution_mode != 'normal':
        original_height, original_width = arr.shape[:2]
        scale = _get_resize_scale(resolution_mode)
        new_width = original_width // scale
        new_height = original_height // scale
        arr = cv2.resize(arr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    arr_tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
    image = arr_tensor
    
    # Prefer per-frame camera_angle_x, otherwise use the global default.
    # camera_angle_x is already an angle and does not depend on image size.
    # If FOV is derived from fl_x and w, account for downsampling.
    if "camera_angle_x" in frame:
        fovx = frame["camera_angle_x"]
    elif default_fovx is not None:
        fovx = default_fovx
    else:
        # Try deriving FOV from per-frame fl_x and width.
        try:
            # Account for the configured resolution mode.
            resolution_mode = _normalize_resolution_mode(half_res)
            scale = _get_resize_scale(resolution_mode)
            original_w = frame.get('w', image.shape[2] * scale)
            effective_w = original_w // scale
            fovx = focal2fov(frame['fl_x'], effective_w)
        except:
            raise ValueError(f"Cannot determine FOV for camera {idx}; provide camera_angle_x or fl_x/w")
    
    # Compute FOVY from the actual image dimensions.
    # image is a CHW tensor.
    # shape[0] = channels, shape[1] = height, shape[2] = width
    if coordinate_system == "colmap":
        focal = fov2focal(fovx, image.shape[2])
        fovy = focal2fov(focal, image.shape[1])
    else:
        fovy = focal2fov(fov2focal(fovx, image.shape[2]), image.shape[1])
    FovY = fovy 
    FovX = fovx

    # CHW format: width = shape[2], height = shape[1].
    return (idx, CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                            time=time, mask=None))

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", mapper = {}, num_workers=None, half_res='normal'):
    """
    Read camera metadata and images, using a thread pool for image loading.
    
    Args:
        path: Dataset path
        transformsfile: transforms JSON filename
        white_background: Whether to composite over white
        extension: Image extension
        mapper: Timestamp normalization map
        num_workers: Worker count. None selects a bounded default.
        half_res: Resolution mode: 'normal', 'half', or 'quarter'.
                  True/False are also accepted for backward compatibility.
    """
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # Default to the Blender/NeRF convention when the file is not explicit.
        coordinate_system = contents.get("coordinate_system", "blender").lower()
        if coordinate_system not in ["blender", "nerf", "colmap"]:
            # Keep backward compatibility with older Blender-style metadata.
            coordinate_system = "blender"
            print(f"Warning: Unknown coordinate_system '{contents.get('coordinate_system')}', defaulting to 'blender' format")
        
        if coordinate_system == "colmap":
            print("Detected Colmap coordinate system format")
        else:
            print("Detected Blender/NeRF coordinate system format (default)")
        
        # Global camera_angle_x is used as a backward-compatible default.
        try:
            default_fovx = contents["camera_angle_x"]
        except:
            try:
                default_fovx = focal2fov(contents['fl_x'],contents['w'])
            except:
                default_fovx = None
        
        frames = contents["frames"]
        
        # Prepare per-frame jobs for the thread pool.
        if num_workers is None:
            # Use at most 16 workers and avoid over-subscribing small datasets.
            num_workers = min(16, len(frames) + 4)
        
        # Build all frame-processing arguments.
        process_args = [
            (idx, frame, path, extension, mapper, white_background, coordinate_system, default_fovx, half_res)
            for idx, frame in enumerate(frames)
        ]
        
        # Load images in parallel.
        cam_infos_dict = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all jobs.
            future_to_idx = {executor.submit(_process_single_frame, args): args[0] for args in process_args}
            
            # Show camera-loading progress.
            for future in tqdm(as_completed(future_to_idx), total=len(frames), desc="Reading Cameras"):
                idx, cam_info = future.result()
                cam_infos_dict[idx] = cam_info
        
        # Restore the original frame order.
        cam_infos = [cam_infos_dict[i] for i in range(len(frames))]
            
    return cam_infos
def read_timeline(path):
    with open(os.path.join(path, "transforms_train.json")) as json_file:
        train_json = json.load(json_file)
    with open(os.path.join(path, "transforms_test.json")) as json_file:
        test_json = json.load(json_file)  
    time_line = [frame["time"] for frame in train_json["frames"]] + [frame["time"] for frame in test_json["frames"]]
    time_line = set(time_line)
    time_line = list(time_line)
    time_line.sort()
    timestamp_mapper = {}
    max_time_float = max(time_line)
    for index, time in enumerate(time_line):
        timestamp_mapper[time] = time/max_time_float

    return timestamp_mapper, max_time_float
def readNerfSyntheticInfo(path, white_background, eval, extension=".png", num_init_points=2000, half_res='normal'):
    timestamp_mapper, max_time = read_timeline(path)
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, timestamp_mapper, half_res=half_res)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, timestamp_mapper, half_res=half_res)
    print("Generating Video Transforms")
    video_cam_infos = generateCamerasFromTransforms(path, "transforms_train.json", extension, max_time, half_res=half_res)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "fused.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = num_init_points
        print(f"Generating random point cloud ({num_pts})...")

        # Try to read voxel_scale and voxel_matrix from transforms_train.json.
        voxel_scale = None
        voxel_matrix = None
        try:
            with open(os.path.join(path, "transforms_train.json")) as json_file:
                transforms_json = json.load(json_file)
                voxel_scale = transforms_json.get("voxel_scale", None)
                voxel_matrix = transforms_json.get("voxel_matrix", None)
        except:
            pass
        
        # Use voxel metadata to derive the world-space bounding box when available.
        if voxel_scale is not None and voxel_matrix is not None:
            print("Computing the world-space bounding box from voxel metadata...")
            voxel_scale = np.array(voxel_scale, dtype=np.float32)  # [3]
            voxel_matrix = np.array(voxel_matrix, dtype=np.float32)  # [4, 4]
            voxel_matrix = np.stack([voxel_matrix[:,2],voxel_matrix[:,1],voxel_matrix[:,0],voxel_matrix[:,3]],axis=1) # swap_zx
            voxel_tran = voxel_matrix[:3, :]  # [3, 4]
            
            # Corners of the unit voxel domain.
            voxel_corners = np.array([
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
            ], dtype=np.float32)  # [8, 3]
            
            # Transform voxel corners into world space.
            # pos_scale = pts * voxel_scale
            pos_scale = voxel_corners * voxel_scale[None, :]  # [8, 3]
            # pos_rot = sum(pos_scale[..., None, :] * voxel_tran[:3, :3], -1)
            pos_rot = np.sum(pos_scale[:, None, :] * voxel_tran[None, :3, :3], axis=-1)  # [8, 3]
            # pos_off = voxel_tran[:3, -1]
            pos_off = voxel_tran[:3, -1]  # [3]
            # out_pts = pos_rot + pos_off
            world_corners = pos_rot + pos_off[None, :]  # [8, 3]
            
            # Compute the world-space bounding box.
            bbox_min = np.min(world_corners, axis=0)
            bbox_max = np.max(world_corners, axis=0)
            print(f"Computed bbox_min: {bbox_min}")
            print(f"Computed bbox_max: {bbox_max}")
        else:
            # Fallback bounding box for older metadata.
            print("Voxel metadata not found; using fallback bounding box")
            bbox_min = np.array([ 0.0818, -0.0446, -0.4958])
            bbox_max = np.array([ 0.5727,  0.6917, -0.0049])
        
        # Generate random initialization points inside the bounding box.
        center = (bbox_min + bbox_max) / 2.0
        scale = (bbox_max - bbox_min) * 1.0
        bbox_min_center = center - scale / 2.0
        bbox_max_center = center + scale / 2.0
        print("bbox_min_center:", bbox_min_center)
        print("bbox_max_center:", bbox_max_center)
        xyz = np.random.rand(num_pts, 3) * (bbox_max_center - bbox_min_center) + bbox_min_center
        # xyz = np.random.random((num_pts, 3)) * 2.0 - 0.5
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        pcd = fetchPly(ply_path)
        # xyz = -np.array(pcd.points)
        # pcd = pcd._replace(points=xyz)


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )
    return scene_info
def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            R,T = dataset.load_pose(idx)
            FovX = focal2fov(dataset.focal[0], image.shape[1])
            FovY = focal2fov(dataset.focal[0], image.shape[2])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))

    return cameras


def readHyperDataInfos(datadir,use_bg_points,eval):
    train_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split ="train")
    test_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split="test")
    print("load finished")
    train_cam = format_hyper_data(train_cam_infos,"train")
    print("format finished")
    max_time = train_cam_infos.max_time
    video_cam_infos = copy.deepcopy(test_cam_infos)
    video_cam_infos.split="video"


    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    pcd = fetchPly(ply_path)
    xyz = np.array(pcd.points)

    pcd = pcd._replace(points=xyz)
    nerf_normalization = getNerfppNorm(train_cam)
    plot_camera_orientations(train_cam_infos, pcd.points)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )

    return scene_info
def format_render_poses(poses,data_infos):
    cameras = []
    tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    image = data_infos[0][0]
    for idx, p in tqdm(enumerate(poses)):
        # image = None
        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        # matrix = np.linalg.inv(np.array(pose))
        R = pose[:3,:3]
        R = - R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(data_infos.focal[0], image.shape[2])
        FovY = focal2fov(data_infos.focal[0], image.shape[1])
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                            time = time, mask=None))
    return cameras

def add_points(pointsclouds, xyz_min, xyz_max):
    add_points = (np.random.random((100000, 3)))* (xyz_max-xyz_min) + xyz_min
    add_points = add_points.astype(np.float32)
    addcolors = np.random.random((100000, 3)).astype(np.float32)
    addnormals = np.random.random((100000, 3)).astype(np.float32)
    # breakpoint()
    new_points = np.vstack([pointsclouds.points,add_points])
    new_colors = np.vstack([pointsclouds.colors,addcolors])
    new_normals = np.vstack([pointsclouds.normals,addnormals])
    pointsclouds=pointsclouds._replace(points=new_points)
    pointsclouds=pointsclouds._replace(colors=new_colors)
    pointsclouds=pointsclouds._replace(normals=new_normals)
    return pointsclouds
    # breakpoint()
    # new_
def readdynerfInfo(datadir,use_bg_points,eval):
    # loading all the data follow hexplane format
    # ply_path = os.path.join(datadir, "points3D_dense.ply")
    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
    datadir,
    "train",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )    
    test_dataset = Neural3D_NDC_Dataset(
    datadir,
    "test",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )
    train_cam_infos = format_infos(train_dataset,"train")
    val_cam_infos = format_render_poses(test_dataset.val_poses,test_dataset)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # xyz = np.load
    pcd = fetchPly(ply_path)
    print("origin points,",pcd.points.shape[0])
    
    print("after points,",pcd.points.shape[0])

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                           video_cameras=val_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=300
                           )
    return scene_info

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=True
    )
    return cam
def plot_camera_orientations(cam_list, xyz):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    # xyz = xyz[xyz[:,0]<1]
    threshold=2
    xyz = xyz[(xyz[:, 0] >= -threshold) & (xyz[:, 0] <= threshold) &
                         (xyz[:, 1] >= -threshold) & (xyz[:, 1] <= threshold) &
                         (xyz[:, 2] >= -threshold) & (xyz[:, 2] <= threshold)]

    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c='r',s=0.1)
    for cam in tqdm(cam_list):
        R = cam.R
        T = cam.T

        direction = R @ np.array([0, 0, 1])

        ax.quiver(T[0], T[1], T[2], direction[0], direction[1], direction[2], length=1)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.savefig("output.png")
    # breakpoint()
def readPanopticmeta(datadir, json_path):
    with open(os.path.join(datadir,json_path)) as f:
        test_meta = json.load(f)
    w = test_meta['w']
    h = test_meta['h']
    max_time = len(test_meta['fn'])
    cam_infos = []
    for index in range(len(test_meta['fn'])):
        focals = test_meta['k'][index]
        w2cs = test_meta['w2c'][index]
        fns = test_meta['fn'][index]
        cam_ids = test_meta['cam_id'][index]

        time = index / len(test_meta['fn'])
        for focal, w2c, fn, cam in zip(focals, w2cs, fns, cam_ids):
            image_path = os.path.join(datadir,"ims")
            image_name=fn
            image = Image.open(os.path.join(datadir,"ims",fn))
            im_data = np.array(image.convert("RGBA"))
            im_data = PILtoTorch(im_data,None)[:3,:,:]
            camera = setup_camera(w, h, focal, w2c)
            cam_infos.append({
                "camera":camera,
                "time":time,
                "image":im_data})
            
    cam_centers = np.linalg.inv(test_meta['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    return cam_infos, max_time, scene_radius 

def readPanopticSportsinfos(datadir):
    train_cam_infos, max_time, scene_radius = readPanopticmeta(datadir, "train_meta.json")
    test_cam_infos,_, _ = readPanopticmeta(datadir, "test_meta.json")
    nerf_normalization = {
        "radius":scene_radius,
        "translate":torch.tensor([0,0,0])
    }

    ply_path = os.path.join(datadir, "pointd3D.ply")

        # Since this data set has no colmap data, we start with random points
    plz_path = os.path.join(datadir, "init_pt_cld.npz")
    data = np.load(plz_path)["data"]
    xyz = data[:,:3]
    rgb = data[:,3:6]
    num_pts = xyz.shape[0]
    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.ones((num_pts, 3)))
    storePly(ply_path, xyz, rgb)
    # pcd = fetchPly(ply_path)
    # breakpoint()
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time,
                           )
    return scene_info

def readMultipleViewinfos(datadir,llffhold=8):

    cameras_extrinsic_file = os.path.join(datadir, "sparse_/images.bin")
    cameras_intrinsic_file = os.path.join(datadir, "sparse_/cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    from scene.multipleview_dataset import multipleview_dataset
    train_cam_infos = multipleview_dataset(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, cam_folder=datadir,split="train")
    test_cam_infos = multipleview_dataset(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, cam_folder=datadir,split="test")

    train_cam_infos_ = format_infos(train_cam_infos,"train")
    nerf_normalization = getNerfppNorm(train_cam_infos_)

    ply_path = os.path.join(datadir, "points3D_multipleview.ply")
    bin_path = os.path.join(datadir, "points3D_multipleview.bin")
    txt_path = os.path.join(datadir, "points3D_multipleview.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=test_cam_infos.video_cam_infos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "dynerf" : readdynerfInfo,
    "nerfies": readHyperDataInfos,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "PanopticSports" : readPanopticSportsinfos,
    "MultipleView": readMultipleViewinfos
}
