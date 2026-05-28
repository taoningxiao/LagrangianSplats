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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
from scene.dataset_readers import add_points
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False, load_only_xyz=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold)
            dataset_type="colmap"
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            num_init_points = getattr(args, 'num_init_points', 2000)
            half_res = getattr(args, 'half_res', False)
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.extension, num_init_points, half_res=half_res)
            dataset_type="blender"
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
            dataset_type="dynerf"
        elif os.path.exists(os.path.join(args.source_path,"dataset.json")):
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, False, args.eval)
            dataset_type="nerfies"
        elif os.path.exists(os.path.join(args.source_path,"train_meta.json")):
            scene_info = sceneLoadTypeCallbacks["PanopticSports"](args.source_path)
            dataset_type="PanopticSports"
        elif os.path.exists(os.path.join(args.source_path,"points3D_multipleview.ply")):
            scene_info = sceneLoadTypeCallbacks["MultipleView"](args.source_path)
            dataset_type="MultipleView"
        else:
            assert False, "Could not recognize scene type!"
        self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)

        # self.video_camera = cameraList_from_camInfos(scene_info.video_cameras,-1,args)
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        if args.add_points:
            print("add points.")
            # breakpoint()
            scene_info = scene_info._replace(point_cloud=add_points(scene_info.point_cloud, xyz_max=xyz_max, xyz_min=xyz_min))
        self.gaussians._deformation = self.gaussians._deformation.to("cuda")
        self.gaussians._deformation.deformation_net.set_aabb(xyz_max,xyz_min)
        if hasattr(self.gaussians._deformation, 'use_velocity_advection') and self.gaussians._deformation.use_velocity_advection:
            self.gaussians._deformation.set_velocity_aabb(xyz_max, xyz_min)
        
        if hasattr(self.gaussians._deformation, 'use_velocity_kernel') and self.gaussians._deformation.use_velocity_kernel:
            if self.loaded_iter and not load_only_xyz:
                self.gaussians._deformation.set_velocity_kernel_aabb(xyz_max, xyz_min)
            else:
                pass
        
        if self.loaded_iter:
            coarse_path = os.path.join(self.model_path, "point_cloud", "coarse_iteration_" + str(self.loaded_iter))
            normal_path = os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter))
            
            if os.path.exists(coarse_path):
                iteration_folder = "coarse_iteration_" + str(self.loaded_iter)
            elif os.path.exists(normal_path):
                iteration_folder = "iteration_" + str(self.loaded_iter)
            else:
                point_cloud_dir = os.path.join(self.model_path, "point_cloud")
                if os.path.exists(point_cloud_dir):
                    for fname in os.listdir(point_cloud_dir):
                        if fname.endswith("_" + str(self.loaded_iter)):
                            iteration_folder = fname
                            break
                    else:
                        raise FileNotFoundError(f"Could not find iteration folder for iteration {self.loaded_iter} in {point_cloud_dir}")
                else:
                    raise FileNotFoundError(f"Point cloud directory does not exist: {point_cloud_dir}")
            
            ply_path = os.path.join(self.model_path, "point_cloud", iteration_folder, "point_cloud.ply")
            self.gaussians.load_ply(ply_path, load_only_xyz=load_only_xyz)
            
            if not load_only_xyz:
                deformation_path = os.path.join(self.model_path, "point_cloud", iteration_folder)
                self.gaussians.load_model(deformation_path)
        else:
            if self.gaussians._xyz.numel() == 0:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)
            else:
                print(f"Gaussian already initialized ({self.gaussians._xyz.shape[0]} points), skipping create_from_pcd")
        
        if hasattr(self.gaussians._deformation, 'use_velocity_kernel') and self.gaussians._deformation.use_velocity_kernel:
            if self.gaussians._deformation.velocity_kernel is None:
                self.gaussians._deformation.initialize_velocity_kernel(
                    self.gaussians,
                    xyz_max,
                    xyz_min,
                    device='cuda'
                )

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)
    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera
    def getVideoCameras(self, scale=1.0):
        return self.video_camera