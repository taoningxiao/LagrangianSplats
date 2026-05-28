import os
import json
import torch
import numpy as np
from tqdm import tqdm
import imageio
import glob
import re
import torch.nn.functional as F
from velocity_common.dfrbf import TiDFRBF
import utils.grid_utils as utils_grid
import torch.nn as nn
from utils.image_utils import psnr
from utils.loss_utils import ssim
from pytorch_msssim import ms_ssim
from lpipsPyTorch.modules.lpips import LPIPS
from scene.gaussian_loader import load_gaussian_model
from gaussian_renderer import render
from gaussian_renderer.training import combine_train_test_datasets
from argparse import ArgumentParser
from arguments import PipelineParams, ModelParams

from velocity_common.coordinate_transform import CoordinateTransform
from velocity_common.kernels import generate_kernels
from velocity_common.utils import get_background_color, set_device
from velocity_training.models import InflowGaussians
from velocity_training.wrappers import ExtendedGaussianWrapper, GaussianOverrideWrapper

def evaluate_gaussian_advection_with_velocity(args, load_path: str, savedir: str, checkpoint_epoch: int = None, 
											  gaussian_iteration: int = None, scale: float = None):
	"""
	evaluate train_velocity_model_with_gaussian 
	test view Gaussian  advect ， GT  PSNR、DSSIM  LPIPS
	
	Args:
		args: argument object
		load_path: training-output directorypath（contains ckpt file Gaussian checkpoint ，load_path  gaussian_ckpt_path ）
		savedir: evaluateoutput directory
		checkpoint_epoch: to loadvelocity field checkpoint epoch ，if None find the latest automatically
		gaussian_iteration: Gaussian ，if None find the latest automatically
		scale: resize scale，if None read from args.scale
	"""
	
	device = set_device(args)
	
	import glob
	import re
	from utils.image_utils import psnr
	from utils.loss_utils import ssim
	from pytorch_msssim import ms_ssim
	from lpipsPyTorch.modules.lpips import LPIPS
	
	lpips_vgg_model = LPIPS(net_type='vgg', version='0.1').to(device).eval()
	lpips_alex_model = LPIPS(net_type='alex', version='0.1').to(device).eval()
	
	frame_velocities_dir = os.path.join(load_path, "frame_velocities")
	frame_gaussians_dir = os.path.join(load_path, "frame_gaussians")
	is_sliding_window = os.path.exists(frame_velocities_dir) and os.path.exists(frame_gaussians_dir)
	
	if not is_sliding_window and checkpoint_epoch is None:
		print(f"Searching automatically in {load_path}/ckpt infor the largest epoch...")
		ckpt_dir = os.path.join(load_path, 'ckpt')
		if not os.path.exists(ckpt_dir):
			raise ValueError(f"Checkpoint directory does not exist: {ckpt_dir}")
		
		epochs = set()
		patterns = [
			'velrbf_frame_*_ckpt_*.pth',
			'gt_fit_velrbf_frame_*_ckpt_*.pth',
		]
		
		for pattern in patterns:
			files = glob.glob(os.path.join(ckpt_dir, pattern))
			for file in files:
				match = re.search(r'_ckpt_(\d{6})\.pth', os.path.basename(file))
				if match:
					epoch_num = int(match.group(1))
					epochs.add(epoch_num)
		
		if not epochs:
			raise ValueError(f" {ckpt_dir} innot foundvelocity field checkpoint file")
		
		checkpoint_epoch = max(epochs)
		print(f"Foundlatestvelocity field epoch: {checkpoint_epoch}")
	elif is_sliding_window:
		if checkpoint_epoch is not None:
			print(f"Warning: Sliding-window checkpoints ignore checkpoint_epoch; ignoring it")
		checkpoint_epoch = None
	
	with open(os.path.join(args.datadir, 'info.json'), 'r') as fp:
		meta = json.load(fp)
		voxel_tran = np.float32(meta['voxel_matrix'])
		voxel_tran = np.stack([voxel_tran[:,2],voxel_tran[:,1],voxel_tran[:,0],voxel_tran[:,3]],axis=1)
		voxel_scale = np.broadcast_to(meta['voxel_scale'],[3])
		scene_scale = getattr(args, 'scene_scale', 1.0)
		voxel_scale = voxel_scale.copy() * scene_scale
		voxel_tran[:3,3] *= scene_scale
		train_video = meta['train_videos'][0]
		frame_num = train_video['frame_num']
		delta_t = 1.0 / frame_num
	
	coord_trans = CoordinateTransform(voxel_tran, voxel_scale, device)
	
	if scale is None:
		s = getattr(args, 'scale', 1.0)
	else:
		s = float(scale)
	
	sim_steps = getattr(args, 'sim_steps', 1.0)
	dt = sim_steps / s
	
	gaussian_ckpt_path = load_path
	if not is_sliding_window and gaussian_iteration is None:
		print(f"Searching automatically in {gaussian_ckpt_path} insearch for the latest Gaussian iteration...")
		chkpnt_pattern = os.path.join(gaussian_ckpt_path, "chkpnt_coarse_*.pth")
		chkpnt_files = glob.glob(chkpnt_pattern)
		if chkpnt_files:
			iterations = []
			for file in chkpnt_files:
				match = re.search(r'chkpnt_coarse_(\d+)\.pth', os.path.basename(file))
				if match:
					iterations.append(int(match.group(1)))
			if iterations:
				gaussian_iteration = max(iterations)
				print(f"Foundlatest Gaussian iteration: {gaussian_iteration}")
			else:
				print("not foundvalid Gaussian checkpoint，use load_iteration=-1 load the latest")
				gaussian_iteration = -1
		else:
			print("not found chkpnt_coarse_*.pth file，use load_iteration=-1 load the latest")
			gaussian_iteration = -1
	elif is_sliding_window:
		if gaussian_iteration is not None:
			print(f"Warning: Sliding-window checkpoints ignore gaussian_iteration; ignoring it")
		gaussian_iteration = None
	
	if is_sliding_window:
		print("\n[Step 1] Loading Frame 0 Gaussian from sliding window training results...")
		gaussian_state_path = os.path.join(frame_gaussians_dir, "frame_000_gaussian.pth")
		if not os.path.exists(gaussian_state_path):
			raise FileNotFoundError(f"Frame 0 Gaussian state not found: {gaussian_state_path}")
		
		from argparse import ArgumentParser
		from arguments import ModelParams, ModelHiddenParams, OptimizationParams
		from scene.gaussian_model import GaussianModel
		
		parser = ArgumentParser()
		model_params_obj = ModelParams(parser, sentinel=True)
		hyperparam_obj = ModelHiddenParams(parser)
		op = OptimizationParams(parser)
		
		model_params = model_params_obj.extract(args)
		hyperparam = hyperparam_obj.extract(args)
		opt = op.extract(args)
		
		gaussian_state, _ = torch.load(gaussian_state_path)
		gaussians = GaussianModel(model_params.sh_degree, hyperparam)
		gaussians.restore(gaussian_state, opt)
		
		from scene import Scene
		if not hasattr(model_params, 'eval'):
			model_params.eval = getattr(args, 'eval', True)
		if not hasattr(model_params, 'white_background'):
			model_params.white_background = getattr(args, 'white_background', True)
		if not hasattr(model_params, 'extension'):
			model_params.extension = getattr(args, 'extension', '.png')
		if not hasattr(model_params, 'images'):
			model_params.images = getattr(args, 'images', 'images')
		if not hasattr(model_params, 'llffhold'):
			model_params.llffhold = getattr(args, 'llffhold', 8)
		if not hasattr(model_params, 'add_points'):
			model_params.add_points = getattr(args, 'add_points', False)
		if not hasattr(model_params, 'num_init_points'):
			model_params.num_init_points = getattr(args, 'num_init_points', 2000)
		if not hasattr(model_params, 'half_res'):
			model_params.half_res = getattr(args, 'half_res', False)
		scene = Scene(model_params, gaussians, load_iteration=None, shuffle=False)
		
		print(f"Successfully loaded Frame 0 Gaussian from {gaussian_state_path}")
	else:
		if os.path.exists(gaussian_ckpt_path):
			print("\n[Step 1] Loading pre-trained Frame 0 Gaussian from checkpoint...")
			print(f"Loading from: {gaussian_ckpt_path}, iteration: {gaussian_iteration}")
			try:
				gaussians, scene = load_gaussian_model(
					model_path=gaussian_ckpt_path,
					data_dir=args.datadir,
					load_iteration=gaussian_iteration if gaussian_iteration != -1 else -1,
					load_only_xyz=False,
					base_args=args
				)
				if gaussians is None:
					raise ValueError("gaussians is None after loading")
				if scene is None:
					raise ValueError("scene is None after loading")
				print(f"Successfully loaded Gaussian model from {gaussian_ckpt_path}")
				print(f"Gaussians type: {type(gaussians)}, Scene type: {type(scene)}")
				
				from argparse import ArgumentParser
				from arguments import OptimizationParams
				op = OptimizationParams(ArgumentParser())
				opt = op.extract(args)
				print("Setting up optimizer for loaded Gaussians...")
				gaussians.training_setup(opt)
				print("Optimizer setup completed")
			except Exception as e:
				print(f"Error loading Gaussian model: {e}")
				import traceback
				traceback.print_exc()
				raise
		else:
			raise ValueError(f"Gaussian checkpoint path does not exist: {gaussian_ckpt_path}")
	
	inflow_gaussians = None
	inflow_ratio = 0.0
	checkpoint_path = os.path.join(gaussian_ckpt_path, f"chkpnt_coarse_{gaussian_iteration}.pth")
	if os.path.exists(checkpoint_path):
		try:
			checkpoint_data = torch.load(checkpoint_path, map_location=device)
			if isinstance(checkpoint_data, tuple) and len(checkpoint_data) == 3:
				inflow_checkpoint_data = checkpoint_data[2]
				print(f"\n[Step 1.5] Loading Inflow Gaussians from main checkpoint...")
				inflow_gaussians = InflowGaussians.restore(
					inflow_checkpoint_data,
					gaussians_template=gaussians,
					coord_trans=coord_trans,
					device=device
				)
				inflow_ratio = inflow_checkpoint_data['inflow_ratio']
				print(f"Successfully loaded Inflow Gaussians: {inflow_gaussians.num_groups} groups, {inflow_gaussians.num_points_per_group} points per group")
		except Exception as e:
			print(f"Warning: Could not load inflow from main checkpoint: {e}")
	
	if inflow_gaussians is None:
		inflow_checkpoint_pattern = os.path.join(gaussian_ckpt_path, f"inflow_gaussians_epoch_*.pth")
		inflow_checkpoint_files = glob.glob(inflow_checkpoint_pattern)
		if inflow_checkpoint_files:
			inflow_checkpoint_file = None
			if gaussian_iteration != -1:
				for file in inflow_checkpoint_files:
					match = re.search(r'inflow_gaussians_epoch_(\d+)\.pth', os.path.basename(file))
					if match and int(match.group(1)) == gaussian_iteration:
						inflow_checkpoint_file = file
						break
			
			if inflow_checkpoint_file is None:
				epochs = []
				for file in inflow_checkpoint_files:
					match = re.search(r'inflow_gaussians_epoch_(\d+)\.pth', os.path.basename(file))
					if match:
						epochs.append((int(match.group(1)), file))
				if epochs:
					epochs.sort(key=lambda x: x[0])
					inflow_checkpoint_file = epochs[-1][1]
			
			if inflow_checkpoint_file:
				try:
					print(f"\n[Step 1.5] Loading Inflow Gaussians from independent checkpoint...")
					inflow_checkpoint_data, _ = torch.load(inflow_checkpoint_file, map_location=device)
					inflow_gaussians = InflowGaussians.restore(
						inflow_checkpoint_data,
						gaussians_template=gaussians,
						coord_trans=coord_trans,
						device=device
					)
					inflow_ratio = inflow_checkpoint_data['inflow_ratio']
					print(f"Successfully loaded Inflow Gaussians from {inflow_checkpoint_file}")
					print(f"  {inflow_gaussians.num_groups} groups, {inflow_gaussians.num_points_per_group} points per group")
				except Exception as e:
					print(f"Warning: Could not load inflow from independent checkpoint: {e}")
					import traceback
					traceback.print_exc()
	
	if inflow_gaussians is None:
		print("\n[Step 1.5] No Inflow Gaussians found, proceeding without inflow")

	
	from argparse import ArgumentParser
	from arguments import PipelineParams, ModelParams
	lp = ModelParams(ArgumentParser(), sentinel=True)
	dataset = lp.extract(args)
	pp = PipelineParams(ArgumentParser())
	pipe = pp.extract(args)
	
	for target, source in [(dataset, lp), (pipe, pp)]:
		for key, value in vars(source).items():
			if isinstance(value, ArgumentParser): continue
			attr_name = key[1:] if key.startswith("_") else key
			if not hasattr(target, attr_name):
				setattr(target, attr_name, value)
	
	background = get_background_color(dataset, device=device)
	
	
	if is_sliding_window:
		print(f"\n=== Detected sliding-window training output ===")
		print(f"  Velocity directory: {frame_velocities_dir}")
		print(f"  Gaussian state directory: {frame_gaussians_dir}")
	else:
		print(f"\n=== Using legacy training-output layout ===")
	
	if is_sliding_window:
		print(f"\n=== Loading velocity models from sliding-window output ===")
		vel_models = {}
		for frame_idx in range(frame_num - 1):
			vel_model_path = os.path.join(frame_velocities_dir, f"frame_{frame_idx:03d}_velocity.pth")
			if os.path.exists(vel_model_path):
				model = TiDFRBF.load(vel_model_path, device=device)
				model.eval()
				vel_models[frame_idx] = model
			else:
				print(f"Warning: not foundframe {frame_idx} velocity model: {vel_model_path}")
		print(f"Loaded {len(vel_models)} velocity models")
	else:
		print(f"\n=== Loading velocity models (epoch {checkpoint_epoch}) ===")
		frame_model_files = glob.glob(os.path.join(load_path, "ckpt", f"velrbf_frame_*_ckpt_{checkpoint_epoch:06d}.pth"))
		if len(frame_model_files) == 0:
			raise ValueError(f"not foundvelocity fieldfile: {os.path.join(load_path, 'ckpt', f'velrbf_frame_*_ckpt_{checkpoint_epoch:06d}.pth')}")
		
		frame_model_files.sort()
		vel_models = nn.ModuleList([]).to(device)
		for frame_file in frame_model_files:
			model = TiDFRBF.load(frame_file, device=device)
			model.eval()
			vel_models.append(model)
		
		print(f"Loaded {len(vel_models)} velocity models")
	
	nx, ny, nz = int(args.Nx/s), int(args.Ny/s), int(args.Nz/s)
	grid_shape = (nx, ny, nz)
	
	lengths = np.array([args.Nx, args.Ny, args.Nz])
	lengths_tensor = torch.tensor(lengths, dtype=torch.float32, device=device)
	min_corner = np.zeros(3)
	max_corner = lengths
	grid_points_np = utils_grid.generate_gridpoints(grid_shape, min_corner, max_corner)
	grid_points = torch.from_numpy(grid_points_np).float().to(device)
	
	print("\n[Step 1] Preparing Gaussians for all frames...")
	train_cameras = scene.getTrainCameras()
	test_cameras = scene.getTestCameras()
	
	if is_sliding_window:
		print(f"Reconstructing Gaussians for {frame_num} frames using sliding window pattern with inflow...")
		
		w = getattr(args, 'sliding_window_size', None)
		if w is None:
			max_frame_idx = 0
			for i in range(frame_num):
				vel_path = os.path.join(frame_velocities_dir, f"frame_{i:03d}_velocity.pth")
				if os.path.exists(vel_path):
					max_frame_idx = i
			w = 10
			print(f"Warning: Could not infer window size from args; using default w={w}")
		else:
			print(f"usewindow size: w={w}")
		
		last_window_start = max(0, frame_num - w)
		
		all_inflow_gaussians = None
		inflow_ratio_loaded = 0.0
		
		window_dirs = []
		for start_frame in range(0, frame_num - w + 1):
			end_frame = start_frame + w
			window_dir = os.path.join(load_path, f"window_{start_frame}_{end_frame}")
			if os.path.exists(window_dir):
				window_dirs.append((start_frame, end_frame, window_dir))
		
		if window_dirs:
			print(f"\n[Loading Inflow Gaussians] Scanning {len(window_dirs)} windows...")
			all_inflow_states = []
			
			for start_frame, end_frame, window_dir in window_dirs:
				inflow_pattern = os.path.join(window_dir, "inflow_gaussians_epoch_*.pth")
				inflow_files = glob.glob(inflow_pattern)
				if inflow_files:
					latest_inflow_file = None
					max_epoch = -1
					for inflow_file in inflow_files:
						match = re.search(r'inflow_gaussians_epoch_(\d+)\.pth', os.path.basename(inflow_file))
						if match:
							epoch_num = int(match.group(1))
							if epoch_num > max_epoch:
								max_epoch = epoch_num
								latest_inflow_file = inflow_file
					
					if latest_inflow_file is None:
						latest_inflow_file = max(inflow_files, key=os.path.getctime)
						print(f"  Warning: Could not extract epoch from filename, using getctime for window {start_frame}->{end_frame}")
					
					try:
						inflow_checkpoint_data = torch.load(latest_inflow_file, map_location=device)
						if isinstance(inflow_checkpoint_data, tuple) and len(inflow_checkpoint_data) >= 2:
							inflow_state = inflow_checkpoint_data[0]
							all_inflow_states.append((start_frame, end_frame, inflow_state))
							print(f"  Loaded inflow from window {start_frame}->{end_frame} (epoch {max_epoch}): {latest_inflow_file}")
					except Exception as e:
						print(f"  Warning: Could not load inflow from {latest_inflow_file}: {e}")
			
			if all_inflow_states:
				first_state = all_inflow_states[0][2]
				merged_inflow_state = {
					'num_groups': 0,
					'num_points_per_group': first_state['num_points_per_group'],
					'lengths_tensor': first_state['lengths_tensor'],
					'inflow_ratio': first_state['inflow_ratio'],
					'max_sh_degree': first_state['max_sh_degree'],
					'initialized_groups': [],
					'group_origin_frames': [],
					'xyz_groups': [],
					'features_dc_groups': [],
					'features_rest_groups': [],
					'opacity_groups': [],
					'scaling_groups': [],
					'rotation_groups': [],
				}
				
				origin_frame_to_latest_state = {}
				merged_origin_frames_to_latest_xyz = {}
				merged_origin_frames = set()
				
				for start_frame, end_frame, _ in all_inflow_states:
					merged_origin_frame = start_frame - 1
					if merged_origin_frame >= 0:
						merged_origin_frames.add(merged_origin_frame)
				
				print(f"  Detected merged inflow origin frames (should load attributes from GS checkpoint): {sorted(merged_origin_frames)}")
				
				for start_frame, end_frame, inflow_state in all_inflow_states:
					if 'group_origin_frames' in inflow_state:
						for group_idx in range(inflow_state['num_groups']):
							origin_frame = inflow_state['group_origin_frames'][group_idx]
							if origin_frame is not None:
								if origin_frame in merged_origin_frames:
									if origin_frame not in merged_origin_frames_to_latest_xyz or end_frame > merged_origin_frames_to_latest_xyz[origin_frame][1]:
										merged_origin_frames_to_latest_xyz[origin_frame] = (
											inflow_state['xyz_groups'][group_idx], 
											end_frame, 
											start_frame
										)
								else:
									if origin_frame not in origin_frame_to_latest_state or end_frame > origin_frame_to_latest_state[origin_frame][1]:
										origin_frame_to_latest_state[origin_frame] = (group_idx, end_frame, inflow_state)
				
				sorted_origins = sorted(origin_frame_to_latest_state.keys())
				for origin_frame in sorted_origins:
					group_idx, _, inflow_state = origin_frame_to_latest_state[origin_frame]
					merged_inflow_state['num_groups'] += 1
					merged_inflow_state['initialized_groups'].append(inflow_state['initialized_groups'][group_idx])
					merged_inflow_state['group_origin_frames'].append(origin_frame)
					merged_inflow_state['xyz_groups'].append(inflow_state['xyz_groups'][group_idx])
					merged_inflow_state['features_dc_groups'].append(inflow_state['features_dc_groups'][group_idx])
					merged_inflow_state['features_rest_groups'].append(inflow_state['features_rest_groups'][group_idx])
					merged_inflow_state['opacity_groups'].append(inflow_state['opacity_groups'][group_idx])
					merged_inflow_state['scaling_groups'].append(inflow_state['scaling_groups'][group_idx])
					merged_inflow_state['rotation_groups'].append(inflow_state['rotation_groups'][group_idx])
				
				sorted_merged_origins = sorted(merged_origin_frames_to_latest_xyz.keys())
				for origin_frame in sorted_merged_origins:
					xyz_latest, end_frame, start_frame = merged_origin_frames_to_latest_xyz[origin_frame]
					merged_inflow_state['num_groups'] += 1
					merged_inflow_state['initialized_groups'].append(True)
					merged_inflow_state['group_origin_frames'].append(origin_frame)
					merged_inflow_state['xyz_groups'].append(xyz_latest.clone())
					num_points = xyz_latest.shape[0]
					merged_inflow_state['features_dc_groups'].append(torch.zeros(1, 1, 3, device=device).expand(num_points, -1, -1))
					merged_inflow_state['features_rest_groups'].append(torch.zeros(1, (first_state['max_sh_degree'] + 1) ** 2 - 1, 3, device=device).expand(num_points, -1, -1))
					merged_inflow_state['opacity_groups'].append(torch.zeros(num_points, 1, device=device))
					merged_inflow_state['scaling_groups'].append(torch.zeros(num_points, 3, device=device))
					merged_inflow_state['rotation_groups'].append(torch.zeros(num_points, 4, device=device))
				
				merged_inflow_state['merged_origin_frames'] = sorted_merged_origins
				
				if merged_inflow_state['num_groups'] > 0:
					all_inflow_gaussians = InflowGaussians.restore(
						merged_inflow_state,
						gaussians_template=gaussians,
						coord_trans=coord_trans,
						device=device
					)
					all_inflow_gaussians._merged_origin_frames = set(sorted_merged_origins)
					inflow_ratio_loaded = merged_inflow_state['inflow_ratio']
					print(f"  Successfully merged {len(sorted_origins)} independent inflow groups and {len(sorted_merged_origins)} merged inflow groups (position only)")
					print(f"    Independent inflow groups origin frames: {sorted_origins}")
					print(f"    Merged inflow groups origin frames (attributes from GS): {sorted_merged_origins}")
				else:
					print(f"  Warning: No valid inflow groups found after merging")
			else:
				print(f"  No inflow checkpoints found in any window")
		else:
			print(f"  No window directories found, skipping inflow loading")
		
		def reconstruct_gaussian_for_frame(target_frame):
			"""
			frame Gaussian（position advection，attributes， inflow）
			
			：
			-  GS  inflow（origin_frame == start_frame - 1），attributes target_frame  GS in
			-  inflow（start_frame <= origin_frame < end_frame - 1），attributes inflow checkpoint 
			-  advection ， inflow， inflow  GS  advect
			"""
			gaussian_state_path = os.path.join(frame_gaussians_dir, "frame_000_gaussian.pth")
			gaussian_state, _ = torch.load(gaussian_state_path)
			
			from argparse import ArgumentParser
			from arguments import ModelParams, ModelHiddenParams, OptimizationParams
			from scene.gaussian_model import GaussianModel
			
			parser = ArgumentParser()
			model_params_obj = ModelParams(parser, sentinel=True)
			hyperparam_obj = ModelHiddenParams(parser)
			op = OptimizationParams(parser)
			
			model_params = model_params_obj.extract(args)
			hyperparam = hyperparam_obj.extract(args)
			opt = op.extract(args)
			
			frame_gaussians = GaussianModel(model_params.sh_degree, hyperparam)
			frame_gaussians.restore(gaussian_state, opt)
			
			target_gaussian_state_path = os.path.join(frame_gaussians_dir, f"frame_{target_frame:03d}_gaussian.pth")
			
			if not os.path.exists(target_gaussian_state_path) and target_frame >= last_window_start:
				fallback_frame = last_window_start
				target_gaussian_state_path = os.path.join(frame_gaussians_dir, f"frame_{fallback_frame:03d}_gaussian.pth")
				if os.path.exists(target_gaussian_state_path):
					print(f"  Frame {target_frame} file，use frame {fallback_frame} attributes（start frame）")
				else:
					print(f"  Warning: Frame {target_frame}  fallback frame {fallback_frame} file，use frame 0 attributes")
					target_gaussian_state_path = None
			
			if target_gaussian_state_path and os.path.exists(target_gaussian_state_path):
				target_gaussian_state, _ = torch.load(target_gaussian_state_path)
				(active_sh_degree, xyz, deform_state, deformation_table, features_dc, features_rest,
				 scaling, rotation, opacity, max_radii2D, xyz_gradient_accum, denom, opt_dict, spatial_lr_scale) = target_gaussian_state
				
				frame_gaussians._features_dc.data = features_dc
				frame_gaussians._features_rest.data = features_rest
				frame_gaussians._scaling.data = scaling
				frame_gaussians._rotation.data = rotation
				frame_gaussians._opacity.data = opacity
				frame_gaussians.max_radii2D = max_radii2D
			
			orig_num_points = frame_gaussians.get_xyz.shape[0]
			
			merged_inflow_attributes = None
			if target_gaussian_state_path and os.path.exists(target_gaussian_state_path):
				target_gaussian_state, _ = torch.load(target_gaussian_state_path)
				(active_sh_degree, xyz_target, deform_state, deformation_table, features_dc_target, features_rest_target,
				 scaling_target, rotation_target, opacity_target, max_radii2D, xyz_gradient_accum, denom, opt_dict, spatial_lr_scale) = target_gaussian_state
				
				if xyz_target.shape[0] > orig_num_points:
					merged_inflow_attributes = {
						'features_dc': features_dc_target[orig_num_points:],
						'features_rest': features_rest_target[orig_num_points:],
						'scaling': scaling_target[orig_num_points:],
						'rotation': rotation_target[orig_num_points:],
						'opacity': opacity_target[orig_num_points:],
					}
					print(f"  Extracted merged inflow attributes from target GS state: {xyz_target.shape[0] - orig_num_points} points")
			
			xyz_world = frame_gaussians.get_xyz.detach().clone()
			xyz_smoke = coord_trans.world2smoke(xyz_world)
			xyz_sim = xyz_smoke * lengths_tensor
			
			all_positions_sim = [xyz_sim]
			
			merged_inflow_group_indices = []
			
			with torch.no_grad():
				for frame_idx in range(target_frame):
					current_pos_sim = all_positions_sim[-1]
					
					if all_inflow_gaussians is not None and inflow_ratio_loaded > 0:
						inflow_groups_to_add = []
						for group_idx in range(all_inflow_gaussians.num_groups):
							origin_frame = all_inflow_gaussians.get_group_origin_frame(group_idx)
							if origin_frame == frame_idx:
								if hasattr(all_inflow_gaussians, '_merged_origin_frames') and origin_frame in all_inflow_gaussians._merged_origin_frames:
									merged_inflow_group_indices.append(group_idx)
								inflow_xyz_sim = all_inflow_gaussians.get_group_xyz_sim(group_idx)
								inflow_groups_to_add.append((group_idx, inflow_xyz_sim))
						
						if inflow_groups_to_add:
							inflow_positions = [xyz_sim for _, xyz_sim in inflow_groups_to_add]
							if inflow_positions:
								combined_inflow_pos = torch.cat(inflow_positions, dim=0)
								current_pos_sim = torch.cat([current_pos_sim, combined_inflow_pos], dim=0)
					
					if frame_idx in vel_models:
						vel_model = vel_models[frame_idx]
						
						v_flat = vel_model(grid_points)
						v_vol = v_flat.view(nx, ny, nz, 3).permute(3, 2, 1, 0).unsqueeze(0)
						
						norm_pos = current_pos_sim.clone()
						norm_pos = 2 * (norm_pos / lengths_tensor) - 1.0
						grid_in = norm_pos.view(1, 1, 1, -1, 3)
						v_part = F.grid_sample(v_vol, grid_in, align_corners=True, mode='bilinear')
						v_part = v_part.view(3, -1).permute(1, 0)
						
						advected_pos_sim = current_pos_sim + v_part * dt
					else:
						advected_pos_sim = current_pos_sim
					
					all_positions_sim.append(advected_pos_sim)
				
				final_pos_sim = all_positions_sim[-1]
				
				final_pos_smoke = final_pos_sim / lengths_tensor
				final_pos_world = coord_trans.smoke2world(final_pos_smoke)
				
				frame_gaussians._xyz.data = final_pos_world[:orig_num_points]
				
				if merged_inflow_attributes is not None and all_inflow_gaussians is not None:
					merged_origin_frames_sorted = sorted(all_inflow_gaussians._merged_origin_frames)
					
					total_merged_points = sum(
						all_inflow_gaussians._xyz_groups[g_idx].shape[0]
						for g_idx in range(all_inflow_gaussians.num_groups)
						if all_inflow_gaussians.get_group_origin_frame(g_idx) in all_inflow_gaussians._merged_origin_frames
					)
					
					merged_attr_size = merged_inflow_attributes['opacity'].shape[0]
					if merged_attr_size != total_merged_points:
						print(f"  Warning: Merged inflow attributes size mismatch: GS checkpoint has {merged_attr_size} points, but expected {total_merged_points} points")
						print(f"    This may happen if some inflow points were removed during training. Using available attributes.")
					
					attr_offset = 0
					for origin_frame in merged_origin_frames_sorted:
						group_idx = None
						for g_idx in range(all_inflow_gaussians.num_groups):
							if all_inflow_gaussians.get_group_origin_frame(g_idx) == origin_frame:
								group_idx = g_idx
								break
						
						if group_idx is not None:
							num_points_in_group = all_inflow_gaussians._xyz_groups[group_idx].shape[0]
							
							if attr_offset + num_points_in_group <= merged_attr_size:
								all_inflow_gaussians._features_dc_groups[group_idx].data.copy_(
									merged_inflow_attributes['features_dc'][attr_offset:attr_offset + num_points_in_group]
								)
								all_inflow_gaussians._features_rest_groups[group_idx].data.copy_(
									merged_inflow_attributes['features_rest'][attr_offset:attr_offset + num_points_in_group]
								)
								all_inflow_gaussians._opacity_groups[group_idx].data.copy_(
									merged_inflow_attributes['opacity'][attr_offset:attr_offset + num_points_in_group]
								)
								all_inflow_gaussians._scaling_groups[group_idx].data.copy_(
									merged_inflow_attributes['scaling'][attr_offset:attr_offset + num_points_in_group]
								)
								all_inflow_gaussians._rotation_groups[group_idx].data.copy_(
									merged_inflow_attributes['rotation'][attr_offset:attr_offset + num_points_in_group]
								)
								
								attr_offset += num_points_in_group
							else:
								print(f"  Warning: Skipping merged inflow group {group_idx} (origin_frame={origin_frame}) due to attribute size mismatch")
					
					print(f"  Updated attributes for {len(merged_origin_frames_sorted)} merged inflow groups from target GS state")
			
			return frame_gaussians, final_pos_world, all_inflow_gaussians
		
		frame_gaussians_cache = {}
		frame_positions_cache = {}
		
		traj_world = [None] * frame_num
		
		def get_gaussian_for_frame(frame_idx):
			"""getframe Gaussian（use），return (gaussians, pos_world, inflow_gaussians)"""
			frame_gaussians, pos_world, inflow_gs = reconstruct_gaussian_for_frame(frame_idx)
			return frame_gaussians, pos_world, inflow_gs
		
		def reconstruct_gaussian_for_frame_advect_only(target_frame):
			"""use frame_0  Gaussian position advection（ inflow），attributes（opacity ）use target_frame """
			gaussian_state_path = os.path.join(frame_gaussians_dir, "frame_000_gaussian.pth")
			gaussian_state, _ = torch.load(gaussian_state_path)
			
			from argparse import ArgumentParser
			from arguments import ModelParams, ModelHiddenParams, OptimizationParams
			from scene.gaussian_model import GaussianModel
			
			parser = ArgumentParser()
			model_params_obj = ModelParams(parser, sentinel=True)
			hyperparam_obj = ModelHiddenParams(parser)
			op = OptimizationParams(parser)
			
			model_params = model_params_obj.extract(args)
			hyperparam = hyperparam_obj.extract(args)
			opt = op.extract(args)
			
			frame_gaussians = GaussianModel(model_params.sh_degree, hyperparam)
			frame_gaussians.restore(gaussian_state, opt)
			
			target_gaussian_state_path = os.path.join(frame_gaussians_dir, f"frame_{target_frame:03d}_gaussian.pth")
			
			if not os.path.exists(target_gaussian_state_path) and target_frame >= last_window_start:
				fallback_frame = last_window_start
				target_gaussian_state_path = os.path.join(frame_gaussians_dir, f"frame_{fallback_frame:03d}_gaussian.pth")
				if os.path.exists(target_gaussian_state_path):
					print(f"  [Advect Only] Frame {target_frame} file，use frame {fallback_frame} attributes（start frame）")
				else:
					print(f"  [Advect Only] Warning: Frame {target_frame}  fallback frame {fallback_frame} file，use frame 0 attributes")
					target_gaussian_state_path = None
			
			if target_gaussian_state_path and os.path.exists(target_gaussian_state_path):
				target_gaussian_state, _ = torch.load(target_gaussian_state_path)
				(active_sh_degree, xyz, deform_state, deformation_table, features_dc, features_rest,
				 scaling, rotation, opacity, max_radii2D, xyz_gradient_accum, denom, opt_dict, spatial_lr_scale) = target_gaussian_state
				
				frame_gaussians._features_dc.data = features_dc
				frame_gaussians._features_rest.data = features_rest
				frame_gaussians._scaling.data = scaling
				frame_gaussians._rotation.data = rotation
				frame_gaussians._opacity.data = opacity
				frame_gaussians.max_radii2D = max_radii2D
			
			xyz_world = frame_gaussians.get_xyz.detach().clone()
			xyz_smoke = coord_trans.world2smoke(xyz_world)
			xyz_sim = xyz_smoke * lengths_tensor
			
			current_pos_sim = xyz_sim
			
			with torch.no_grad():
				for frame_idx in range(target_frame):
					if frame_idx in vel_models:
						vel_model = vel_models[frame_idx]
						
						v_flat = vel_model(grid_points)
						v_vol = v_flat.view(nx, ny, nz, 3).permute(3, 2, 1, 0).unsqueeze(0)
						
						norm_pos = current_pos_sim.clone()
						norm_pos = 2 * (norm_pos / lengths_tensor) - 1.0
						grid_in = norm_pos.view(1, 1, 1, -1, 3)
						v_part = F.grid_sample(v_vol, grid_in, align_corners=True, mode='bilinear')
						v_part = v_part.view(3, -1).permute(1, 0)
						
						current_pos_sim = current_pos_sim + v_part * dt
				
				final_pos_sim = current_pos_sim
				
				final_pos_smoke = final_pos_sim / lengths_tensor
				final_pos_world = coord_trans.smoke2world(final_pos_smoke)
				
				frame_gaussians._xyz.data = final_pos_world
			
			return frame_gaussians, final_pos_world, None
		
		frame_gaussians_cache_advect_only = {}
		frame_positions_cache_advect_only = {}
		
		def get_gaussian_for_frame_advect_only(frame_idx):
			"""getframe Gaussian（use frame_0 position advection， inflow，attributesuseper-frame，use），return (gaussians, pos_world, None)"""
			frame_gaussians, pos_world, inflow_gs = reconstruct_gaussian_for_frame_advect_only(frame_idx)
			return frame_gaussians, pos_world, inflow_gs
	else:
		get_gaussian_for_frame = None
		
		xyz_world_0 = gaussians.get_xyz.detach().clone()
		xyz_smoke_0 = coord_trans.world2smoke(xyz_world_0)
		xyz_sim_0 = xyz_smoke_0 * lengths_tensor
		
		traj_sim = [xyz_sim_0]
		current_sim_pos = xyz_sim_0
		
		print(f"Advecting Gaussian through {frame_num} frames...")
		with torch.no_grad():
			for frame_idx in tqdm(range(frame_num - 1), desc="Advecting"):
				# === Inflow: Merge pre-created inflow points at frame_idx > 0 ===
				if frame_idx > 0 and inflow_ratio > 0 and inflow_gaussians is not None:
					# Get pre-created inflow points for this time step (group index = frame_idx - 1)
					inflow_xyz_sim = inflow_gaussians.get_group_xyz_sim(frame_idx - 1)
					
					# Merge to current position (they start at their initial positions)
					current_sim_pos = torch.cat([current_sim_pos, inflow_xyz_sim], dim=0)
					
					# Update the last trajectory entry to include new points at their initial positions
					traj_sim[-1] = torch.cat([traj_sim[-1], inflow_xyz_sim], dim=0)
				
				v_flat = vel_models[frame_idx](grid_points)  # [N_grid, 3]
				
				# 2. Reshape to Volume [1, 3, D, H, W] = [1, 3, nz, ny, nx]
				v_vol = v_flat.view(nx, ny, nz, 3).permute(3, 2, 1, 0).unsqueeze(0)  # [1, 3, nz, ny, nx]
				
				# 3. Normalize particle pos to [-1, 1] for grid_sample
				norm_pos = current_sim_pos.clone()
				norm_pos = 2 * (norm_pos / lengths_tensor) - 1.0
				
				# 4. Reshape for grid_sample: [1, 1, 1, N, 3]
				grid_in = norm_pos.view(1, 1, 1, -1, 3)
				
				# 5. Sample Velocity using grid_sample
				v_part = F.grid_sample(v_vol, grid_in, align_corners=True, mode='bilinear')
				v_part = v_part.view(3, -1).permute(1, 0)  # [N, 3]
				
				# 6. Advect in Sim Space
				current_sim_pos = current_sim_pos + v_part * dt
				traj_sim.append(current_sim_pos)
		
		print(f"Advection complete. Trajectory stored for {len(traj_sim)} frames.")
		
		traj_world = []
		for pos_sim in traj_sim:
			pos_smoke = pos_sim / lengths_tensor
			pos_world = coord_trans.smoke2world(pos_smoke)
			traj_world.append(pos_world)
	
	print(f"training cameras: {len(train_cameras)}, test cameras: {len(test_cameras)}")
	
	os.makedirs(savedir, exist_ok=True)
	os.makedirs(os.path.join(savedir, "images", "train"), exist_ok=True)
	os.makedirs(os.path.join(savedir, "images", "test"), exist_ok=True)
	os.makedirs(os.path.join(savedir, "video", "train"), exist_ok=True)
	os.makedirs(os.path.join(savedir, "video", "test"), exist_ok=True)
	
	if is_sliding_window:
		os.makedirs(os.path.join(savedir, "images_advect_frame0_only", "train"), exist_ok=True)
		os.makedirs(os.path.join(savedir, "images_advect_frame0_only", "test"), exist_ok=True)
		os.makedirs(os.path.join(savedir, "video_advect_frame0_only", "train"), exist_ok=True)
		os.makedirs(os.path.join(savedir, "video_advect_frame0_only", "test"), exist_ok=True)
	
	train_camera_renders = []  # List of images
	train_camera_gts = []      # List of images
	train_camera_psnrs = []    # List of psnr values
	train_camera_ssims = []    # List of ssim values
	train_camera_dssims = []   # List of dssim values
	train_camera_lpips_vgg = []  # List of lpips values
	train_camera_lpips_alex = [] # List of lpips values
	train_camera_frame_indices = []  # List of frame indices for each camera
	
	test_camera_renders = []
	test_camera_gts = []
	test_camera_psnrs = []
	test_camera_ssims = []    # List of ssim values
	test_camera_dssims = []
	test_camera_lpips_vgg = []
	test_camera_lpips_alex = []
	test_camera_frame_indices = []
	
	if is_sliding_window:
		train_camera_renders_advect_only = []
		train_camera_psnrs_advect_only = []
		train_camera_ssims_advect_only = []
		train_camera_dssims_advect_only = []
		train_camera_lpips_vgg_advect_only = []
		train_camera_lpips_alex_advect_only = []
		
		test_camera_renders_advect_only = []
		test_camera_psnrs_advect_only = []
		test_camera_ssims_advect_only = []
		test_camera_dssims_advect_only = []
		test_camera_lpips_vgg_advect_only = []
		test_camera_lpips_alex_advect_only = []
	
	def get_frame_idx_from_camera_time(cam):
		"""based oncamera time attributesframe index"""
		if hasattr(cam, 'time') and cam.time is not None:
			# cam.time is in [0, 1], map to [0, frame_num-1]
			if frame_num > 1:
				frame_idx = int(cam.time * (frame_num - 1))
			else:
				frame_idx = 0
			# Clamp to valid range
			frame_idx = max(0, min(frame_idx, frame_num - 1))
			return frame_idx
		return 0  # Default to frame 0
	
	print(f"\n=== startevaluatetraining cameras ===")
	for idx, view in enumerate(tqdm(train_cameras, desc="Rendering train cameras")):
		frame_idx = get_frame_idx_from_camera_time(view)
		
		if is_sliding_window:
			frame_gaussians, pos_world, frame_inflow_gaussians = get_gaussian_for_frame(frame_idx)
			
			has_inflow_frame = (frame_idx > 0 and frame_inflow_gaussians is not None)
			if has_inflow_frame:
				inflow_group_indices_frame = []
				for group_idx in range(frame_inflow_gaussians.num_groups):
					origin_frame = frame_inflow_gaussians.get_group_origin_frame(group_idx)
					if origin_frame is not None and origin_frame < frame_idx:
						inflow_group_indices_frame.append(group_idx)
				
				if len(inflow_group_indices_frame) > 0:
					wrapped_gaussians = ExtendedGaussianWrapper(frame_gaussians, pos_world, frame_inflow_gaussians, inflow_group_indices_frame)
				else:
					wrapped_gaussians = GaussianOverrideWrapper(frame_gaussians, pos_world[:frame_gaussians.get_xyz.shape[0]])
			else:
				wrapped_gaussians = GaussianOverrideWrapper(frame_gaussians, pos_world[:frame_gaussians.get_xyz.shape[0]])
		else:
			pos_world = traj_world[frame_idx]
			
			has_inflow_frame = (frame_idx > 0 and inflow_ratio > 0 and inflow_gaussians is not None)
			if has_inflow_frame:
				# Split pos_world into original and inflow parts
				orig_num_points = gaussians.get_xyz.shape[0]
				pos_world_orig = pos_world[:orig_num_points]
				pos_world_inflow = pos_world[orig_num_points:]
				
				# Determine which inflow groups should be included at this frame
				inflow_group_indices_frame = list(range(frame_idx))
				
				# Use ExtendedGaussianWrapper to merge original and inflow GS
				wrapped_gaussians = ExtendedGaussianWrapper(gaussians, pos_world, inflow_gaussians, inflow_group_indices_frame)
			else:
				# Only original GS
				wrapped_gaussians = GaussianOverrideWrapper(gaussians, pos_world)
		
		with torch.no_grad():
			render_result = render(view, wrapped_gaussians, pipe, background, stage="coarse")["render"]
			gt_image = view.original_image.cuda()
		
		render_tensor = render_result.unsqueeze(0)  # [1, 3, H, W]
		gt_tensor = gt_image.unsqueeze(0)  # [1, 3, H, W]
		
		psnr_val = psnr(render_tensor, gt_tensor).item()
		train_camera_psnrs.append(psnr_val)
		
		ssim_val = ssim(render_tensor, gt_tensor)
		if isinstance(ssim_val, torch.Tensor):
			ssim_val = ssim_val.item()
		train_camera_ssims.append(ssim_val)
		
		ms_ssim_val = ms_ssim(render_tensor, gt_tensor, data_range=1, size_average=True)
		if isinstance(ms_ssim_val, torch.Tensor):
			ms_ssim_val = ms_ssim_val.item()
		dssim_ms_val = (1 - ms_ssim_val) / 2
		train_camera_dssims.append(dssim_ms_val)
		
		torch.cuda.empty_cache()
		lpips_vgg_val = lpips_vgg_model(render_tensor, gt_tensor).squeeze().item()
		lpips_alex_val = lpips_alex_model(render_tensor, gt_tensor).squeeze().item()
		train_camera_lpips_vgg.append(lpips_vgg_val)
		train_camera_lpips_alex.append(lpips_alex_val)
		
		render_np = render_result.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
		gt_np = gt_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
		
		render_img = (render_np * 255).astype(np.uint8)
		gt_img = (gt_np * 255).astype(np.uint8)
		
		train_camera_renders.append(render_img)
		train_camera_gts.append(gt_img)
		train_camera_frame_indices.append(frame_idx)
		
		render_path = os.path.join(savedir, "images", "train", f"cam_{idx:03d}_frame_{frame_idx:03d}_render.png")
		gt_path = os.path.join(savedir, "images", "train", f"cam_{idx:03d}_frame_{frame_idx:03d}_gt.png")
		imageio.imwrite(render_path, render_img)
		imageio.imwrite(gt_path, gt_img)
		
		if not is_sliding_window:
			del render_result, render_tensor, render_np, render_img, gt_np, gt_img
			del wrapped_gaussians
			del gt_tensor, gt_image
			torch.cuda.empty_cache()
		else:
			del render_result, render_tensor, render_np, render_img
			del wrapped_gaussians, frame_gaussians, pos_world
			if frame_inflow_gaussians is not None:
				del frame_inflow_gaussians
			torch.cuda.empty_cache()
			
			frame_gaussians_advect_only, pos_world_advect_only, _ = get_gaussian_for_frame_advect_only(frame_idx)
			
			wrapped_gaussians_advect_only = GaussianOverrideWrapper(frame_gaussians_advect_only, pos_world_advect_only)
			
			with torch.no_grad():
				render_result_advect_only = render(view, wrapped_gaussians_advect_only, pipe, background, stage="coarse")["render"]
			
			render_tensor_advect_only = render_result_advect_only.unsqueeze(0)
			
			psnr_val_advect_only = psnr(render_tensor_advect_only, gt_tensor).item()
			train_camera_psnrs_advect_only.append(psnr_val_advect_only)
			
			ssim_val_advect_only = ssim(render_tensor_advect_only, gt_tensor)
			if isinstance(ssim_val_advect_only, torch.Tensor):
				ssim_val_advect_only = ssim_val_advect_only.item()
			train_camera_ssims_advect_only.append(ssim_val_advect_only)
			
			ms_ssim_val_advect_only = ms_ssim(render_tensor_advect_only, gt_tensor, data_range=1, size_average=True)
			if isinstance(ms_ssim_val_advect_only, torch.Tensor):
				ms_ssim_val_advect_only = ms_ssim_val_advect_only.item()
			dssim_ms_val_advect_only = (1 - ms_ssim_val_advect_only) / 2
			train_camera_dssims_advect_only.append(dssim_ms_val_advect_only)
			
			torch.cuda.empty_cache()
			lpips_vgg_val_advect_only = lpips_vgg_model(render_tensor_advect_only, gt_tensor).squeeze().item()
			lpips_alex_val_advect_only = lpips_alex_model(render_tensor_advect_only, gt_tensor).squeeze().item()
			train_camera_lpips_vgg_advect_only.append(lpips_vgg_val_advect_only)
			train_camera_lpips_alex_advect_only.append(lpips_alex_val_advect_only)
			
			render_np_advect_only = render_result_advect_only.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
			render_img_advect_only = (render_np_advect_only * 255).astype(np.uint8)
			
			train_camera_renders_advect_only.append(render_img_advect_only)
			
			render_path_advect_only = os.path.join(savedir, "images_advect_frame0_only", "train", f"cam_{idx:03d}_frame_{frame_idx:03d}_render.png")
			imageio.imwrite(render_path_advect_only, render_img_advect_only)
			
			del render_result_advect_only, render_tensor_advect_only, render_np_advect_only, render_img_advect_only
			del wrapped_gaussians_advect_only, frame_gaussians_advect_only, pos_world_advect_only
			del gt_tensor, gt_image, gt_np
			torch.cuda.empty_cache()
	
	print(f"\n=== startevaluatetest cameras ===")
	for idx, view in enumerate(tqdm(test_cameras, desc="Rendering test cameras")):
		frame_idx = get_frame_idx_from_camera_time(view)
		
		if is_sliding_window:
			frame_gaussians, pos_world, frame_inflow_gaussians = get_gaussian_for_frame(frame_idx)
			
			has_inflow_frame = (frame_idx > 0 and frame_inflow_gaussians is not None)
			if has_inflow_frame:
				inflow_group_indices_frame = []
				for group_idx in range(frame_inflow_gaussians.num_groups):
					origin_frame = frame_inflow_gaussians.get_group_origin_frame(group_idx)
					if origin_frame is not None and origin_frame < frame_idx:
						inflow_group_indices_frame.append(group_idx)
				
				if len(inflow_group_indices_frame) > 0:
					wrapped_gaussians = ExtendedGaussianWrapper(frame_gaussians, pos_world, frame_inflow_gaussians, inflow_group_indices_frame)
				else:
					wrapped_gaussians = GaussianOverrideWrapper(frame_gaussians, pos_world[:frame_gaussians.get_xyz.shape[0]])
			else:
				wrapped_gaussians = GaussianOverrideWrapper(frame_gaussians, pos_world[:frame_gaussians.get_xyz.shape[0]])
		else:
			pos_world = traj_world[frame_idx]
			
			has_inflow_frame = (frame_idx > 0 and inflow_ratio > 0 and inflow_gaussians is not None)
			if has_inflow_frame:
				# Split pos_world into original and inflow parts
				orig_num_points = gaussians.get_xyz.shape[0]
				pos_world_orig = pos_world[:orig_num_points]
				pos_world_inflow = pos_world[orig_num_points:]
				
				# Determine which inflow groups should be included at this frame
				inflow_group_indices_frame = list(range(frame_idx))
				
				# Use ExtendedGaussianWrapper to merge original and inflow GS
				wrapped_gaussians = ExtendedGaussianWrapper(gaussians, pos_world, inflow_gaussians, inflow_group_indices_frame)
			else:
				# Only original GS
				wrapped_gaussians = GaussianOverrideWrapper(gaussians, pos_world)
		
		with torch.no_grad():
			render_result = render(view, wrapped_gaussians, pipe, background, stage="coarse")["render"]
			gt_image = view.original_image.cuda()
		
		render_tensor = render_result.unsqueeze(0)  # [1, 3, H, W]
		gt_tensor = gt_image.unsqueeze(0)  # [1, 3, H, W]
		
		psnr_val = psnr(render_tensor, gt_tensor).item()
		test_camera_psnrs.append(psnr_val)
		
		ssim_val = ssim(render_tensor, gt_tensor)
		if isinstance(ssim_val, torch.Tensor):
			ssim_val = ssim_val.item()
		test_camera_ssims.append(ssim_val)
		
		ms_ssim_val = ms_ssim(render_tensor, gt_tensor, data_range=1, size_average=True)
		if isinstance(ms_ssim_val, torch.Tensor):
			ms_ssim_val = ms_ssim_val.item()
		dssim_ms_val = (1 - ms_ssim_val) / 2
		test_camera_dssims.append(dssim_ms_val)
		
		torch.cuda.empty_cache()
		lpips_vgg_val = lpips_vgg_model(render_tensor, gt_tensor).squeeze().item()
		lpips_alex_val = lpips_alex_model(render_tensor, gt_tensor).squeeze().item()
		test_camera_lpips_vgg.append(lpips_vgg_val)
		test_camera_lpips_alex.append(lpips_alex_val)
		
		render_np = render_result.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
		gt_np = gt_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
		
		render_img = (render_np * 255).astype(np.uint8)
		gt_img = (gt_np * 255).astype(np.uint8)
		
		test_camera_renders.append(render_img)
		test_camera_gts.append(gt_img)
		test_camera_frame_indices.append(frame_idx)
		
		render_path = os.path.join(savedir, "images", "test", f"cam_{idx:03d}_frame_{frame_idx:03d}_render.png")
		gt_path = os.path.join(savedir, "images", "test", f"cam_{idx:03d}_frame_{frame_idx:03d}_gt.png")
		imageio.imwrite(render_path, render_img)
		imageio.imwrite(gt_path, gt_img)
		
		if not is_sliding_window:
			del render_result, render_tensor, render_np, render_img, gt_np, gt_img
			del wrapped_gaussians
			del gt_tensor, gt_image
			torch.cuda.empty_cache()
		else:
			del render_result, render_tensor, render_np, render_img
			del wrapped_gaussians, frame_gaussians, pos_world
			if frame_inflow_gaussians is not None:
				del frame_inflow_gaussians
			torch.cuda.empty_cache()
			
			frame_gaussians_advect_only, pos_world_advect_only, _ = get_gaussian_for_frame_advect_only(frame_idx)
			
			wrapped_gaussians_advect_only = GaussianOverrideWrapper(frame_gaussians_advect_only, pos_world_advect_only)
			
			with torch.no_grad():
				render_result_advect_only = render(view, wrapped_gaussians_advect_only, pipe, background, stage="coarse")["render"]
			
			render_tensor_advect_only = render_result_advect_only.unsqueeze(0)
			
			psnr_val_advect_only = psnr(render_tensor_advect_only, gt_tensor).item()
			test_camera_psnrs_advect_only.append(psnr_val_advect_only)
			
			ssim_val_advect_only = ssim(render_tensor_advect_only, gt_tensor)
			if isinstance(ssim_val_advect_only, torch.Tensor):
				ssim_val_advect_only = ssim_val_advect_only.item()
			test_camera_ssims_advect_only.append(ssim_val_advect_only)
			
			ms_ssim_val_advect_only = ms_ssim(render_tensor_advect_only, gt_tensor, data_range=1, size_average=True)
			if isinstance(ms_ssim_val_advect_only, torch.Tensor):
				ms_ssim_val_advect_only = ms_ssim_val_advect_only.item()
			dssim_ms_val_advect_only = (1 - ms_ssim_val_advect_only) / 2
			test_camera_dssims_advect_only.append(dssim_ms_val_advect_only)
			
			torch.cuda.empty_cache()
			lpips_vgg_val_advect_only = lpips_vgg_model(render_tensor_advect_only, gt_tensor).squeeze().item()
			lpips_alex_val_advect_only = lpips_alex_model(render_tensor_advect_only, gt_tensor).squeeze().item()
			test_camera_lpips_vgg_advect_only.append(lpips_vgg_val_advect_only)
			test_camera_lpips_alex_advect_only.append(lpips_alex_val_advect_only)
			
			render_np_advect_only = render_result_advect_only.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
			render_img_advect_only = (render_np_advect_only * 255).astype(np.uint8)
			
			test_camera_renders_advect_only.append(render_img_advect_only)
			
			render_path_advect_only = os.path.join(savedir, "images_advect_frame0_only", "test", f"cam_{idx:03d}_frame_{frame_idx:03d}_render.png")
			imageio.imwrite(render_path_advect_only, render_img_advect_only)
			
			del render_result_advect_only, render_tensor_advect_only, render_np_advect_only, render_img_advect_only
			del wrapped_gaussians_advect_only, frame_gaussians_advect_only, pos_world_advect_only
			del gt_tensor, gt_image, gt_np
			torch.cuda.empty_cache()
	
	print(f"\n===  ===")
	
	if len(train_camera_renders) > 0:
		train_render_video = np.stack(train_camera_renders, axis=0)
		train_render_video_path = os.path.join(savedir, "video", "train", "render_video.mp4")
		imageio.mimwrite(train_render_video_path, train_render_video, fps=30, quality=8)
		print(f"train viewrender video: {train_render_video_path} ({len(train_camera_renders)} frame)")
		
		train_gt_video = np.stack(train_camera_gts, axis=0)
		train_gt_video_path = os.path.join(savedir, "video", "train", "gt_video.mp4")
		imageio.mimwrite(train_gt_video_path, train_gt_video, fps=30, quality=8)
		print(f"train view GT : {train_gt_video_path}")
	
	if len(test_camera_renders) > 0:
		test_render_video = np.stack(test_camera_renders, axis=0)
		test_render_video_path = os.path.join(savedir, "video", "test", "render_video.mp4")
		imageio.mimwrite(test_render_video_path, test_render_video, fps=30, quality=8)
		print(f"Test-view render video: {test_render_video_path} ({len(test_camera_renders)} frame)")
		
		test_gt_video = np.stack(test_camera_gts, axis=0)
		test_gt_video_path = os.path.join(savedir, "video", "test", "gt_video.mp4")
		imageio.mimwrite(test_gt_video_path, test_gt_video, fps=30, quality=8)
		print(f"Test-view GT video: {test_gt_video_path}")
	
	if is_sliding_window:
		if len(train_camera_renders_advect_only) > 0:
			train_render_video_advect_only = np.stack(train_camera_renders_advect_only, axis=0)
			train_render_video_path_advect_only = os.path.join(savedir, "video_advect_frame0_only", "train", "render_video.mp4")
			imageio.mimwrite(train_render_video_path_advect_only, train_render_video_advect_only, fps=30, quality=8)
			print(f"train viewadvection only render video: {train_render_video_path_advect_only} ({len(train_camera_renders_advect_only)} frame)")
		
		if len(test_camera_renders_advect_only) > 0:
			test_render_video_advect_only = np.stack(test_camera_renders_advect_only, axis=0)
			test_render_video_path_advect_only = os.path.join(savedir, "video_advect_frame0_only", "test", "render_video.mp4")
			imageio.mimwrite(test_render_video_path_advect_only, test_render_video_advect_only, fps=30, quality=8)
			print(f"Test-view advection-only render video: {test_render_video_path_advect_only} ({len(test_camera_renders_advect_only)} frame)")
	
	print(f"\n=== evaluatemetrics ===")
	
	if len(train_camera_psnrs) > 0:
		mean_train_psnr = np.mean(train_camera_psnrs)
		mean_train_ssim = np.mean(train_camera_ssims)
		mean_train_dssim = np.mean(train_camera_dssims)
		mean_train_lpips_vgg = np.mean(train_camera_lpips_vgg)
		mean_train_lpips_alex = np.mean(train_camera_lpips_alex)
		print(f"train view - PSNR: {mean_train_psnr:.2f} dB, SSIM: {mean_train_ssim:.4f}, DSSIM: {mean_train_dssim:.4f}, LPIPS-VGG: {mean_train_lpips_vgg:.4f}, LPIPS-Alex: {mean_train_lpips_alex:.4f}")
	else:
		mean_train_psnr = float('nan')
		mean_train_ssim = float('nan')
		mean_train_dssim = float('nan')
		mean_train_lpips_vgg = float('nan')
		mean_train_lpips_alex = float('nan')
		print("Warning: train view")
	
	if len(test_camera_psnrs) > 0:
		mean_test_psnr = np.mean(test_camera_psnrs)
		mean_test_ssim = np.mean(test_camera_ssims)
		mean_test_dssim = np.mean(test_camera_dssims)
		mean_test_lpips_vgg = np.mean(test_camera_lpips_vgg)
		mean_test_lpips_alex = np.mean(test_camera_lpips_alex)
		print(f"Mean test view - PSNR: {mean_test_psnr:.2f} dB, SSIM: {mean_test_ssim:.4f}, DSSIM: {mean_test_dssim:.4f}, LPIPS-VGG: {mean_test_lpips_vgg:.4f}, LPIPS-Alex: {mean_test_lpips_alex:.4f}")
	else:
		mean_test_psnr = float('nan')
		mean_test_ssim = float('nan')
		mean_test_dssim = float('nan')
		mean_test_lpips_vgg = float('nan')
		mean_test_lpips_alex = float('nan')
		print("Warning: No test-view data available")
	
	metrics_path = os.path.join(savedir, "metrics.txt")
	file_exists = os.path.exists(metrics_path)
	mode = "a" if file_exists else "w"
	
	with open(metrics_path, mode) as f:
		if file_exists:
			f.write("\n" + "="*50 + "\n")
		f.write("=== Gaussian Advection Evaluation Metrics ===\n")
		f.write("\n=== Train View Metrics ===\n")
		f.write(f"Mean PSNR: {mean_train_psnr:.4f} dB\n")
		f.write(f"Mean SSIM: {mean_train_ssim:.4f}\n")
		f.write(f"Mean DSSIM: {mean_train_dssim:.4f}\n")
		f.write(f"Mean LPIPS-VGG: {mean_train_lpips_vgg:.4f}\n")
		f.write(f"Mean LPIPS-Alex: {mean_train_lpips_alex:.4f}\n")
		f.write(f"Total cameras: {len(train_camera_psnrs)}\n")
		
		f.write(f"\nPer-camera train metrics:\n")
		for idx in range(len(train_camera_psnrs)):
			f.write(f"  Camera {idx}: Frame {train_camera_frame_indices[idx]}, PSNR={train_camera_psnrs[idx]:.4f} dB, SSIM={train_camera_ssims[idx]:.4f}, DSSIM={train_camera_dssims[idx]:.4f}, LPIPS-VGG={train_camera_lpips_vgg[idx]:.4f}, LPIPS-Alex={train_camera_lpips_alex[idx]:.4f}\n")
		
		f.write("\n=== Test View Metrics ===\n")
		f.write(f"Mean PSNR: {mean_test_psnr:.4f} dB\n")
		f.write(f"Mean SSIM: {mean_test_ssim:.4f}\n")
		f.write(f"Mean DSSIM: {mean_test_dssim:.4f}\n")
		f.write(f"Mean LPIPS-VGG: {mean_test_lpips_vgg:.4f}\n")
		f.write(f"Mean LPIPS-Alex: {mean_test_lpips_alex:.4f}\n")
		f.write(f"Total cameras: {len(test_camera_psnrs)}\n")
		
		f.write(f"\nPer-camera test metrics:\n")
		for idx in range(len(test_camera_psnrs)):
			f.write(f"  Camera {idx}: Frame {test_camera_frame_indices[idx]}, PSNR={test_camera_psnrs[idx]:.4f} dB, SSIM={test_camera_ssims[idx]:.4f}, DSSIM={test_camera_dssims[idx]:.4f}, LPIPS-VGG={test_camera_lpips_vgg[idx]:.4f}, LPIPS-Alex={test_camera_lpips_alex[idx]:.4f}\n")
		
		if is_sliding_window:
			if len(train_camera_psnrs_advect_only) > 0:
				mean_train_psnr_advect_only = np.mean(train_camera_psnrs_advect_only)
				mean_train_ssim_advect_only = np.mean(train_camera_ssims_advect_only)
				mean_train_dssim_advect_only = np.mean(train_camera_dssims_advect_only)
				mean_train_lpips_vgg_advect_only = np.mean(train_camera_lpips_vgg_advect_only)
				mean_train_lpips_alex_advect_only = np.mean(train_camera_lpips_alex_advect_only)
				print(f"train view（advection only， inflow）- PSNR: {mean_train_psnr_advect_only:.2f} dB, SSIM: {mean_train_ssim_advect_only:.4f}, DSSIM: {mean_train_dssim_advect_only:.4f}, LPIPS-VGG: {mean_train_lpips_vgg_advect_only:.4f}, LPIPS-Alex: {mean_train_lpips_alex_advect_only:.4f}")
			else:
				mean_train_psnr_advect_only = float('nan')
				mean_train_ssim_advect_only = float('nan')
				mean_train_dssim_advect_only = float('nan')
				mean_train_lpips_vgg_advect_only = float('nan')
				mean_train_lpips_alex_advect_only = float('nan')
			
			if len(test_camera_psnrs_advect_only) > 0:
				mean_test_psnr_advect_only = np.mean(test_camera_psnrs_advect_only)
				mean_test_ssim_advect_only = np.mean(test_camera_ssims_advect_only)
				mean_test_dssim_advect_only = np.mean(test_camera_dssims_advect_only)
				mean_test_lpips_vgg_advect_only = np.mean(test_camera_lpips_vgg_advect_only)
				mean_test_lpips_alex_advect_only = np.mean(test_camera_lpips_alex_advect_only)
				print(f"Mean test view（advection only， inflow）- PSNR: {mean_test_psnr_advect_only:.2f} dB, SSIM: {mean_test_ssim_advect_only:.4f}, DSSIM: {mean_test_dssim_advect_only:.4f}, LPIPS-VGG: {mean_test_lpips_vgg_advect_only:.4f}, LPIPS-Alex: {mean_test_lpips_alex_advect_only:.4f}")
			else:
				mean_test_psnr_advect_only = float('nan')
				mean_test_ssim_advect_only = float('nan')
				mean_test_dssim_advect_only = float('nan')
				mean_test_lpips_vgg_advect_only = float('nan')
				mean_test_lpips_alex_advect_only = float('nan')
			
			f.write("\n" + "="*50 + "\n")
			f.write("=== Gaussian Advection Only (Frame 0, No Inflow) Evaluation Metrics ===\n")
			f.write("\n=== Train View Metrics (Advect Only) ===\n")
			f.write(f"Mean PSNR: {mean_train_psnr_advect_only:.4f} dB\n")
			f.write(f"Mean SSIM: {mean_train_ssim_advect_only:.4f}\n")
			f.write(f"Mean DSSIM: {mean_train_dssim_advect_only:.4f}\n")
			f.write(f"Mean LPIPS-VGG: {mean_train_lpips_vgg_advect_only:.4f}\n")
			f.write(f"Mean LPIPS-Alex: {mean_train_lpips_alex_advect_only:.4f}\n")
			f.write(f"Total cameras: {len(train_camera_psnrs_advect_only)}\n")
			
			f.write(f"\nPer-camera train metrics (advect only):\n")
			for idx in range(len(train_camera_psnrs_advect_only)):
				f.write(f"  Camera {idx}: Frame {train_camera_frame_indices[idx]}, PSNR={train_camera_psnrs_advect_only[idx]:.4f} dB, SSIM={train_camera_ssims_advect_only[idx]:.4f}, DSSIM={train_camera_dssims_advect_only[idx]:.4f}, LPIPS-VGG={train_camera_lpips_vgg_advect_only[idx]:.4f}, LPIPS-Alex={train_camera_lpips_alex_advect_only[idx]:.4f}\n")
			
			f.write("\n=== Test View Metrics (Advect Only) ===\n")
			f.write(f"Mean PSNR: {mean_test_psnr_advect_only:.4f} dB\n")
			f.write(f"Mean SSIM: {mean_test_ssim_advect_only:.4f}\n")
			f.write(f"Mean DSSIM: {mean_test_dssim_advect_only:.4f}\n")
			f.write(f"Mean LPIPS-VGG: {mean_test_lpips_vgg_advect_only:.4f}\n")
			f.write(f"Mean LPIPS-Alex: {mean_test_lpips_alex_advect_only:.4f}\n")
			f.write(f"Total cameras: {len(test_camera_psnrs_advect_only)}\n")
			
			f.write(f"\nPer-camera test metrics (advect only):\n")
			for idx in range(len(test_camera_psnrs_advect_only)):
				f.write(f"  Camera {idx}: Frame {test_camera_frame_indices[idx]}, PSNR={test_camera_psnrs_advect_only[idx]:.4f} dB, SSIM={test_camera_ssims_advect_only[idx]:.4f}, DSSIM={test_camera_dssims_advect_only[idx]:.4f}, LPIPS-VGG={test_camera_lpips_vgg_advect_only[idx]:.4f}, LPIPS-Alex={test_camera_lpips_alex_advect_only[idx]:.4f}\n")
	
	print(f"Metrics saved to: {metrics_path}")
	print(f"\n=== evaluatecomplete ===")
	print(f"Results saved to: {savedir}")


def evaluate_sliding_window_window(args, savedir: str, start_frame: int, end_frame: int, 
								   scale: float = None):
	"""
	evaluate sliding window 
	
	Args:
		args: argument object
		savedir: sliding window output directory（contains frame_gaussians/, frame_velocities/, window_*_*/ ）
		start_frame: start frame
		end_frame: end frame（contains）
		scale: resize scale（optional， args ）
	
	Returns:
		metrics_dict: contains metrics 
	"""
	device = set_device(args)
	
	lpips_vgg_model = LPIPS(net_type='vgg', version='0.1').to(device).eval()
	lpips_alex_model = LPIPS(net_type='alex', version='0.1').to(device).eval()
	
	if scale is None:
		scale = getattr(args, 'scale', 1)
	
	with open(os.path.join(args.datadir, 'info.json'), 'r') as fp:
		meta = json.load(fp)
		voxel_tran = np.float32(meta['voxel_matrix'])
		voxel_tran = np.stack([voxel_tran[:,2],voxel_tran[:,1],voxel_tran[:,0],voxel_tran[:,3]],axis=1)
		voxel_scale = np.broadcast_to(meta['voxel_scale'],[3])
		voxel_scale = voxel_scale * args.scene_scale
		voxel_tran[:3,3] *= args.scene_scale
		train_video = meta['train_videos'][0]
		total_frame_num = train_video['frame_num']
		eval_frame_limit = getattr(args, 'eval_frame_limit', None)
		if eval_frame_limit is not None:
			total_frame_num = min(total_frame_num, max(1, eval_frame_limit))
		args.frame_num = total_frame_num
	
	coord_trans = CoordinateTransform(voxel_tran, voxel_scale, device)
	lengths = np.array([args.Nx, args.Ny, args.Nz])
	lengths_tensor = torch.from_numpy(lengths).float().to(device)
	
	s = float(scale)
	nx, ny, nz = int(args.Nx/s), int(args.Ny/s), int(args.Nz/s)
	grid_shape = (nx, ny, nz)
	min_corner = np.zeros(3)
	max_corner = lengths
	grid_points_np = utils_grid.generate_gridpoints(grid_shape, min_corner, max_corner)
	grid_points = torch.from_numpy(grid_points_np).float().to(device)
	
	if hasattr(args, 'sim_steps') and args.sim_steps is not None:
		dt = args.sim_steps / s
	else:
		dt = 1.0 / s
	
	print(f"\n{'='*80}")
	print(f"Evaluating Sliding Window: {start_frame}->{end_frame}")
	print(f"{'='*80}")
	print(f"Window: frames {start_frame} to {end_frame-1}")
	print(f"Grid shape: {grid_shape}, dt: {dt:.6f}")
	
	# ============================================================
	# ============================================================
	print(f"\n[Step 1] Loading initialGS...")
	from argparse import ArgumentParser
	from arguments import ModelParams, ModelHiddenParams, OptimizationParams
	from scene.gaussian_model import GaussianModel
	from scene import Scene
	
	parser = ArgumentParser()
	model_params_obj = ModelParams(parser, sentinel=True)
	hyperparam_obj = ModelHiddenParams(parser)
	op = OptimizationParams(parser)
	
	model_params = model_params_obj.extract(args)
	hyperparam = hyperparam_obj.extract(args)
	opt = op.extract(args)
	
	if not hasattr(model_params, 'eval'):
		model_params.eval = getattr(args, 'eval', True)
	if not hasattr(model_params, 'white_background'):
		model_params.white_background = getattr(args, 'white_background', True)
	if not hasattr(model_params, 'extension'):
		model_params.extension = getattr(args, 'extension', '.png')
	if not hasattr(model_params, 'images'):
		model_params.images = getattr(args, 'images', 'images')
	if not hasattr(model_params, 'llffhold'):
		model_params.llffhold = getattr(args, 'llffhold', 8)
	if not hasattr(model_params, 'add_points'):
		model_params.add_points = getattr(args, 'add_points', False)
	if not hasattr(model_params, 'num_init_points'):
		model_params.num_init_points = getattr(args, 'num_init_points', 2000)
	if not hasattr(model_params, 'half_res'):
		model_params.half_res = getattr(args, 'half_res', False)
	
	gaussians = GaussianModel(model_params.sh_degree, hyperparam)
	
	if start_frame == 0:
		gaussian_state_path = os.path.join(savedir, "frame_gaussians", f"frame_000_gaussian.pth")
		if not os.path.exists(gaussian_state_path):
			raise FileNotFoundError(f"Frame 0 Gaussian state not found: {gaussian_state_path}")
		gaussian_state, _ = torch.load(gaussian_state_path)
		gaussians.restore(gaussian_state, opt)
		print(f"  Loaded initialGS from frame 0: {gaussian_state_path}")
	else:
		prev_gaussian_state_path = os.path.join(savedir, "frame_gaussians", f"frame_{start_frame-1:03d}_gaussian.pth")
		if not os.path.exists(prev_gaussian_state_path):
			raise FileNotFoundError(f"Frame {start_frame-1} Gaussian state not found: {prev_gaussian_state_path}")
		prev_gaussian_state, _ = torch.load(prev_gaussian_state_path)
		gaussians.restore(prev_gaussian_state, opt)
		print(f"  Loaded GS state from frame {start_frame-1} for position: {prev_gaussian_state_path}")
		
		prev_window_dir = None
		window_pattern = os.path.join(savedir, "window_*_*")
		window_dirs = glob.glob(window_pattern)
		for window_dir in window_dirs:
			window_name = os.path.basename(window_dir)
			match = re.match(r'window_(\d+)_(\d+)', window_name)
			if match:
				w_start = int(match.group(1))
				w_end = int(match.group(2))
				if w_start <= start_frame - 1 < w_end:
					prev_window_dir = window_dir
					break
		
		if prev_window_dir:
			prev_inflow_pattern = os.path.join(prev_window_dir, "inflow_gaussians_epoch_*.pth")
			prev_inflow_files = glob.glob(prev_inflow_pattern)
			if prev_inflow_files:
				latest_prev_inflow_file = None
				max_epoch = -1
				for inflow_file in prev_inflow_files:
					match = re.search(r'inflow_gaussians_epoch_(\d+)\.pth', os.path.basename(inflow_file))
					if match:
						epoch_num = int(match.group(1))
						if epoch_num > max_epoch:
							max_epoch = epoch_num
							latest_prev_inflow_file = inflow_file
				
				if latest_prev_inflow_file is None:
					latest_prev_inflow_file = max(prev_inflow_files, key=os.path.getctime)
				
				try:
					prev_inflow_checkpoint_data = torch.load(latest_prev_inflow_file, map_location=device)
					if isinstance(prev_inflow_checkpoint_data, tuple) and len(prev_inflow_checkpoint_data) >= 2:
						prev_inflow_state = prev_inflow_checkpoint_data[0]
						prev_inflow_gaussians = InflowGaussians.restore(
							prev_inflow_state,
							gaussians_template=gaussians,
							coord_trans=coord_trans,
							device=device
						)
						
						merge_group_indices = []
						for group_idx in range(prev_inflow_gaussians.num_groups):
							origin_frame = prev_inflow_gaussians.get_group_origin_frame(group_idx)
							if origin_frame is not None and origin_frame == start_frame - 1:
								merge_group_indices.append(group_idx)
						
						if len(merge_group_indices) > 0:
							from velocity_training.train import _merge_inflow_to_gaussians
							gaussians, _ = _merge_inflow_to_gaussians(
								gaussians, prev_inflow_gaussians, merge_group_indices, 
								coord_trans, lengths_tensor, device
							)
							print(f"  Merged {len(merge_group_indices)} inflow groups from frame {start_frame-1} (origin_frame == {start_frame-1})")
				except Exception as e:
					print(f"  Warning: Could not load or merge inflow from frame {start_frame-1}: {e}")
		
		prev_vel_model_path = os.path.join(savedir, "frame_velocities", f"frame_{start_frame-1:03d}_velocity.pth")
		if not os.path.exists(prev_vel_model_path):
			prev_vel_model_path = None
			if prev_window_dir:
				vel_pattern = os.path.join(prev_window_dir, "ckpt", f"velrbf_frame_{start_frame-1:03d}_ckpt_*.pth")
				vel_files = glob.glob(vel_pattern)
				if vel_files:
					prev_vel_model_path = max(vel_files, key=os.path.getctime)
		
		if prev_vel_model_path and os.path.exists(prev_vel_model_path):
			prev_velocity_model = TiDFRBF.load(prev_vel_model_path, device=device)
			print(f"  Loaded velocity model for frame {start_frame-1}: {prev_vel_model_path}")
			
			with torch.no_grad():
				xyz_world_i = gaussians.get_xyz.detach().clone()
				xyz_smoke_i = coord_trans.world2smoke(xyz_world_i)
				xyz_sim_i = xyz_smoke_i * lengths_tensor
				
				v_flat = prev_velocity_model(grid_points)
				v_vol = v_flat.view(nx, ny, nz, 3).permute(3, 2, 1, 0).unsqueeze(0)
				
				norm_pos = xyz_sim_i.clone()
				norm_pos = 2 * (norm_pos / lengths_tensor) - 1.0
				grid_in = norm_pos.view(1, 1, 1, -1, 3)
				v_part = F.grid_sample(v_vol, grid_in, align_corners=True, mode='bilinear')
				v_part = v_part.view(3, -1).permute(1, 0)
				
				xyz_sim_i_plus_1 = xyz_sim_i + v_part * dt
				
				xyz_smoke_i_plus_1 = xyz_sim_i_plus_1 / lengths_tensor
				xyz_world_i_plus_1 = coord_trans.smoke2world(xyz_smoke_i_plus_1)
				
				advected_xyz = xyz_world_i_plus_1.detach().clone()
				print(f"  Advected {xyz_world_i.shape[0]} points from frame {start_frame-1} to frame {start_frame}")
		else:
			print(f"  Warning: Velocity model for frame {start_frame-1} not found, using frame {start_frame-1} position directly")
			advected_xyz = gaussians.get_xyz.detach().clone()
		
		gaussian_state_path = os.path.join(savedir, "frame_gaussians", f"frame_{start_frame:03d}_gaussian.pth")
		if not os.path.exists(gaussian_state_path):
			raise FileNotFoundError(f"Frame {start_frame} Gaussian state not found: {gaussian_state_path}")
		gaussian_state, _ = torch.load(gaussian_state_path)
		
		current_num_points = gaussians.get_xyz.shape[0]
		
		# (active_sh_degree, _xyz, _deformation.state_dict(), _deformation_table,
		#  _features_dc, _features_rest, _scaling, _rotation, _opacity, ...)
		if isinstance(gaussian_state, tuple):
			if len(gaussian_state) >= 2:
				target_num_points = gaussian_state[1].shape[0]
			else:
				raise ValueError(f"Invalid gaussian_state tuple length: {len(gaussian_state)}")
		elif isinstance(gaussian_state, dict):
			if '_xyz' in gaussian_state:
				target_num_points = gaussian_state['_xyz'].shape[0]
			elif 'xyz' in gaussian_state:
				target_num_points = gaussian_state['xyz'].shape[0]
			elif '_opacity' in gaussian_state:
				target_num_points = gaussian_state['_opacity'].shape[0]
			elif 'opacity' in gaussian_state:
				target_num_points = gaussian_state['opacity'].shape[0]
			else:
				raise ValueError(f"Cannot determine number of points from gaussian_state dict")
		else:
			raise ValueError(f"Unexpected gaussian_state type: {type(gaussian_state)}")
		
		if current_num_points != target_num_points:
			print(f"  Warning: Point count mismatch (current: {current_num_points}, target: {target_num_points})")
			if current_num_points > target_num_points:
				print(f"    Using first {target_num_points} points from advected positions")
				advected_xyz_to_use = advected_xyz[:target_num_points]
			else:
				print(f"    Frame {start_frame} GS checkpoint has more points, using advected positions for first {current_num_points} points")
				advected_xyz_to_use = advected_xyz
			
			gaussians.restore(gaussian_state, opt)
			if current_num_points > target_num_points:
				gaussians._xyz.data = advected_xyz_to_use.requires_grad_(False)
			else:
				gaussians._xyz.data[:current_num_points] = advected_xyz_to_use.requires_grad_(False)
			print(f"  Loaded GS attributes from frame {start_frame} and applied advected positions (partial): {gaussian_state_path}")
		else:
			gaussians.restore(gaussian_state, opt)
			gaussians._xyz.data = advected_xyz.requires_grad_(False)
			print(f"  Loaded GS attributes from frame {start_frame} and applied advected positions: {gaussian_state_path}")
	
	scene = Scene(model_params, gaussians, load_iteration=None, shuffle=False)
	
	# ============================================================
	# ============================================================
	print(f"\n[Step 2] Loading velocity models...")
	vel_models = []
	for vel_frame_idx in range(start_frame, end_frame - 1):
		vel_model_path = os.path.join(savedir, "frame_velocities", f"frame_{vel_frame_idx:03d}_velocity.pth")
		if not os.path.exists(vel_model_path):
			window_dir = os.path.join(savedir, f"window_{start_frame}_{end_frame}")
			vel_pattern = os.path.join(window_dir, "ckpt", f"velrbf_frame_{vel_frame_idx:03d}_ckpt_*.pth")
			vel_files = glob.glob(vel_pattern)
			if vel_files:
				vel_model_path = max(vel_files, key=os.path.getctime)
			else:
				raise FileNotFoundError(f"Velocity model for frame {vel_frame_idx} not found")
		
		vel_model = TiDFRBF.load(vel_model_path, device=device)
		vel_models.append(vel_model)
		print(f"  Loaded velocity model for frame {vel_frame_idx}: {vel_model_path}")
	
	print(f"  Loaded {len(vel_models)} velocity models (frames {start_frame} to {end_frame-2})")
	
	# ============================================================
	# ============================================================
	print(f"\n[Step 3] Loading inflow gaussians...")
	inflow_gaussians = None
	window_dir = os.path.join(savedir, f"window_{start_frame}_{end_frame}")
	inflow_pattern = os.path.join(window_dir, "inflow_gaussians_epoch_*.pth")
	inflow_files = glob.glob(inflow_pattern)
	
	if inflow_files:
		latest_inflow_file = None
		max_epoch = -1
		for inflow_file in inflow_files:
			match = re.search(r'inflow_gaussians_epoch_(\d+)\.pth', os.path.basename(inflow_file))
			if match:
				epoch_num = int(match.group(1))
				if epoch_num > max_epoch:
					max_epoch = epoch_num
					latest_inflow_file = inflow_file
		
		if latest_inflow_file is None:
			latest_inflow_file = max(inflow_files, key=os.path.getctime)
		
		try:
			inflow_checkpoint_data = torch.load(latest_inflow_file, map_location=device)
			if isinstance(inflow_checkpoint_data, tuple) and len(inflow_checkpoint_data) >= 2:
				inflow_state = inflow_checkpoint_data[0]
				inflow_gaussians = InflowGaussians.restore(
					inflow_state,
					gaussians_template=gaussians,
					coord_trans=coord_trans,
					device=device
				)
				print(f"  Loaded inflow gaussians from: {latest_inflow_file}")
				print(f"    Groups: {inflow_gaussians.num_groups}, Points per group: {inflow_gaussians.num_points_per_group}")
		except Exception as e:
			print(f"  Warning: Could not load inflow gaussians: {e}")
			inflow_gaussians = None
	else:
		print(f"  No inflow gaussians found in {window_dir}")
	
	# ============================================================
	# ============================================================
	print(f"\n[Step 4] Computing camera mapping...")
	train_cameras = scene.getTrainCameras()
	test_cameras = scene.getTestCameras()
	
	use_test_in_training = getattr(args, 'use_test_in_training', False)
	if use_test_in_training:
		train_cameras = combine_train_test_datasets(train_cameras, test_cameras)
	
	frame_to_train_cameras = {}
	frame_to_test_cameras = {}
	for t in range(process_frame_num):
		frame_to_train_cameras[t] = []
		frame_to_test_cameras[t] = []
	
	for cam_idx, cam in enumerate(train_cameras):
		if hasattr(cam, 'time') and cam.time is not None:
			if total_frame_num > 1:
				frame_idx = round(cam.time * (total_frame_num - 1))
			else:
				frame_idx = 0
			frame_idx = max(0, min(frame_idx, total_frame_num - 1))
			if frame_idx >= process_frame_num:
				continue
			frame_to_train_cameras[frame_idx].append((cam_idx, cam))
	
	for cam_idx, cam in enumerate(test_cameras):
		if hasattr(cam, 'time') and cam.time is not None:
			if total_frame_num > 1:
				frame_idx = round(cam.time * (total_frame_num - 1))
			else:
				frame_idx = 0
			frame_idx = max(0, min(frame_idx, total_frame_num - 1))
			if frame_idx >= process_frame_num:
				continue
			frame_to_test_cameras[frame_idx].append((cam_idx, cam))
	
	print(f"  Mapped cameras for {total_frame_num} frames")
	
	# ============================================================
	# ============================================================
	pp = PipelineParams(ArgumentParser())
	pipe = pp.extract(args)
	background = get_background_color(args, device)
	
	# ============================================================
	# ============================================================
	print(f"\n[Step 5] Advecting and rendering frames {start_frame} to {end_frame-1}...")
	
	current_pos_sim = None
	if start_frame == 0:
		xyz_world_0 = gaussians.get_xyz.detach().clone()
		xyz_smoke_0 = coord_trans.world2smoke(xyz_world_0)
		current_pos_sim = xyz_smoke_0 * lengths_tensor
	else:
		xyz_world_start = gaussians.get_xyz.detach().clone()
		xyz_smoke_start = coord_trans.world2smoke(xyz_world_start)
		current_pos_sim = xyz_smoke_start * lengths_tensor
	
	train_camera_psnrs = []
	train_camera_ssims = []
	train_camera_lpips_vgg = []
	train_camera_lpips_alex = []
	train_camera_frame_indices = []
	train_camera_renders = []
	train_camera_gts = []
	train_camera_info = []
	
	test_camera_psnrs = []
	test_camera_ssims = []
	test_camera_lpips_vgg = []
	test_camera_lpips_alex = []
	test_camera_frame_indices = []
	test_camera_renders = []
	test_camera_gts = []
	test_camera_info = []
	
	for t in range(start_frame, end_frame):
		print(f"\n  Processing frame {t}...")
		
		has_inflow = (t > start_frame and inflow_gaussians is not None)
		inflow_group_indices = []
		if has_inflow:
			for group_idx in range(inflow_gaussians.num_groups):
				origin_frame = inflow_gaussians.get_group_origin_frame(group_idx)
				if origin_frame is not None and start_frame <= origin_frame < t:
					inflow_group_indices.append(group_idx)
			print(f"    Inflow groups for frame {t}: {inflow_group_indices} (origin_frames: {[inflow_gaussians.get_group_origin_frame(g) for g in inflow_group_indices]})")
		
		if t > start_frame:
			vel_model_idx = t - 1 - start_frame
			if vel_model_idx < len(vel_models):
				with torch.no_grad():
					v_flat = vel_models[vel_model_idx](grid_points)
					v_vol = v_flat.view(nx, ny, nz, 3).permute(3, 2, 1, 0).unsqueeze(0)
					
					norm_pos = current_pos_sim.clone()
					norm_pos = 2 * (norm_pos / lengths_tensor) - 1.0
					grid_in = norm_pos.view(1, 1, 1, -1, 3)
					v_part = F.grid_sample(v_vol, grid_in, align_corners=True, mode='bilinear')
					v_part = v_part.view(3, -1).permute(1, 0)
					
					current_pos_sim = current_pos_sim + v_part * dt
					print(f"    Advected to frame {t} using velocity model {vel_model_idx}")
		
		pos_smoke = current_pos_sim / lengths_tensor
		pos_world = coord_trans.smoke2world(pos_smoke)
		
		orig_num_points = gaussians.get_xyz.shape[0]
		if has_inflow and len(inflow_group_indices) > 0:
			inflow_positions = []
			for group_idx in inflow_group_indices:
				group_origin_frame = inflow_gaussians.get_group_origin_frame(group_idx)
				if group_origin_frame is not None:
					group_xyz_sim = inflow_gaussians.get_group_xyz_sim(group_idx)
					
					group_pos_sim = group_xyz_sim.clone()
					
					advect_start = max(group_origin_frame, start_frame)
					for advect_step in range(advect_start, t):
						if advect_step >= end_frame - 1:
							break
						vel_idx = advect_step - start_frame
						if vel_idx >= 0 and vel_idx < len(vel_models):
							with torch.no_grad():
								v_flat = vel_models[vel_idx](grid_points)
								v_vol = v_flat.view(nx, ny, nz, 3).permute(3, 2, 1, 0).unsqueeze(0)
								
								norm_pos = group_pos_sim.clone()
								norm_pos = 2 * (norm_pos / lengths_tensor) - 1.0
								grid_in = norm_pos.view(1, 1, 1, -1, 3)
								v_part = F.grid_sample(v_vol, grid_in, align_corners=True, mode='bilinear')
								v_part = v_part.view(3, -1).permute(1, 0)
								
								group_pos_sim = group_pos_sim + v_part * dt
					
					group_pos_smoke = group_pos_sim / lengths_tensor
					group_pos_world = coord_trans.smoke2world(group_pos_smoke)
					inflow_positions.append(group_pos_world)
			
			if inflow_positions:
				all_pos_world = torch.cat([pos_world] + inflow_positions, dim=0)
			else:
				all_pos_world = pos_world
		else:
			all_pos_world = pos_world
		
		train_cameras_for_frame = frame_to_train_cameras.get(t, [])
		test_cameras_for_frame = frame_to_test_cameras.get(t, [])
		
		print(f"    Train cameras: {len(train_cameras_for_frame)}, Test cameras: {len(test_cameras_for_frame)}")
		
		for cam_idx, cam in train_cameras_for_frame:
			with torch.no_grad():
				curr_pos_world = all_pos_world.clone()
				
				if has_inflow and len(inflow_group_indices) > 0:
					wrapped_gaussians = ExtendedGaussianWrapper(gaussians, curr_pos_world, inflow_gaussians, inflow_group_indices)
				else:
					wrapped_gaussians = GaussianOverrideWrapper(gaussians, curr_pos_world)
				
				render_result = render(cam, wrapped_gaussians, pipe, background, stage="coarse")["render"]
				gt_image = cam.original_image.cuda()
				
				render_tensor = render_result.unsqueeze(0)  # [1, 3, H, W]
				gt_tensor = gt_image.unsqueeze(0)  # [1, 3, H, W]
				
				psnr_val = psnr(render_tensor, gt_tensor).item()
				ssim_val = ssim(render_tensor, gt_tensor)
				if isinstance(ssim_val, torch.Tensor):
					ssim_val = ssim_val.item()
				
				torch.cuda.empty_cache()
				lpips_vgg_val = lpips_vgg_model(render_tensor, gt_tensor).squeeze().item()
				lpips_alex_val = lpips_alex_model(render_tensor, gt_tensor).squeeze().item()
				
				train_camera_psnrs.append(psnr_val)
				train_camera_ssims.append(ssim_val)
				train_camera_lpips_vgg.append(lpips_vgg_val)
				train_camera_lpips_alex.append(lpips_alex_val)
				train_camera_frame_indices.append(t)
				
				render_np = render_result.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
				gt_np = gt_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
				render_img = (render_np * 255).astype(np.uint8)
				gt_img = (gt_np * 255).astype(np.uint8)
				
				train_camera_renders.append(render_img)
				train_camera_gts.append(gt_img)
				train_camera_info.append((t, cam_idx))
		
		for cam_idx, cam in test_cameras_for_frame:
			with torch.no_grad():
				curr_pos_world = all_pos_world.clone()
				
				if has_inflow and len(inflow_group_indices) > 0:
					wrapped_gaussians = ExtendedGaussianWrapper(gaussians, curr_pos_world, inflow_gaussians, inflow_group_indices)
				else:
					wrapped_gaussians = GaussianOverrideWrapper(gaussians, curr_pos_world)
				
				render_result = render(cam, wrapped_gaussians, pipe, background, stage="coarse")["render"]
				gt_image = cam.original_image.cuda()
				
				render_tensor = render_result.unsqueeze(0)  # [1, 3, H, W]
				gt_tensor = gt_image.unsqueeze(0)  # [1, 3, H, W]
				
				psnr_val = psnr(render_tensor, gt_tensor).item()
				ssim_val = ssim(render_tensor, gt_tensor)
				if isinstance(ssim_val, torch.Tensor):
					ssim_val = ssim_val.item()
				
				torch.cuda.empty_cache()
				lpips_vgg_val = lpips_vgg_model(render_tensor, gt_tensor).squeeze().item()
				lpips_alex_val = lpips_alex_model(render_tensor, gt_tensor).squeeze().item()
				
				test_camera_psnrs.append(psnr_val)
				test_camera_ssims.append(ssim_val)
				test_camera_lpips_vgg.append(lpips_vgg_val)
				test_camera_lpips_alex.append(lpips_alex_val)
				test_camera_frame_indices.append(t)
				
				render_np = render_result.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
				gt_np = gt_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
				render_img = (render_np * 255).astype(np.uint8)
				gt_img = (gt_np * 255).astype(np.uint8)
				
				test_camera_renders.append(render_img)
				test_camera_gts.append(gt_img)
				test_camera_info.append((t, cam_idx))
	
	# ============================================================
	# ============================================================
	print(f"\n[Step 6] Computing metrics and saving results...")
	
	eval_savedir = os.path.join(savedir, f"window_{start_frame}_{end_frame}", "evaluation")
	os.makedirs(eval_savedir, exist_ok=True)
	os.makedirs(os.path.join(eval_savedir, "images", "train"), exist_ok=True)
	os.makedirs(os.path.join(eval_savedir, "images", "test"), exist_ok=True)
	
	mean_train_psnr = np.mean(train_camera_psnrs) if train_camera_psnrs else float('nan')
	mean_train_ssim = np.mean(train_camera_ssims) if train_camera_ssims else float('nan')
	mean_train_lpips_vgg = np.mean(train_camera_lpips_vgg) if train_camera_lpips_vgg else float('nan')
	mean_train_lpips_alex = np.mean(train_camera_lpips_alex) if train_camera_lpips_alex else float('nan')
	
	mean_test_psnr = np.mean(test_camera_psnrs) if test_camera_psnrs else float('nan')
	mean_test_ssim = np.mean(test_camera_ssims) if test_camera_ssims else float('nan')
	mean_test_lpips_vgg = np.mean(test_camera_lpips_vgg) if test_camera_lpips_vgg else float('nan')
	mean_test_lpips_alex = np.mean(test_camera_lpips_alex) if test_camera_lpips_alex else float('nan')
	
	print(f"\n{'='*80}")
	print(f"Evaluation Results for Window {start_frame}->{end_frame}")
	print(f"{'='*80}")
	if train_camera_psnrs:
		print(f"Train View - PSNR: {mean_train_psnr:.2f} dB, SSIM: {mean_train_ssim:.4f}, LPIPS-VGG: {mean_train_lpips_vgg:.4f}, LPIPS-Alex: {mean_train_lpips_alex:.4f}")
		print(f"  Total cameras: {len(train_camera_psnrs)}")
	if test_camera_psnrs:
		print(f"Test View - PSNR: {mean_test_psnr:.2f} dB, SSIM: {mean_test_ssim:.4f}, LPIPS-VGG: {mean_test_lpips_vgg:.4f}, LPIPS-Alex: {mean_test_lpips_alex:.4f}")
		print(f"  Total cameras: {len(test_camera_psnrs)}")
	
	metrics_path = os.path.join(eval_savedir, "metrics.txt")
	with open(metrics_path, 'w', encoding='utf-8') as f:
		f.write("=" * 80 + "\n")
		f.write(f"Sliding Window Evaluation Results\n")
		f.write(f"Window: {start_frame}->{end_frame}\n")
		f.write("=" * 80 + "\n\n")
		
		if train_camera_psnrs:
			f.write("Train View Metrics:\n")
			f.write("-" * 80 + "\n")
			f.write(f"Mean PSNR: {mean_train_psnr:.4f} dB\n")
			f.write(f"Mean SSIM: {mean_train_ssim:.4f}\n")
			f.write(f"Mean LPIPS-VGG: {mean_train_lpips_vgg:.4f}\n")
			f.write(f"Mean LPIPS-Alex: {mean_train_lpips_alex:.4f}\n")
			f.write(f"Total cameras: {len(train_camera_psnrs)}\n")
			f.write("\nPer-camera train metrics:\n")
			for idx in range(len(train_camera_psnrs)):
				f.write(f"  Camera {idx}: Frame {train_camera_frame_indices[idx]}, PSNR={train_camera_psnrs[idx]:.4f} dB, SSIM={train_camera_ssims[idx]:.4f}, LPIPS-VGG={train_camera_lpips_vgg[idx]:.4f}, LPIPS-Alex={train_camera_lpips_alex[idx]:.4f}\n")
			f.write("\n")
		
		if test_camera_psnrs:
			f.write("Test View Metrics:\n")
			f.write("-" * 80 + "\n")
			f.write(f"Mean PSNR: {mean_test_psnr:.4f} dB\n")
			f.write(f"Mean SSIM: {mean_test_ssim:.4f}\n")
			f.write(f"Mean LPIPS-VGG: {mean_test_lpips_vgg:.4f}\n")
			f.write(f"Mean LPIPS-Alex: {mean_test_lpips_alex:.4f}\n")
			f.write(f"Total cameras: {len(test_camera_psnrs)}\n")
			f.write("\nPer-camera test metrics:\n")
			for idx in range(len(test_camera_psnrs)):
				f.write(f"  Camera {idx}: Frame {test_camera_frame_indices[idx]}, PSNR={test_camera_psnrs[idx]:.4f} dB, SSIM={test_camera_ssims[idx]:.4f}, LPIPS-VGG={test_camera_lpips_vgg[idx]:.4f}, LPIPS-Alex={test_camera_lpips_alex[idx]:.4f}\n")
	
	print(f"\nMetrics saved to: {metrics_path}")
	print(f"Results saved to: {eval_savedir}")
	
	# ============================================================
	# ============================================================
	print(f"\n[Step 7] Saving images and generating comparison videos...")
	
	train_images_dir = os.path.join(eval_savedir, "images", "train")
	os.makedirs(train_images_dir, exist_ok=True)
	for idx, (render_img, gt_img, (frame_idx, cam_idx)) in enumerate(zip(train_camera_renders, train_camera_gts, train_camera_info)):
		render_path = os.path.join(train_images_dir, f"cam_{cam_idx:03d}_frame_{frame_idx:03d}_render.png")
		gt_path = os.path.join(train_images_dir, f"cam_{cam_idx:03d}_frame_{frame_idx:03d}_gt.png")
		imageio.imwrite(render_path, render_img)
		imageio.imwrite(gt_path, gt_img)
	
	print(f"  Saved {len(train_camera_renders)} train view images")
	
	test_images_dir = os.path.join(eval_savedir, "images", "test")
	os.makedirs(test_images_dir, exist_ok=True)
	for idx, (render_img, gt_img, (frame_idx, cam_idx)) in enumerate(zip(test_camera_renders, test_camera_gts, test_camera_info)):
		render_path = os.path.join(test_images_dir, f"cam_{cam_idx:03d}_frame_{frame_idx:03d}_render.png")
		gt_path = os.path.join(test_images_dir, f"cam_{cam_idx:03d}_frame_{frame_idx:03d}_gt.png")
		imageio.imwrite(render_path, render_img)
		imageio.imwrite(gt_path, gt_img)
	
	print(f"  Saved {len(test_camera_renders)} test view images")
	
	if train_camera_renders:
		train_sorted_indices = sorted(range(len(train_camera_info)), key=lambda i: (train_camera_info[i][1], train_camera_info[i][0]))
		train_sorted_renders = [train_camera_renders[i] for i in train_sorted_indices]
		train_sorted_gts = [train_camera_gts[i] for i in train_sorted_indices]
		
		train_comparison_frames = []
		for render_img, gt_img in zip(train_sorted_renders, train_sorted_gts):
			h_render, w_render = render_img.shape[:2]
			h_gt, w_gt = gt_img.shape[:2]
			h = max(h_render, h_gt)
			
			if h_render != h:
				from PIL import Image
				render_img_pil = Image.fromarray(render_img)
				render_img_pil = render_img_pil.resize((w_render, h), Image.Resampling.LANCZOS)
				render_img = np.array(render_img_pil)
			if h_gt != h:
				from PIL import Image
				gt_img_pil = Image.fromarray(gt_img)
				gt_img_pil = gt_img_pil.resize((w_gt, h), Image.Resampling.LANCZOS)
				gt_img = np.array(gt_img_pil)
			
			comparison = np.concatenate([render_img, gt_img], axis=1)
			train_comparison_frames.append(comparison)
		
		train_video_path = os.path.join(eval_savedir, "comparison_train.mp4")
		imageio.mimwrite(train_video_path, train_comparison_frames, fps=10, quality=8)
		print(f"  Generated train view comparison video: {train_video_path}")
	
	if test_camera_renders:
		test_sorted_indices = sorted(range(len(test_camera_info)), key=lambda i: (test_camera_info[i][1], test_camera_info[i][0]))
		test_sorted_renders = [test_camera_renders[i] for i in test_sorted_indices]
		test_sorted_gts = [test_camera_gts[i] for i in test_sorted_indices]
		
		test_comparison_frames = []
		for render_img, gt_img in zip(test_sorted_renders, test_sorted_gts):
			h_render, w_render = render_img.shape[:2]
			h_gt, w_gt = gt_img.shape[:2]
			h = max(h_render, h_gt)
			
			if h_render != h:
				from PIL import Image
				render_img_pil = Image.fromarray(render_img)
				render_img_pil = render_img_pil.resize((w_render, h), Image.Resampling.LANCZOS)
				render_img = np.array(render_img_pil)
			if h_gt != h:
				from PIL import Image
				gt_img_pil = Image.fromarray(gt_img)
				gt_img_pil = gt_img_pil.resize((w_gt, h), Image.Resampling.LANCZOS)
				gt_img = np.array(gt_img_pil)
			
			comparison = np.concatenate([render_img, gt_img], axis=1)
			test_comparison_frames.append(comparison)
		
		test_video_path = os.path.join(eval_savedir, "comparison_test.mp4")
		imageio.mimwrite(test_video_path, test_comparison_frames, fps=10, quality=8)
		print(f"  Generated test view comparison video: {test_video_path}")
	
	return {
		'train': {
			'psnr': mean_train_psnr,
			'ssim': mean_train_ssim,
			'lpips_vgg': mean_train_lpips_vgg,
			'lpips_alex': mean_train_lpips_alex,
			'num_cameras': len(train_camera_psnrs),
			'per_camera': {
				'psnr': train_camera_psnrs,
				'ssim': train_camera_ssims,
				'lpips_vgg': train_camera_lpips_vgg,
				'lpips_alex': train_camera_lpips_alex,
				'frame_indices': train_camera_frame_indices
			}
		},
		'test': {
			'psnr': mean_test_psnr,
			'ssim': mean_test_ssim,
			'lpips_vgg': mean_test_lpips_vgg,
			'lpips_alex': mean_test_lpips_alex,
			'num_cameras': len(test_camera_psnrs),
			'per_camera': {
				'psnr': test_camera_psnrs,
				'ssim': test_camera_ssims,
				'lpips_vgg': test_camera_lpips_vgg,
				'lpips_alex': test_camera_lpips_alex,
				'frame_indices': test_camera_frame_indices
			}
		}
	}


def evaluate_full_reconstruction(args, savedir: str, scale: float = None, ckpt_load_path: str = None):
	"""
	evaluatefull reconstruction process， sliding window 
	
	This function will:：
	1. Loading from window 0->w start，frame 0 frame
	2. useframe 0 frame 1->w+1 ，frame 1 frame
	3. ，frame
	4. frame metrics（PSNR, SSIM, LPIPS）
	
	Args:
		args: argument object
		savedir: output directory（used forevaluate）
		scale: resize scale（optional， args ）
		ckpt_load_path: checkpoint path（contains frame_gaussians/, frame_velocities/, window_*_*/ ），if None use savedir
	
	Returns:
		metrics_dict: contains metrics 
	"""
	device = set_device(args)
	
	if ckpt_load_path is None:
		ckpt_load_path = savedir
	print(f"Loading checkpoints from: {ckpt_load_path}")
	print(f"Saving results to: {savedir}")
	
	lpips_vgg_model = LPIPS(net_type='vgg', version='0.1').to(device).eval()
	lpips_alex_model = LPIPS(net_type='alex', version='0.1').to(device).eval()
	
	if scale is None:
		scale = getattr(args, 'scale', 1)
	
	# ============================================================
	# ============================================================
	print(f"\n{'='*80}")
	print(f"Evaluating Full Reconstruction")
	print(f"{'='*80}")
	
	with open(os.path.join(args.datadir, 'info.json'), 'r') as fp:
		meta = json.load(fp)
		voxel_tran = np.float32(meta['voxel_matrix'])
		voxel_tran = np.stack([voxel_tran[:,2],voxel_tran[:,1],voxel_tran[:,0],voxel_tran[:,3]],axis=1)
		voxel_scale = np.broadcast_to(meta['voxel_scale'],[3])
		voxel_scale = voxel_scale * args.scene_scale
		voxel_tran[:3,3] *= args.scene_scale
		train_video = meta['train_videos'][0]
		total_frame_num = train_video['frame_num']
		args.frame_num = total_frame_num
	eval_frame_limit = getattr(args, 'eval_frame_limit', None)
	process_frame_num = total_frame_num
	if eval_frame_limit is not None:
		process_frame_num = min(total_frame_num, max(1, eval_frame_limit))
	
	window_pattern = os.path.join(ckpt_load_path, "window_*_*")
	window_dirs = glob.glob(window_pattern)
	if not window_dirs:
		raise ValueError(f"No window directories found in {ckpt_load_path}")
	
	first_window_name = os.path.basename(window_dirs[0])
	match = re.match(r'window_(\d+)_(\d+)', first_window_name)
	if not match:
		raise ValueError(f"Invalid window directory name: {first_window_name}")
	
	w_start = int(match.group(1))
	w_end = int(match.group(2))
	window_size = w_end - w_start
	print(f"Detected window size: {window_size} (from {first_window_name})")
	
	coord_trans = CoordinateTransform(voxel_tran, voxel_scale, device)
	lengths = np.array([args.Nx, args.Ny, args.Nz])
	lengths_tensor = torch.from_numpy(lengths).float().to(device)
	
	s = float(scale)
	nx, ny, nz = int(args.Nx/s), int(args.Ny/s), int(args.Nz/s)
	grid_shape = (nx, ny, nz)
	min_corner = np.zeros(3)
	max_corner = lengths
	grid_points_np = utils_grid.generate_gridpoints(grid_shape, min_corner, max_corner)
	grid_points = torch.from_numpy(grid_points_np).float().to(device)
	
	if hasattr(args, 'sim_steps') and args.sim_steps is not None:
		dt = args.sim_steps / s
	else:
		dt = 1.0 / s
	
	print(f"Grid shape: {grid_shape}, dt: {dt:.6f}")
	print(f"Total frames: {total_frame_num}; processing frames: {process_frame_num}")
	
	from argparse import ArgumentParser
	from arguments import ModelParams, ModelHiddenParams, OptimizationParams
	from scene.gaussian_model import GaussianModel
	from scene import Scene
	
	parser = ArgumentParser()
	model_params_obj = ModelParams(parser, sentinel=True)
	hyperparam_obj = ModelHiddenParams(parser)
	op = OptimizationParams(parser)
	
	model_params = model_params_obj.extract(args)
	hyperparam = hyperparam_obj.extract(args)
	opt = op.extract(args)
	
	if not hasattr(model_params, 'eval'):
		model_params.eval = getattr(args, 'eval', True)
	if not hasattr(model_params, 'white_background'):
		model_params.white_background = getattr(args, 'white_background', True)
	if not hasattr(model_params, 'extension'):
		model_params.extension = getattr(args, 'extension', '.png')
	if not hasattr(model_params, 'images'):
		model_params.images = getattr(args, 'images', 'images')
	if not hasattr(model_params, 'llffhold'):
		model_params.llffhold = getattr(args, 'llffhold', 8)
	if not hasattr(model_params, 'add_points'):
		model_params.add_points = getattr(args, 'add_points', False)
	if not hasattr(model_params, 'num_init_points'):
		model_params.num_init_points = getattr(args, 'num_init_points', 2000)
	if not hasattr(model_params, 'half_res'):
		model_params.half_res = getattr(args, 'half_res', False)
	
	# ============================================================
	# ============================================================
	print(f"\n[Step 1] Loading frame 0 initial GS...")
	gaussians = GaussianModel(model_params.sh_degree, hyperparam)
	
	gaussian_state_path = os.path.join(ckpt_load_path, "frame_gaussians", f"frame_000_gaussian.pth")
	if not os.path.exists(gaussian_state_path):
		raise FileNotFoundError(f"Frame 0 Gaussian state not found: {gaussian_state_path}")
	gaussian_state, _ = torch.load(gaussian_state_path)
	gaussians.restore(gaussian_state, opt)
	print(f"  Loaded initialGS from frame 0: {gaussian_state_path}")
	
	scene = Scene(model_params, gaussians, load_iteration=None, shuffle=False)
	
	# ============================================================
	# ============================================================
	print(f"\n[Step 2] Computing camera mapping...")
	train_cameras = scene.getTrainCameras()
	test_cameras = scene.getTestCameras()
	
	use_test_in_training = getattr(args, 'use_test_in_training', False)
	if use_test_in_training:
		train_cameras = combine_train_test_datasets(train_cameras, test_cameras)
	
	frame_to_train_cameras = {}
	frame_to_test_cameras = {}
	for t in range(process_frame_num):
		frame_to_train_cameras[t] = []
		frame_to_test_cameras[t] = []
	
	for cam_idx, cam in enumerate(train_cameras):
		if hasattr(cam, 'time') and cam.time is not None:
			if total_frame_num > 1:
				frame_idx = round(cam.time * (total_frame_num - 1))
			else:
				frame_idx = 0
			frame_idx = max(0, min(frame_idx, total_frame_num - 1))
			if frame_idx >= process_frame_num:
				continue
			frame_to_train_cameras[frame_idx].append((cam_idx, cam))
	
	for cam_idx, cam in enumerate(test_cameras):
		if hasattr(cam, 'time') and cam.time is not None:
			if total_frame_num > 1:
				frame_idx = round(cam.time * (total_frame_num - 1))
			else:
				frame_idx = 0
			frame_idx = max(0, min(frame_idx, total_frame_num - 1))
			if frame_idx >= process_frame_num:
				continue
			frame_to_test_cameras[frame_idx].append((cam_idx, cam))
	
	print(f"  Mapped cameras for {total_frame_num} frames")
	
	# ============================================================
	# ============================================================
	pp = PipelineParams(ArgumentParser())
	pipe = pp.extract(args)
	background = get_background_color(args, device)
	
	# ============================================================
	# ============================================================
	print(f"\n[Step 3] Processing {process_frame_num} frame(s)...")
	
	train_camera_psnrs = []
	train_camera_ssims = []
	train_camera_lpips_vgg = []
	train_camera_lpips_alex = []
	train_camera_frame_indices = []
	train_camera_renders = []
	train_camera_gts = []
	train_camera_info = []
	
	test_camera_psnrs = []
	test_camera_ssims = []
	test_camera_lpips_vgg = []
	test_camera_lpips_alex = []
	test_camera_frame_indices = []
	test_camera_renders = []
	test_camera_gts = []
	test_camera_info = []
	
	current_pos_sim = None
	xyz_world_0 = gaussians.get_xyz.detach().clone()
	xyz_smoke_0 = coord_trans.world2smoke(xyz_world_0)
	current_pos_sim = xyz_smoke_0 * lengths_tensor
	
	from velocity_training.train import _merge_inflow_to_gaussians
	
	for t in range(process_frame_num):
		print(f"\n  Processing frame {t}...")
		
		if t < total_frame_num - window_size:
			window_start = t
			window_end = t + window_size
		else:
			window_start = total_frame_num - window_size
			window_end = total_frame_num
		
		print(f"    Window: {window_start}->{window_end}")
		
		window_dir = os.path.join(ckpt_load_path, f"window_{window_start}_{window_end}")
		inflow_gaussians = None
		inflow_pattern = os.path.join(window_dir, "inflow_gaussians_epoch_*.pth")
		inflow_files = glob.glob(inflow_pattern)
		
		if inflow_files:
			latest_inflow_file = None
			max_epoch = -1
			for inflow_file in inflow_files:
				match = re.search(r'inflow_gaussians_epoch_(\d+)\.pth', os.path.basename(inflow_file))
				if match:
					epoch_num = int(match.group(1))
					if epoch_num > max_epoch:
						max_epoch = epoch_num
						latest_inflow_file = inflow_file
			
			if latest_inflow_file is None:
				latest_inflow_file = max(inflow_files, key=os.path.getctime)
			
			try:
				inflow_checkpoint_data = torch.load(latest_inflow_file, map_location=device)
				if isinstance(inflow_checkpoint_data, tuple) and len(inflow_checkpoint_data) >= 2:
					inflow_state = inflow_checkpoint_data[0]
					inflow_gaussians = InflowGaussians.restore(
						inflow_state,
						gaussians_template=gaussians,
						coord_trans=coord_trans,
						device=device
					)
					print(f"    Loaded inflow gaussians from: {latest_inflow_file}")
			except Exception as e:
				print(f"    Warning: Could not load inflow gaussians: {e}")
				inflow_gaussians = None
		
		if t == 0:
			if inflow_gaussians is not None:
				merge_group_indices = []
				for group_idx in range(inflow_gaussians.num_groups):
					origin_frame = inflow_gaussians.get_group_origin_frame(group_idx)
					if origin_frame is not None and origin_frame == 0:
						merge_group_indices.append(group_idx)
				
				if len(merge_group_indices) > 0:
					gaussians, _ = _merge_inflow_to_gaussians(
						gaussians, inflow_gaussians, merge_group_indices,
						coord_trans, lengths_tensor, device
					)
					print(f"    Merged {len(merge_group_indices)} inflow groups from frame 0")
			
			xyz_world_current = gaussians.get_xyz.detach().clone()
			xyz_smoke_current = coord_trans.world2smoke(xyz_world_current)
			current_pos_sim = xyz_smoke_current * lengths_tensor
			
		elif t < total_frame_num - window_size:
			prev_vel_model_path = os.path.join(ckpt_load_path, "frame_velocities", f"frame_{t-1:03d}_velocity.pth")
			if not os.path.exists(prev_vel_model_path):
				current_window_dir = os.path.join(ckpt_load_path, f"window_{window_start}_{window_end}")
				vel_pattern = os.path.join(current_window_dir, "ckpt", f"velrbf_frame_{t-1:03d}_ckpt_*.pth")
				vel_files = glob.glob(vel_pattern)
				if not vel_files:
					prev_window_dir = os.path.join(ckpt_load_path, f"window_{t-1}_{t-1+window_size}")
					vel_pattern = os.path.join(prev_window_dir, "ckpt", f"velrbf_frame_{t-1:03d}_ckpt_*.pth")
					vel_files = glob.glob(vel_pattern)
				if vel_files:
					prev_vel_model_path = max(vel_files, key=os.path.getctime)
			
			if prev_vel_model_path and os.path.exists(prev_vel_model_path):
				prev_velocity_model = TiDFRBF.load(prev_vel_model_path, device=device)
				print(f"    Loaded velocity model for frame {t-1}: {prev_vel_model_path}")
				
				with torch.no_grad():
					v_flat = prev_velocity_model(grid_points)
					v_vol = v_flat.view(nx, ny, nz, 3).permute(3, 2, 1, 0).unsqueeze(0)
					
					norm_pos = current_pos_sim.clone()
					norm_pos = 2 * (norm_pos / lengths_tensor) - 1.0
					grid_in = norm_pos.view(1, 1, 1, -1, 3)
					v_part = F.grid_sample(v_vol, grid_in, align_corners=True, mode='bilinear')
					v_part = v_part.view(3, -1).permute(1, 0)
					
					current_pos_sim = current_pos_sim + v_part * dt
					print(f"    Advected to frame {t} using velocity model for frame {t-1}")
			else:
				print(f"    Warning: Velocity model for frame {t-1} not found, using previous position")
			
			gaussian_state_path = os.path.join(ckpt_load_path, "frame_gaussians", f"frame_{t:03d}_gaussian.pth")
			if not os.path.exists(gaussian_state_path):
				raise FileNotFoundError(f"Frame {t} Gaussian state not found: {gaussian_state_path}")
			gaussian_state, _ = torch.load(gaussian_state_path)
			
			current_num_points = current_pos_sim.shape[0]
			
			if isinstance(gaussian_state, tuple):
				if len(gaussian_state) >= 2:
					target_num_points = gaussian_state[1].shape[0]
					xyz_from_state_world = gaussian_state[1]
					if isinstance(xyz_from_state_world, np.ndarray):
						xyz_from_state_world = torch.from_numpy(xyz_from_state_world).float().to(device)
				else:
					raise ValueError(f"Invalid gaussian_state tuple length: {len(gaussian_state)}")
			elif isinstance(gaussian_state, dict):
				if '_xyz' in gaussian_state:
					target_num_points = gaussian_state['_xyz'].shape[0]
					xyz_from_state_world = torch.from_numpy(gaussian_state['_xyz']).float().to(device) if isinstance(gaussian_state['_xyz'], np.ndarray) else gaussian_state['_xyz']
				elif 'xyz' in gaussian_state:
					target_num_points = gaussian_state['xyz'].shape[0]
					xyz_from_state_world = torch.from_numpy(gaussian_state['xyz']).float().to(device) if isinstance(gaussian_state['xyz'], np.ndarray) else gaussian_state['xyz']
				else:
					raise ValueError(f"Cannot determine number of points from gaussian_state dict")
			else:
				raise ValueError(f"Unexpected gaussian_state type: {type(gaussian_state)}")
			
			if current_num_points != target_num_points:
				raise ValueError(f"Point count mismatch: advected GS has {current_num_points} points, but loaded GS has {target_num_points} points")
			
			pos_smoke = current_pos_sim / lengths_tensor
			advected_xyz = coord_trans.smoke2world(pos_smoke)
			
			if advected_xyz.shape != xyz_from_state_world.shape:
				raise ValueError(f"XYZ shape mismatch: advected_xyz shape {advected_xyz.shape} != loaded_xyz shape {xyz_from_state_world.shape}")
			
			xyz_diff = torch.abs(advected_xyz - xyz_from_state_world)
			max_diff = torch.max(xyz_diff).item()
			mean_diff = torch.mean(xyz_diff).item()
			
			if not torch.allclose(advected_xyz, xyz_from_state_world, atol=1e-6, rtol=1e-6):
				print(f"    Error: XYZ values are not identical (max_diff: {max_diff:.6e}, mean_diff: {mean_diff:.6e})")
			
			gaussians.restore(gaussian_state, opt)
			gaussians._xyz.data = advected_xyz.requires_grad_(False)
			
			print(f"    Loaded GS attributes from frame {t} and applied advected positions")
			
			if inflow_gaussians is not None:
				merge_group_indices = []
				for group_idx in range(inflow_gaussians.num_groups):
					origin_frame = inflow_gaussians.get_group_origin_frame(group_idx)
					if origin_frame is not None and origin_frame == t:
						merge_group_indices.append(group_idx)
				
				if len(merge_group_indices) > 0:
					gaussians, _ = _merge_inflow_to_gaussians(
						gaussians, inflow_gaussians, merge_group_indices,
						coord_trans, lengths_tensor, device
					)
					print(f"    Merged {len(merge_group_indices)} inflow groups from frame {t}")
					
					xyz_world_current = gaussians.get_xyz.detach().clone()
					xyz_smoke_current = coord_trans.world2smoke(xyz_world_current)
					current_pos_sim = xyz_smoke_current * lengths_tensor
			
		else:
			prev_vel_model_path = os.path.join(ckpt_load_path, "frame_velocities", f"frame_{t-1:03d}_velocity.pth")
			if not os.path.exists(prev_vel_model_path):
				current_window_dir = os.path.join(ckpt_load_path, f"window_{window_start}_{window_end}")
				vel_pattern = os.path.join(current_window_dir, "ckpt", f"velrbf_frame_{t-1:03d}_ckpt_*.pth")
				vel_files = glob.glob(vel_pattern)
				if not vel_files:
					prev_window_dir = os.path.join(ckpt_load_path, f"window_{t-1}_{t-1+window_size}")
					vel_pattern = os.path.join(prev_window_dir, "ckpt", f"velrbf_frame_{t-1:03d}_ckpt_*.pth")
					vel_files = glob.glob(vel_pattern)
				if vel_files:
					prev_vel_model_path = max(vel_files, key=os.path.getctime)
			
			if prev_vel_model_path and os.path.exists(prev_vel_model_path):
				prev_velocity_model = TiDFRBF.load(prev_vel_model_path, device=device)
				print(f"    Loaded velocity model for frame {t-1}: {prev_vel_model_path}")
				
				with torch.no_grad():
					v_flat = prev_velocity_model(grid_points)
					v_vol = v_flat.view(nx, ny, nz, 3).permute(3, 2, 1, 0).unsqueeze(0)
					
					norm_pos = current_pos_sim.clone()
					norm_pos = 2 * (norm_pos / lengths_tensor) - 1.0
					grid_in = norm_pos.view(1, 1, 1, -1, 3)
					v_part = F.grid_sample(v_vol, grid_in, align_corners=True, mode='bilinear')
					v_part = v_part.view(3, -1).permute(1, 0)
					
					current_pos_sim = current_pos_sim + v_part * dt
					print(f"    Advected to frame {t} using velocity model for frame {t-1}")
			else:
				print(f"    Warning: Velocity model for frame {t-1} not found, using previous position")
			
			pos_smoke = current_pos_sim / lengths_tensor
			advected_xyz = coord_trans.smoke2world(pos_smoke)
			
			current_num_points = gaussians.get_xyz.shape[0]
			if current_num_points == advected_xyz.shape[0]:
				gaussians._xyz.data = advected_xyz.requires_grad_(False)
			else:
				if current_num_points > advected_xyz.shape[0]:
					gaussians._xyz.data[:advected_xyz.shape[0]] = advected_xyz.requires_grad_(False)
				else:
					gaussians._xyz.data = advected_xyz[:current_num_points].requires_grad_(False)
			
			print(f"    Updated position for frame {t} (attributes unchanged)")
			
			if inflow_gaussians is not None:
				merge_group_indices = []
				for group_idx in range(inflow_gaussians.num_groups):
					origin_frame = inflow_gaussians.get_group_origin_frame(group_idx)
					if origin_frame is not None and window_start <= origin_frame < t:
						merge_group_indices.append(group_idx)
				
				if len(merge_group_indices) > 0:
					gaussians, _ = _merge_inflow_to_gaussians(
						gaussians, inflow_gaussians, merge_group_indices,
						coord_trans, lengths_tensor, device
					)
					print(f"    Merged {len(merge_group_indices)} inflow groups")
					
					xyz_world_current = gaussians.get_xyz.detach().clone()
					xyz_smoke_current = coord_trans.world2smoke(xyz_world_current)
					current_pos_sim = xyz_smoke_current * lengths_tensor
		
		train_cameras_for_frame = frame_to_train_cameras.get(t, [])
		test_cameras_for_frame = frame_to_test_cameras.get(t, [])
		
		print(f"    Train cameras: {len(train_cameras_for_frame)}, Test cameras: {len(test_cameras_for_frame)}")
		
		pos_smoke = current_pos_sim / lengths_tensor
		pos_world = coord_trans.smoke2world(pos_smoke)
		
		if gaussians.get_xyz.shape[0] == pos_world.shape[0]:
			render_pos_world = pos_world
		else:
			render_pos_world = gaussians.get_xyz.detach().clone()
			print(f"    Warning: Point count mismatch, using gaussians.get_xyz ({gaussians.get_xyz.shape[0]}) instead of pos_world ({pos_world.shape[0]})")
		
		for cam_idx, cam in train_cameras_for_frame:
			with torch.no_grad():
				wrapped_gaussians = GaussianOverrideWrapper(gaussians, render_pos_world)
				
				render_result = render(cam, wrapped_gaussians, pipe, background, stage="coarse")["render"]
				gt_image = cam.original_image.cuda()
				
				render_tensor = render_result.unsqueeze(0)
				gt_tensor = gt_image.unsqueeze(0)
				
				psnr_val = psnr(render_tensor, gt_tensor).item()
				ssim_val = ssim(render_tensor, gt_tensor)
				if isinstance(ssim_val, torch.Tensor):
					ssim_val = ssim_val.item()
				
				torch.cuda.empty_cache()
				lpips_vgg_val = lpips_vgg_model(render_tensor, gt_tensor).squeeze().item()
				lpips_alex_val = lpips_alex_model(render_tensor, gt_tensor).squeeze().item()
				
				train_camera_psnrs.append(psnr_val)
				train_camera_ssims.append(ssim_val)
				train_camera_lpips_vgg.append(lpips_vgg_val)
				train_camera_lpips_alex.append(lpips_alex_val)
				train_camera_frame_indices.append(t)
				
				render_np = render_result.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
				gt_np = gt_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
				render_img = (render_np * 255).astype(np.uint8)
				gt_img = (gt_np * 255).astype(np.uint8)
				
				train_camera_renders.append(render_img)
				train_camera_gts.append(gt_img)
				train_camera_info.append((t, cam_idx))
		
		for cam_idx, cam in test_cameras_for_frame:
			with torch.no_grad():
				wrapped_gaussians = GaussianOverrideWrapper(gaussians, render_pos_world)
				
				render_result = render(cam, wrapped_gaussians, pipe, background, stage="coarse")["render"]
				gt_image = cam.original_image.cuda()
				
				render_tensor = render_result.unsqueeze(0)
				gt_tensor = gt_image.unsqueeze(0)
				
				psnr_val = psnr(render_tensor, gt_tensor).item()
				ssim_val = ssim(render_tensor, gt_tensor)
				if isinstance(ssim_val, torch.Tensor):
					ssim_val = ssim_val.item()
				
				torch.cuda.empty_cache()
				lpips_vgg_val = lpips_vgg_model(render_tensor, gt_tensor).squeeze().item()
				lpips_alex_val = lpips_alex_model(render_tensor, gt_tensor).squeeze().item()
				
				test_camera_psnrs.append(psnr_val)
				test_camera_ssims.append(ssim_val)
				test_camera_lpips_vgg.append(lpips_vgg_val)
				test_camera_lpips_alex.append(lpips_alex_val)
				test_camera_frame_indices.append(t)
				
				render_np = render_result.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
				gt_np = gt_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
				render_img = (render_np * 255).astype(np.uint8)
				gt_img = (gt_np * 255).astype(np.uint8)
				
				test_camera_renders.append(render_img)
				test_camera_gts.append(gt_img)
				test_camera_info.append((t, cam_idx))
		
	
	# ============================================================
	# ============================================================
	print(f"\n[Step 4] Computing metrics and saving results...")
	
	eval_savedir = os.path.join(savedir, "full_reconstruction_evaluation")
	os.makedirs(eval_savedir, exist_ok=True)
	os.makedirs(os.path.join(eval_savedir, "images", "train"), exist_ok=True)
	os.makedirs(os.path.join(eval_savedir, "images", "test"), exist_ok=True)
	
	mean_train_psnr = np.mean(train_camera_psnrs) if train_camera_psnrs else float('nan')
	mean_train_ssim = np.mean(train_camera_ssims) if train_camera_ssims else float('nan')
	mean_train_lpips_vgg = np.mean(train_camera_lpips_vgg) if train_camera_lpips_vgg else float('nan')
	mean_train_lpips_alex = np.mean(train_camera_lpips_alex) if train_camera_lpips_alex else float('nan')
	
	mean_test_psnr = np.mean(test_camera_psnrs) if test_camera_psnrs else float('nan')
	mean_test_ssim = np.mean(test_camera_ssims) if test_camera_ssims else float('nan')
	mean_test_lpips_vgg = np.mean(test_camera_lpips_vgg) if test_camera_lpips_vgg else float('nan')
	mean_test_lpips_alex = np.mean(test_camera_lpips_alex) if test_camera_lpips_alex else float('nan')
	
	print(f"\n{'='*80}")
	print(f"Full Reconstruction Evaluation Results")
	print(f"{'='*80}")
	if train_camera_psnrs:
		print(f"Train View - PSNR: {mean_train_psnr:.2f} dB, SSIM: {mean_train_ssim:.4f}, LPIPS-VGG: {mean_train_lpips_vgg:.4f}, LPIPS-Alex: {mean_train_lpips_alex:.4f}")
		print(f"  Total cameras: {len(train_camera_psnrs)}")
	if test_camera_psnrs:
		print(f"Test View - PSNR: {mean_test_psnr:.2f} dB, SSIM: {mean_test_ssim:.4f}, LPIPS-VGG: {mean_test_lpips_vgg:.4f}, LPIPS-Alex: {mean_test_lpips_alex:.4f}")
		print(f"  Total cameras: {len(test_camera_psnrs)}")
	
	metrics_path = os.path.join(eval_savedir, "metrics.txt")
	with open(metrics_path, 'w', encoding='utf-8') as f:
		f.write("=" * 80 + "\n")
		f.write(f"Full Reconstruction Evaluation Results\n")
		f.write("=" * 80 + "\n\n")
		
		if train_camera_psnrs:
			f.write("Train View Metrics:\n")
			f.write("-" * 80 + "\n")
			f.write(f"Mean PSNR: {mean_train_psnr:.4f} dB\n")
			f.write(f"Mean SSIM: {mean_train_ssim:.4f}\n")
			f.write(f"Mean LPIPS-VGG: {mean_train_lpips_vgg:.4f}\n")
			f.write(f"Mean LPIPS-Alex: {mean_train_lpips_alex:.4f}\n")
			f.write(f"Total cameras: {len(train_camera_psnrs)}\n")
			f.write("\nPer-camera train metrics:\n")
			for idx in range(len(train_camera_psnrs)):
				f.write(f"  Camera {train_camera_info[idx][1]}: Frame {train_camera_frame_indices[idx]}, PSNR={train_camera_psnrs[idx]:.4f} dB, SSIM={train_camera_ssims[idx]:.4f}, LPIPS-VGG={train_camera_lpips_vgg[idx]:.4f}, LPIPS-Alex={train_camera_lpips_alex[idx]:.4f}\n")
			f.write("\n")
		
		if test_camera_psnrs:
			f.write("Test View Metrics:\n")
			f.write("-" * 80 + "\n")
			f.write(f"Mean PSNR: {mean_test_psnr:.4f} dB\n")
			f.write(f"Mean SSIM: {mean_test_ssim:.4f}\n")
			f.write(f"Mean LPIPS-VGG: {mean_test_lpips_vgg:.4f}\n")
			f.write(f"Mean LPIPS-Alex: {mean_test_lpips_alex:.4f}\n")
			f.write(f"Total cameras: {len(test_camera_psnrs)}\n")
			f.write("\nPer-camera test metrics:\n")
			for idx in range(len(test_camera_psnrs)):
				f.write(f"  Camera {test_camera_info[idx][1]}: Frame {test_camera_frame_indices[idx]}, PSNR={test_camera_psnrs[idx]:.4f} dB, SSIM={test_camera_ssims[idx]:.4f}, LPIPS-VGG={test_camera_lpips_vgg[idx]:.4f}, LPIPS-Alex={test_camera_lpips_alex[idx]:.4f}\n")
	
	print(f"\nMetrics saved to: {metrics_path}")
	print(f"Results saved to: {eval_savedir}")
	
	# ============================================================
	# ============================================================
	print(f"\n[Step 5] Saving images and generating comparison videos...")
	
	train_images_dir = os.path.join(eval_savedir, "images", "train")
	os.makedirs(train_images_dir, exist_ok=True)
	for idx, (render_img, gt_img, (frame_idx, cam_idx)) in enumerate(zip(train_camera_renders, train_camera_gts, train_camera_info)):
		render_path = os.path.join(train_images_dir, f"cam_{cam_idx:03d}_frame_{frame_idx:03d}_render.png")
		gt_path = os.path.join(train_images_dir, f"cam_{cam_idx:03d}_frame_{frame_idx:03d}_gt.png")
		imageio.imwrite(render_path, render_img)
		imageio.imwrite(gt_path, gt_img)
	
	print(f"  Saved {len(train_camera_renders)} train view images")
	
	test_images_dir = os.path.join(eval_savedir, "images", "test")
	os.makedirs(test_images_dir, exist_ok=True)
	for idx, (render_img, gt_img, (frame_idx, cam_idx)) in enumerate(zip(test_camera_renders, test_camera_gts, test_camera_info)):
		render_path = os.path.join(test_images_dir, f"cam_{cam_idx:03d}_frame_{frame_idx:03d}_render.png")
		gt_path = os.path.join(test_images_dir, f"cam_{cam_idx:03d}_frame_{frame_idx:03d}_gt.png")
		imageio.imwrite(render_path, render_img)
		imageio.imwrite(gt_path, gt_img)
	
	print(f"  Saved {len(test_camera_renders)} test view images")
	
	if train_camera_renders:
		train_sorted_indices = sorted(range(len(train_camera_info)), key=lambda i: (train_camera_info[i][1], train_camera_info[i][0]))
		train_sorted_renders = [train_camera_renders[i] for i in train_sorted_indices]
		train_sorted_gts = [train_camera_gts[i] for i in train_sorted_indices]
		
		train_comparison_frames = []
		for render_img, gt_img in zip(train_sorted_renders, train_sorted_gts):
			h_render, w_render = render_img.shape[:2]
			h_gt, w_gt = gt_img.shape[:2]
			h = max(h_render, h_gt)
			
			if h_render != h:
				from PIL import Image
				render_img_pil = Image.fromarray(render_img)
				render_img_pil = render_img_pil.resize((w_render, h), Image.Resampling.LANCZOS)
				render_img = np.array(render_img_pil)
			if h_gt != h:
				from PIL import Image
				gt_img_pil = Image.fromarray(gt_img)
				gt_img_pil = gt_img_pil.resize((w_gt, h), Image.Resampling.LANCZOS)
				gt_img = np.array(gt_img_pil)
			
			comparison = np.concatenate([render_img, gt_img], axis=1)
			train_comparison_frames.append(comparison)
		
		train_video_path = os.path.join(eval_savedir, "comparison_train.mp4")
		imageio.mimwrite(train_video_path, train_comparison_frames, fps=10, quality=8)
		print(f"  Generated train view comparison video: {train_video_path}")
	
	if test_camera_renders:
		test_sorted_indices = sorted(range(len(test_camera_info)), key=lambda i: (test_camera_info[i][1], test_camera_info[i][0]))
		test_sorted_renders = [test_camera_renders[i] for i in test_sorted_indices]
		test_sorted_gts = [test_camera_gts[i] for i in test_sorted_indices]
		
		test_comparison_frames = []
		for render_img, gt_img in zip(test_sorted_renders, test_sorted_gts):
			h_render, w_render = render_img.shape[:2]
			h_gt, w_gt = gt_img.shape[:2]
			h = max(h_render, h_gt)
			
			if h_render != h:
				from PIL import Image
				render_img_pil = Image.fromarray(render_img)
				render_img_pil = render_img_pil.resize((w_render, h), Image.Resampling.LANCZOS)
				render_img = np.array(render_img_pil)
			if h_gt != h:
				from PIL import Image
				gt_img_pil = Image.fromarray(gt_img)
				gt_img_pil = gt_img_pil.resize((w_gt, h), Image.Resampling.LANCZOS)
				gt_img = np.array(gt_img_pil)
			
			comparison = np.concatenate([render_img, gt_img], axis=1)
			test_comparison_frames.append(comparison)
		
		test_video_path = os.path.join(eval_savedir, "comparison_test.mp4")
		imageio.mimwrite(test_video_path, test_comparison_frames, fps=10, quality=8)
		print(f"  Generated test view comparison video: {test_video_path}")
	
	return {
		'train': {
			'psnr': mean_train_psnr,
			'ssim': mean_train_ssim,
			'lpips_vgg': mean_train_lpips_vgg,
			'lpips_alex': mean_train_lpips_alex,
			'num_cameras': len(train_camera_psnrs),
			'per_camera': {
				'psnr': train_camera_psnrs,
				'ssim': train_camera_ssims,
				'lpips_vgg': train_camera_lpips_vgg,
				'lpips_alex': train_camera_lpips_alex,
				'frame_indices': train_camera_frame_indices
			}
		},
		'test': {
			'psnr': mean_test_psnr,
			'ssim': mean_test_ssim,
			'lpips_vgg': mean_test_lpips_vgg,
			'lpips_alex': mean_test_lpips_alex,
			'num_cameras': len(test_camera_psnrs),
			'per_camera': {
				'psnr': test_camera_psnrs,
				'ssim': test_camera_ssims,
				'lpips_vgg': test_camera_lpips_vgg,
				'lpips_alex': test_camera_lpips_alex,
				'frame_indices': test_camera_frame_indices
			}
		}
	}
