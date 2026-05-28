import torch
import numpy as np
import os
import imageio
from tqdm import tqdm
from velocity_common.ray_utils import get_rays_from_camera, ray_inflow_region_intersection, visualize_rays_and_inflow_region
from velocity_training.wrappers import InflowOnlyGaussianWrapper
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render


def camera_poses_similar(cam1, cam2, rotation_threshold=1e-4, translation_threshold=1e-4):
	"""
	Compare whether two camera poses are similar.
	
	Args:
		cam1: first camera object
		cam2: second camera object
		rotation_threshold: rotation
		translation_threshold: vector
	
	Returns:
		bool: returns True when poses are similar, otherwise False
	"""
	if hasattr(cam1, 'R') and hasattr(cam1, 'T') and hasattr(cam2, 'R') and hasattr(cam2, 'T'):
		R1 = cam1.R
		T1 = cam1.T
		R2 = cam2.R
		T2 = cam2.T
	elif hasattr(cam1, 'world_view_transform') and hasattr(cam2, 'world_view_transform'):
		w2v1 = cam1.world_view_transform
		w2v2 = cam2.world_view_transform
		v2w1 = torch.inverse(w2v1) if w2v1.shape == (4, 4) else w2v1.inverse()
		v2w2 = torch.inverse(w2v2) if w2v2.shape == (4, 4) else w2v2.inverse()
		R1 = v2w1[:3, :3]
		T1 = v2w1[:3, 3]
		R2 = v2w2[:3, :3]
		T2 = v2w2[:3, 3]
	else:
		return False
	
	if not isinstance(R1, torch.Tensor):
		R1 = torch.tensor(R1, dtype=torch.float32)
	if not isinstance(T1, torch.Tensor):
		T1 = torch.tensor(T1, dtype=torch.float32)
	if not isinstance(R2, torch.Tensor):
		R2 = torch.tensor(R2, dtype=torch.float32)
	if not isinstance(T2, torch.Tensor):
		T2 = torch.tensor(T2, dtype=torch.float32)
	
	device = R1.device if isinstance(R1, torch.Tensor) else 'cpu'
	if isinstance(R2, torch.Tensor) and R2.device != device:
		R2 = R2.to(device)
	if isinstance(T1, torch.Tensor) and T1.device != device:
		T1 = T1.to(device)
	if isinstance(T2, torch.Tensor) and T2.device != device:
		T2 = T2.to(device)
	
	R_diff = torch.norm(R1 - R2, p='fro').item() if isinstance(R1, torch.Tensor) else np.linalg.norm(R1 - R2, 'fro')
	
	T_diff = torch.norm(T1 - T2, p=2).item() if isinstance(T1, torch.Tensor) else np.linalg.norm(T1 - T2, 2)
	
	return R_diff < rotation_threshold and T_diff < translation_threshold


def visualize_inflow_region_all_cameras(gaussians, scene, coord_trans, lengths_tensor, savedir, render_func, pipe, background, inflow_region_min=None, inflow_region_max=None):
	"""
	visualization inflow region camera
	use ray casting ： GS，each ray  inflow region ，
	
	Args:
		gaussians: GaussianModel object
		scene: Scene object
		coord_trans: CoordinateTransform object
		lengths_tensor: [3] tensor，Sim Space 
		savedir: output directory
		render_func: （ gaussian_renderer  render）
		pipe: PipelineParams object
		background: background color tensor
		inflow_region_min: bbox  [x_min, y_min, z_min]（ lengths_tensor ）
		inflow_region_max: bbox latest [x_max, y_max, z_max]（ lengths_tensor ）
	"""
	print("\n=== visualization Inflow Region camera（Ray Casting ）===")
	
	if inflow_region_min is None:
		inflow_region_min = [0.0, 0.1, 0.0]
	if inflow_region_max is None:
		inflow_region_max = [1.0, 0.3, 1.0]
	
	device = lengths_tensor.device
	if isinstance(inflow_region_min, (list, tuple)):
		inflow_region_min = torch.tensor(inflow_region_min, device=device, dtype=torch.float32)
	if isinstance(inflow_region_max, (list, tuple)):
		inflow_region_max = torch.tensor(inflow_region_max, device=device, dtype=torch.float32)
	
	output_dir = os.path.join(savedir, "vis_inflow_region")
	os.makedirs(output_dir, exist_ok=True)
	
	train_cameras = scene.getTrainCameras()
	print(f"Found {len(train_cameras)} training cameras")
	
	orange_color = torch.tensor([1.0, 0.647, 0.0], device=device)
	alpha = 0.5
	
	seen_poses = []
	rendered_images = []
	unique_pose_count = 0
	
	for cam_idx, cam in enumerate(tqdm(train_cameras, desc="checkcamerapose")):
		is_new_pose = True
		for seen_cam in seen_poses:
			if camera_poses_similar(cam, seen_cam):
				is_new_pose = False
				break
		
		if is_new_pose:
			seen_poses.append(cam)
			unique_pose_count += 1
			
			try:
				with torch.no_grad():
					render_result = render_func(cam, gaussians, pipe, background, stage="coarse")
					render_image = render_result['render']  # [3, H, W]
				
				H, W = render_image.shape[1], render_image.shape[2]
				render_image = render_image.permute(1, 2, 0).clamp(0, 1)  # [H, W, 3]
				
				rays_o, rays_d = get_rays_from_camera(cam)
				
				if unique_pose_count == 1:
					debug_viz_path = os.path.join(output_dir, f"debug_rays_inflow_region_pose_{unique_pose_count:03d}.png")
					visualize_rays_and_inflow_region(
						rays_o, rays_d, coord_trans, lengths_tensor,
						inflow_region_min, inflow_region_max,
						debug_viz_path, num_rays_to_plot=200
					)
				
				intersection_mask = ray_inflow_region_intersection(
					rays_o, rays_d, coord_trans, lengths_tensor, 
					device=device,
					inflow_region_min=inflow_region_min,
					inflow_region_max=inflow_region_max
				)
				
				intersection_mask_3d = intersection_mask.unsqueeze(-1).float()  # [H, W, 1]
				orange_overlay = orange_color.unsqueeze(0).unsqueeze(0).expand(H, W, -1)  # [H, W, 3]
				
				blended_image = render_image * (1.0 - alpha * intersection_mask_3d) + orange_overlay * (alpha * intersection_mask_3d)
				
				blended_np = (blended_image.detach().cpu().numpy() * 255).astype(np.uint8)
				image_path = os.path.join(output_dir, f"pose_{unique_pose_count:03d}_cam_{cam_idx:03d}.png")
				imageio.imwrite(image_path, blended_np)
				
				rendered_images.append(blended_np)
				print(f"  pose {unique_pose_count} (camera {cam_idx})")
				
			except Exception as e:
				print(f"  Warning: camera {cam_idx} render failed: {e}")
				import traceback
				traceback.print_exc()
				continue
		else:
			print(f"  Skippingcamera {cam_idx}（posecamera）")
	
	if len(rendered_images) > 1:
		video_path = os.path.join(output_dir, "inflow_region_visualization.mp4")
		try:
			imageio.mimwrite(video_path, rendered_images, fps=10)
			print(f"Video saved: {video_path}")
		except Exception as e:
			print(f"Failed to generate video: {e}")
	
	print(f"Visualization complete！images saved in: {output_dir}")
	print(f" {len(train_cameras)} camera，where {unique_pose_count} differentpose， {len(rendered_images)} ")
	print(f"use ray casting  inflow region ")


def precompute_camera_masks(scene, coord_trans, lengths_tensor, device, inflow_region_min=None, inflow_region_max=None):
	"""
	precomputeeachuniquecameraposition inflow region mask
	identify fix camera（positiondifferent），only foreachuniquepositioncompute once mask
	
	Args:
		scene: Scene object
		coord_trans: CoordinateTransform object
		lengths_tensor: [3] tensor，Sim Space 
		device: device
		inflow_region_min: bbox  [x_min, y_min, z_min]（ lengths_tensor ）
		inflow_region_max: bbox latest [x_max, y_max, z_max]（ lengths_tensor ）
	
	Returns:
		camera_position_to_mask: dict，{camera_key: mask_tensor[H, W]}
		camera_to_position_key: dict，{cam_idx: camera_key}
	"""
	print("\n=== precomputecamera mask（identify fix camera）===")
	
	if inflow_region_min is None:
		inflow_region_min = [0.0, 0.1, 0.0]
	if inflow_region_max is None:
		inflow_region_max = [1.0, 0.3, 1.0]
	
	if isinstance(inflow_region_min, (list, tuple)):
		inflow_region_min = torch.tensor(inflow_region_min, device=device, dtype=torch.float32)
	if isinstance(inflow_region_max, (list, tuple)):
		inflow_region_max = torch.tensor(inflow_region_max, device=device, dtype=torch.float32)
	
	train_cameras = scene.getTrainCameras()
	print(f"Found {len(train_cameras)} training cameras")
	
	camera_position_to_mask = {}  # {camera_key: mask_tensor}
	camera_to_position_key = {}   # {cam_idx: camera_key}
	position_groups = []
	
	for cam_idx, cam in enumerate(train_cameras):
		R = cam.R if hasattr(cam, 'R') else None
		T = cam.T if hasattr(cam, 'T') else None
		
		if R is not None:
			if isinstance(R, np.ndarray):
				R = torch.from_numpy(R).float().to(device)
			elif not isinstance(R, torch.Tensor):
				R = torch.tensor(R, dtype=torch.float32, device=device)
		if T is not None:
			if isinstance(T, np.ndarray):
				T = torch.from_numpy(T).float().to(device)
			elif not isinstance(T, torch.Tensor):
				T = torch.tensor(T, dtype=torch.float32, device=device)
		
		found_group = False
		for group_idx, group_cams in enumerate(position_groups):
			ref_cam = train_cameras[group_cams[0]]
			ref_R = ref_cam.R if hasattr(ref_cam, 'R') else None
			ref_T = ref_cam.T if hasattr(ref_cam, 'T') else None
			
			if ref_R is not None:
				if isinstance(ref_R, np.ndarray):
					ref_R = torch.from_numpy(ref_R).float().to(device)
				elif not isinstance(ref_R, torch.Tensor):
					ref_R = torch.tensor(ref_R, dtype=torch.float32, device=device)
			if ref_T is not None:
				if isinstance(ref_T, np.ndarray):
					ref_T = torch.from_numpy(ref_T).float().to(device)
				elif not isinstance(ref_T, torch.Tensor):
					ref_T = torch.tensor(ref_T, dtype=torch.float32, device=device)
			
			if R is not None and ref_R is not None:
				if not torch.allclose(R, ref_R, atol=1e-5):
					continue
			elif R != ref_R:
				continue
			
			if T is not None and ref_T is not None:
				if not torch.allclose(T, ref_T, atol=1e-5):
					continue
			elif T != ref_T:
				continue
			
			position_groups[group_idx].append(cam_idx)
			camera_to_position_key[cam_idx] = group_idx
			found_group = True
			break
		
		if not found_group:
			position_groups.append([cam_idx])
			camera_to_position_key[cam_idx] = len(position_groups) - 1
	
	print(f"identify {len(position_groups)} uniquecameraposition")
	for group_idx, group_cams in enumerate(position_groups):
		print(f"  position {group_idx}: {len(group_cams)} camera（fix camera）")
	
	print("\neachuniqueposition inflow region mask...")
	for group_idx, group_cams in enumerate(tqdm(position_groups, desc=" mask")):
		ref_cam_idx = group_cams[0]
		ref_cam = train_cameras[ref_cam_idx]
		
		rays_o, rays_d = get_rays_from_camera(ref_cam)
		
		mask = ray_inflow_region_intersection(
			rays_o, rays_d, coord_trans, lengths_tensor,
			device=device,
			inflow_region_min=inflow_region_min,
			inflow_region_max=inflow_region_max
		)
		
		camera_position_to_mask[group_idx] = mask
	
	print(f"complete！ {len(camera_position_to_mask)}  mask")
	
	return camera_position_to_mask, camera_to_position_key


def pretrain_inflow_gaussians(gaussians, inflow_gaussians, scene, coord_trans, lengths_tensor,
                              camera_position_to_mask, camera_to_position_key, frame_to_cameras,
                              pipe, background, savedir, args=None,
                              inflow_region_min=None, inflow_region_max=None,
                              pretrain_iterations=3000):
	"""
	each inflow GS 
	
	Args:
		gaussians: original Gaussian （used forgetoptimizerset，）
		inflow_gaussians: InflowGaussians object
		scene: Scene object
		coord_trans: CoordinateTransform object
		lengths_tensor: Sim Space 
		camera_position_to_mask: precompute mask 
		camera_to_position_key: cameraposition key 
		frame_to_cameras: framecamera
		pipe: PipelineParams
		background: background color
		savedir: output directory
		inflow_region_min: bbox 
		inflow_region_max: bbox latest
		pretrain_iterations: （ 500）
	"""
	import torch.optim as optim
	
	print("\n===  Inflow Gaussians ===")
	
	default_lr = {
		'xyz': 0.00016,
		'f_dc': 0.0025,
		'f_rest': 0.000125,
		'opacity': 0.05,
		'scaling': 0.005,
		'rotation': 0.001
	}
	
	if hasattr(gaussians, 'xyz_scheduler_args'):
		lr_xyz = gaussians.xyz_scheduler_args(0)
		if lr_xyz == 0.0:
			print(f"  Warning:  scheduler get xyz  0，check spatial_lr_scale...")
			spatial_lr_scale = getattr(gaussians, 'spatial_lr_scale', 1.0)
			print(f"  spatial_lr_scale = {spatial_lr_scale}")
			if args is not None and hasattr(args, 'position_lr_init'):
				effective_scale = spatial_lr_scale if spatial_lr_scale > 1e-6 else 1.0
				lr_xyz = args.position_lr_init * effective_scale
				print(f"   args get: position_lr_init={args.position_lr_init}, effective_scale={effective_scale},  lr_xyz={lr_xyz}")
			else:
				lr_xyz = default_lr['xyz']
				print(f"  use: lr_xyz={lr_xyz}")
	elif args is not None and hasattr(args, 'position_lr_init'):
		spatial_lr_scale = getattr(gaussians, 'spatial_lr_scale', 1.0)
		effective_scale = spatial_lr_scale if spatial_lr_scale > 1e-6 else 1.0
		lr_xyz = args.position_lr_init * effective_scale
	else:
		lr_xyz = default_lr['xyz']
	
	if args is not None:
		lr_features_dc = getattr(args, 'feature_lr', default_lr['f_dc'])
		lr_features_rest = getattr(args, 'feature_lr', default_lr['f_dc']) / 20.0
		lr_opacity = getattr(args, 'opacity_lr', default_lr['opacity'])
		lr_scaling = getattr(args, 'scaling_lr', default_lr['scaling'])
		lr_rotation = getattr(args, 'rotation_lr', default_lr['rotation'])
	else:
		lr_features_dc = default_lr['f_dc']
		lr_features_rest = default_lr['f_rest']
		lr_opacity = default_lr['opacity']
		lr_scaling = default_lr['scaling']
		lr_rotation = default_lr['rotation']
	
	print(f"use:")
	print(f"  xyz: {lr_xyz:.6f}")
	print(f"  f_dc: {lr_features_dc:.6f}")
	print(f"  f_rest: {lr_features_rest:.6f}")
	print(f"  opacity: {lr_opacity:.6f}")
	print(f"  scaling: {lr_scaling:.6f}")
	print(f"  rotation: {lr_rotation:.6f}")
	
	if args is not None:
		lambda_dssim = getattr(args, 'lambda_dssim', 0.2)
	else:
		lambda_dssim = 0.2
	
	pretrain_dir = os.path.join(savedir, "vis_inflow_pretrain")
	os.makedirs(pretrain_dir, exist_ok=True)
	
	frame_num = len(frame_to_cameras)
	
	for t in range(1, frame_num):
		group_idx = t - 1
		print(f"\n---  {t}  Inflow GS (group {group_idx}) ---")
		
		if t not in frame_to_cameras or len(frame_to_cameras[t]) == 0:
			print(f"  Warning:  {t} camera，Skipping")
			continue
		
		cameras_t = frame_to_cameras[t]
		print(f"  Found {len(cameras_t)} camera")
		
		optimizer = optim.Adam([
			{'params': [inflow_gaussians._xyz_groups[group_idx]], 'lr': lr_xyz, 'name': 'xyz'},
			{'params': [inflow_gaussians._features_dc_groups[group_idx]], 'lr': lr_features_dc, 'name': 'f_dc'},
			{'params': [inflow_gaussians._features_rest_groups[group_idx]], 'lr': lr_features_rest, 'name': 'f_rest'},
			{'params': [inflow_gaussians._opacity_groups[group_idx]], 'lr': lr_opacity, 'name': 'opacity'},
			{'params': [inflow_gaussians._scaling_groups[group_idx]], 'lr': lr_scaling, 'name': 'scaling'},
			{'params': [inflow_gaussians._rotation_groups[group_idx]], 'lr': lr_rotation, 'name': 'rotation'},
		])
		
		inflow_wrapper = InflowOnlyGaussianWrapper(inflow_gaussians, group_idx)
		
		losses = []
		losses_mask = []
		losses_bg = []
		losses_dssim = []
		for iteration in tqdm(range(pretrain_iterations), desc=f" {t}"):
			optimizer.zero_grad()
			
			cam_idx, cam = cameras_t[np.random.randint(0, len(cameras_t))]
			
			position_key = camera_to_position_key.get(cam_idx, None)
			if position_key is None:
				print(f"  Warning: camera {cam_idx}  position key，Skipping")
				continue
			
			mask = camera_position_to_mask[position_key]  # [H, W]
			
			render_pkg = render(cam, inflow_wrapper, pipe, background, stage="coarse")
			render_image = render_pkg['render']  # [3, H, W]
			
			gt_image = cam.original_image[:3, :, :].cuda()  # [3, H, W]
			
			mask_3d = mask.unsqueeze(0).float()  # [1, H, W]
			background_3d = background.unsqueeze(-1).unsqueeze(-1)  # [3, 1, 1] -> [3, H, W] after expand
			
			diff_mask = torch.abs(render_image - gt_image)  # [3, H, W]
			l1_loss_mask = (diff_mask * mask_3d).sum()
			
			diff_bg = torch.abs(render_image - background_3d.expand_as(render_image))  # [3, H, W]
			l1_loss_bg = (diff_bg * (1 - mask_3d)).sum()
			
			# render_masked = render_image * mask_3d + background_3d.expand_as(render_image) * (1 - mask_3d)
			# gt_masked = gt_image * mask_3d + background_3d.expand_as(gt_image) * (1 - mask_3d)
			
			# render_batch = render_masked.unsqueeze(0)  # [1, 3, H, W]
			# gt_batch = gt_masked.unsqueeze(0)  # [1, 3, H, W]
			# ssim_value = ssim(render_batch, gt_batch)
			# dssim_loss = 1.0 - ssim_value
			dssim_loss = torch.tensor(0.0)
			
			loss = l1_loss_mask + l1_loss_bg + lambda_dssim * dssim_loss
			
			loss.backward()
			
			optimizer.step()
			
			
			losses.append(loss.item())
			losses_mask.append(l1_loss_mask.item())
			losses_bg.append(l1_loss_bg.item())
			losses_dssim.append(dssim_loss.item())
			
			if (iteration + 1) % 100 == 0 or iteration == 0:
				with torch.no_grad():
					vis_images = []
					for vis_cam_idx, vis_cam in cameras_t[:5]:
						vis_render_pkg = render(vis_cam, inflow_wrapper, pipe, background, stage="coarse")
						vis_render_img = vis_render_pkg['render'].permute(1, 2, 0).clamp(0, 1)  # [H, W, 3]
						
						vis_position_key = camera_to_position_key.get(vis_cam_idx, None)
						if vis_position_key is not None:
							vis_mask = camera_position_to_mask[vis_position_key]  # [H, W]
							vis_mask_3d = vis_mask.unsqueeze(-1).float()  # [H, W, 1]
							
							orange_color = torch.tensor([1.0, 0.647, 0.0], device=vis_render_img.device)
							orange_overlay = orange_color.unsqueeze(0).unsqueeze(0).expand_as(vis_render_img)
							vis_render_img = vis_render_img * (1.0 - 0.3 * vis_mask_3d) + orange_overlay * (0.3 * vis_mask_3d)
						
						vis_images.append(vis_render_img.detach().cpu().numpy())
					
					vis_path = os.path.join(pretrain_dir, f"frame_{t:03d}_iter_{iteration+1:04d}.png")
					if vis_images:
						vis_grid = np.concatenate(vis_images, axis=1)
						imageio.imwrite(vis_path, (vis_grid * 255).astype(np.uint8))
		
		avg_loss = np.mean(losses)
		avg_loss_mask = np.mean(losses_mask)
		avg_loss_bg = np.mean(losses_bg)
		avg_loss_dssim = np.mean(losses_dssim)
		print(f"   {t} complete，: {avg_loss:.6f}")
		print(f"    - Mask  L1: {avg_loss_mask:.6f}")
		print(f"    - Background  L1: {avg_loss_bg:.6f}")
		print(f"    - DSSIM: {avg_loss_dssim:.6f}")
		
		loss_curve_path = os.path.join(pretrain_dir, f"frame_{t:03d}_loss_curve.png")
		visualize_loss_curve(losses, losses_mask, losses_bg, losses_dssim, loss_curve_path, t)
		
		print(f"  visualization {t} ...")
		visualize_single_frame_pretrain_result(
			inflow_gaussians, group_idx, cameras_t,
			camera_position_to_mask, camera_to_position_key,
			pipe, background, pretrain_dir, t
		)
	
	print(f"\ncomplete！Results saved to: {pretrain_dir}")


def visualize_loss_curve(losses, losses_mask, losses_bg, losses_dssim, save_path, frame_idx):
	"""
	visualization loss 
	
	Args:
		losses: list
		losses_mask: mask  L1 list
		losses_bg: background  L1 list
		losses_dssim: DSSIM list
		save_path: save path
		frame_idx: frame index
	"""
	import matplotlib.pyplot as plt
	
	iterations = np.arange(1, len(losses) + 1)
	
	fig, axes = plt.subplots(2, 2, figsize=(15, 10))
	fig.suptitle(f'Loss Curves - Frame {frame_idx}', fontsize=16, fontweight='bold')
	
	ax1 = axes[0, 0]
	ax1.plot(iterations, losses, 'b-', linewidth=1.5, label='Total Loss')
	ax1.set_xlabel('Iteration')
	ax1.set_ylabel('Loss')
	ax1.set_title('Total Loss')
	ax1.grid(True, alpha=0.3)
	ax1.legend()
	
	ax2 = axes[0, 1]
	ax2.plot(iterations, losses_mask, 'r-', linewidth=1.5, label='Mask Region L1')
	ax2.set_xlabel('Iteration')
	ax2.set_ylabel('Loss')
	ax2.set_title('Mask Region L1 Loss')
	ax2.grid(True, alpha=0.3)
	ax2.legend()
	
	ax3 = axes[1, 0]
	ax3.plot(iterations, losses_bg, 'g-', linewidth=1.5, label='Background Region L1')
	ax3.set_xlabel('Iteration')
	ax3.set_ylabel('Loss')
	ax3.set_title('Background Region L1 Loss')
	ax3.grid(True, alpha=0.3)
	ax3.legend()
	
	ax4 = axes[1, 1]
	ax4.plot(iterations, losses_dssim, 'm-', linewidth=1.5, label='DSSIM Loss')
	ax4.set_xlabel('Iteration')
	ax4.set_ylabel('Loss')
	ax4.set_title('DSSIM Loss')
	ax4.grid(True, alpha=0.3)
	ax4.legend()
	
	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f"    : {save_path}")


def visualize_single_frame_pretrain_result(inflow_gaussians, group_idx, cameras_t,
                                          camera_position_to_mask, camera_to_position_key,
                                          pipe, background, savedir, frame_idx):
	"""
	visualization
	
	Args:
		inflow_gaussians: InflowGaussians object
		group_idx: inflow GS 
		cameras_t: cameralist
		camera_position_to_mask: precompute mask 
		camera_to_position_key: cameraposition key 
		pipe: PipelineParams
		background: background color
		savedir: output directory
		frame_idx: frame index
	"""
	inflow_wrapper = InflowOnlyGaussianWrapper(inflow_gaussians, group_idx)
	
	vis_images = []
	for cam_idx, cam in cameras_t:
		render_pkg = render(cam, inflow_wrapper, pipe, background, stage="coarse")
		render_image = render_pkg['render']  # [3, H, W]
		render_image = render_image.permute(1, 2, 0).clamp(0, 1)  # [H, W, 3]
		
		gt_image = cam.original_image[:3, :, :].cuda()  # [3, H, W]
		gt_image = gt_image.permute(1, 2, 0).clamp(0, 1)  # [H, W, 3]
		
		position_key = camera_to_position_key.get(cam_idx, None)
		if position_key is not None:
			mask = camera_position_to_mask[position_key]  # [H, W]
			mask_3d = mask.unsqueeze(-1).float()  # [H, W, 1]
			
			error_map = torch.abs(render_image - gt_image)  # [H, W, 3]
			
			orange_color = torch.tensor([1.0, 0.647, 0.0], device=render_image.device)
			orange_overlay = orange_color.unsqueeze(0).unsqueeze(0).expand_as(render_image)
			render_with_mask = render_image * (1.0 - 0.3 * mask_3d) + orange_overlay * (0.3 * mask_3d)
			
			vis_row = torch.cat([gt_image, render_image, render_with_mask, error_map], dim=1)  # [H, 4*W, 3]
			vis_images.append(vis_row.detach().cpu().numpy())
		else:
			vis_row = torch.cat([gt_image, render_image], dim=1)  # [H, 2*W, 3]
			vis_images.append(vis_row.detach().cpu().numpy())
	
	if vis_images:
		vis_grid = np.concatenate(vis_images, axis=0)
		vis_path = os.path.join(savedir, f"frame_{frame_idx:03d}_final_comparison.png")
		imageio.imwrite(vis_path, (vis_grid * 255).astype(np.uint8))
		print(f"    saved to: {vis_path}")
