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
from lpipsPyTorch.modules.lpips import LPIPS
from gaussian_renderer import render
from gaussian_renderer.training import combine_train_test_datasets
from argparse import ArgumentParser
from arguments import PipelineParams, ModelParams, ModelHiddenParams, OptimizationParams
import colorsys
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from velocity_common.coordinate_transform import CoordinateTransform
from velocity_common.kernels import generate_kernels
from velocity_common.utils import get_background_color, set_device
from velocity_common.ray_utils import get_rays_from_camera
from velocity_training.models import InflowGaussians
from velocity_training.wrappers import ExtendedGaussianWrapper, GaussianOverrideWrapper
from velocity_evaluation.visualize import vel2hsv
from utils.sh_utils import RGB2SH
from scene.gaussian_model import GaussianModel
from scene import Scene


class GaussianColorOverrideWrapper:
	"""
	，used for GaussianModel  _features_dc attributes（），
	attributesoriginal。
	"""
	def __init__(self, original_model, override_features_dc):
		"""
		Args:
			original_model: original GaussianModel object
			override_features_dc:  features_dc，shape (N, 1, 3)
		"""
		self.orig = original_model
		self.override_features_dc = override_features_dc
	
	@property
	def get_xyz(self):
		return self.orig.get_xyz
	
	@property
	def get_opacity(self):
		return self.orig.get_opacity
		
	@property
	def get_features(self):
		features_rest = self.orig.get_features_rest
		return torch.cat([self.override_features_dc, features_rest], dim=1)
	
	@property
	def get_features_dc(self):
		return self.override_features_dc
	
	@property
	def get_features_rest(self):
		return self.orig.get_features_rest
	
	@property
	def get_scaling(self):
		return self.orig.get_scaling
	
	@property
	def get_rotation(self):
		return self.orig.get_rotation
	
	@property
	def active_sh_degree(self):
		return self.orig.active_sh_degree
	
	@property
	def _features_dc(self):
		return self.override_features_dc
	
	@property
	def _features_rest(self):
		return self.orig._features_rest
	
	@property
	def _opacity(self):
		return self.orig._opacity
	
	@property
	def _scaling(self):
		return self.orig._scaling
	
	@property
	def _rotation(self):
		return self.orig._rotation
	
	def __getattr__(self, name):
		return getattr(self.orig, name)


def velocity_to_rgb(velocity, scale=300, is3D=True, device='cuda', colormap='hsv'):
	"""
	velocity vectorconvert to RGB （use，convert to SH）
	
	Args:
		velocity: velocity vector torch.Tensor，shape (N, 3)
		scale: scaling，used for HSV （ colormap='hsv' use）
		is3D:  3D velocity field（ colormap='hsv' use）
		device: device
		colormap: 
			- 'hsv': use HSV （，）
			- 'y_projection':  Y （->，->）
			- 'magnitude_inferno': （magnitude）use matplotlib inferno colormap
			- 'magnitude_afmhot': （magnitude）use matplotlib afmhot colormap
	
	Returns:
		rgb: RGB  torch.Tensor，shape (N, 3)， [0, 1]
	"""
	if colormap == 'hsv':
		vel_np = velocity.detach().cpu().numpy()
		
		hsv_uint8 = vel2hsv(vel_np, is3D=is3D, logv=False, scale=scale)
		
		hsv_float = hsv_uint8.astype(np.float32) / 255.0
		
		N = hsv_float.shape[0]
		rgb_np = np.zeros((N, 3), dtype=np.float32)
		
		for i in range(N):
			h, s, v = hsv_float[i, 0], hsv_float[i, 1], hsv_float[i, 2]
			rgb_np[i] = colorsys.hsv_to_rgb(h, s, v)
		
		rgb_tensor = torch.from_numpy(rgb_np).float().to(device)
		rgb_tensor = torch.clamp(rgb_tensor, 0.0, 1.0)
		
		return rgb_tensor
		
	elif colormap == 'y_projection':
		
		vy = velocity[:, 1].detach()  # (N,)
		
		vy_abs_max = torch.abs(vy).max()
		if vy_abs_max > 1e-6:
			vy_normalized = vy / vy_abs_max
		else:
			vy_normalized = torch.zeros_like(vy)
		
		t = (vy_normalized + 1.0) / 2.0
		
		cyan = torch.tensor([0.0, 0.8, 0.8], device=device)
		orange = torch.tensor([1.0, 0.65, 0.0], device=device)
		
		# t=0 -> cyan, t=1 -> orange
		t_expanded = t.unsqueeze(1)  # (N, 1)
		rgb_tensor = cyan.unsqueeze(0) * (1.0 - t_expanded) + orange.unsqueeze(0) * t_expanded
		
		rgb_tensor = torch.clamp(rgb_tensor, 0.0, 1.0)
		
		return rgb_tensor
		
	elif colormap in ['magnitude_inferno', 'magnitude_afmhot']:
		vel_magnitude = torch.norm(velocity, dim=1)  # (N,)
		
		vel_max = vel_magnitude.max()
		vel_min = vel_magnitude.min()
		if vel_max > vel_min:
			vel_normalized = (vel_magnitude - vel_min) / (vel_max - vel_min)  # (N,)
		else:
			vel_normalized = torch.ones_like(vel_magnitude) * 0.5
		
		if colormap == 'magnitude_inferno':
			cmap_name = 'inferno'
		elif colormap == 'magnitude_afmhot':
			cmap_name = 'afmhot'
		else:
			raise ValueError(f"Unknown colormap: {colormap}")
		
		magnitude_cm = cm.get_cmap(cmap_name)
		
		vel_normalized_np = vel_normalized.detach().cpu().numpy()  # (N,)
		
		rgba_np = magnitude_cm(vel_normalized_np)  # (N, 4)
		
		rgb_np = rgba_np[:, :3]  # (N, 3)
		
		rgb_tensor = torch.from_numpy(rgb_np).float().to(device)
		
		rgb_tensor = torch.clamp(rgb_tensor, 0.0, 1.0)
		
		return rgb_tensor
		
	else:
		raise ValueError(f"Unknown colormap: {colormap}. Supported: 'hsv', 'y_projection', 'magnitude_inferno', 'magnitude_afmhot'")


def velocity_to_opacity(velocity, min_opacity=0.1, max_opacity=1.0, opacity_scale=None, device='cuda', opacity_mode='magnitude'):
	"""
	based on opacity
	
	Args:
		velocity: velocity vector torch.Tensor，shape (N, 3)
		min_opacity:  opacity 
		max_opacity: latest opacity 
		opacity_scale: opacity scaling。if None，
		device: device
		opacity_mode: opacity 
			- 'magnitude': （L2 ）（）
			- 'y_projection_v':  Y  V （transparent at zero，opaque at extrema）
	
	Returns:
		opacity: opacity  torch.Tensor，shape (N, 1)， [min_opacity, max_opacity]
	"""
	if opacity_mode == 'magnitude':
		vel_magnitude = torch.norm(velocity, dim=1, keepdim=True)  # (N, 1)
		
		if opacity_scale is None:
			vel_max = vel_magnitude.max()
			vel_min = vel_magnitude.min()
			if vel_max > vel_min:
				vel_normalized = (vel_magnitude - vel_min) / (vel_max - vel_min)
			else:
				vel_normalized = torch.ones_like(vel_magnitude) * 0.5
		else:
			vel_normalized = torch.clamp(vel_magnitude * opacity_scale, 0.0, 1.0)
		
		opacity = min_opacity + vel_normalized * (max_opacity - min_opacity)
		
		opacity = torch.clamp(opacity, min_opacity, max_opacity)
		
		return opacity
		
	elif opacity_mode == 'y_projection_v':
		
		vy = velocity[:, 1].detach()  # (N,)
		
		vy_abs_max = torch.abs(vy).max()
		if vy_abs_max > 1e-6:
			vy_abs_normalized = torch.abs(vy) / vy_abs_max  # [0, 1]
		else:
			vy_abs_normalized = torch.zeros_like(vy)
		
		threshold = 0.2
		
		vy_abs_normalized_expanded = vy_abs_normalized.unsqueeze(1)  # (N, 1)
		
		mask_low = (vy_abs_normalized_expanded < threshold)
		opacity = torch.where(
			mask_low,
			min_opacity + (max_opacity - min_opacity) * 0.3 * (vy_abs_normalized_expanded / threshold),
			min_opacity + (max_opacity - min_opacity) * (0.3 + 0.7 * (vy_abs_normalized_expanded - threshold) / (1.0 - threshold))
		)
		
		opacity = torch.clamp(opacity, min_opacity, max_opacity)
		
		return opacity
		
	else:
		raise ValueError(f"Unknown opacity_mode: {opacity_mode}. Supported: 'magnitude', 'y_projection_v'")


def velocity_to_color_sh_dc(velocity, scale=300, is3D=True, device='cuda'):
	"""
	velocity vectorconvert to SH DC （）
	，use velocity_to_rgb + override_color
	
	Args:
		velocity: velocity vector torch.Tensor，shape (N, 3)
		scale: scaling，used for HSV 
		is3D:  3D velocity field
		device: device
	
	Returns:
		features_dc: SH DC  torch.Tensor，shape (N, 1, 3)
	"""
	rgb_tensor = velocity_to_rgb(velocity, scale=scale, is3D=is3D, device=device)
	
	features_dc = RGB2SH(rgb_tensor)  # (N, 3)
	
	features_dc = features_dc.unsqueeze(1)  # (N, 1, 3)
	
	return features_dc


def interpolate_velocity_to_gaussian_points(velocity_model, gaussian_points_smoke, grid_shape, lengths_tensor, device, batchsize=10000):
	"""
	velocity fieldinterpolation Gaussian position
	
	Args:
		velocity_model: TiDFRBF velocity field
		gaussian_points_smoke: GS  smoke coordinate，shape (N, 3)
		grid_shape: shape (nx, ny, nz)
		lengths_tensor: length tensor，shape (3,)
		device: device
		batchsize: 
	
	Returns:
		velocities: GS position，shape (N, 3)
	"""
	nx, ny, nz = grid_shape
	
	min_corner = np.zeros(3)
	max_corner = lengths_tensor.cpu().numpy()
	grid_points_np = utils_grid.generate_gridpoints(grid_shape, min_corner, max_corner)
	grid_points = torch.from_numpy(grid_points_np).float().to(device)
	
	velocity_model.eval()
	with torch.no_grad():
		vel_pred_full = torch.zeros((len(grid_points), 3), device=device)
		
		for v_idx in range(0, len(grid_points), batchsize):
			batch_points = grid_points[v_idx:min(len(grid_points), v_idx + batchsize)]
			batch_vel_pred = velocity_model(batch_points)
			vel_pred_full[v_idx:min(len(grid_points), v_idx + batchsize)] = batch_vel_pred
		
		vel_vol = vel_pred_full.view(nx, ny, nz, 3)
		
		vel_vol_5d = vel_vol.permute(3, 2, 1, 0).unsqueeze(0)  # (1, 3, nz, ny, nx)
		
		norm_pos = gaussian_points_smoke.clone()
		norm_pos = 2 * (norm_pos / lengths_tensor) - 1.0
		
		grid_in = norm_pos.view(1, 1, 1, -1, 3)  # (1, 1, 1, N, 3)
		
		v_part = F.grid_sample(vel_vol_5d, grid_in, align_corners=True, mode='bilinear')
		v_part = v_part.view(3, -1).permute(1, 0)  # (N, 3)
		
		return v_part


def visualize_camera_rays_and_grid(cam, opacity_grid, color_grid, grid_shape, coord_trans, lengths_tensor, 
								   near: float, far: float, save_path, num_rays: int = 50, num_samples_per_ray: int = 100):
	"""
	visualizationcameraposition、 opacity/color grid 
	
	Args:
		cam: Camera object
		opacity_grid: opacity ，shape (nx, ny, nz)，numpy array
		color_grid: color ，shape (nx, ny, nz, 3)，numpy array
		grid_shape: shape (nx, ny, nz)
		coord_trans: coordinateobject
		lengths_tensor: length tensor，shape (3,)
		near: near 
		far: far 
		save_path: save path（PNG ）
		num_rays: visualizationcount（ 50）
		num_samples_per_ray: （ 100）
	"""
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	
	device = lengths_tensor.device
	H = int(cam.image_height)
	W = int(cam.image_width)
	
	with torch.no_grad():
		rays_o, rays_d = get_rays_from_camera(cam)  # (H, W, 3), (H, W, 3)
		rays_o_flat = rays_o.detach().reshape(-1, 3)  # (H*W, 3)
		rays_d_flat = rays_d.detach().reshape(-1, 3)  # (H*W, 3)
		rays_d_norm = torch.norm(rays_d_flat, dim=-1, keepdim=True)
		rays_d_normalized = (rays_d_flat / (rays_d_norm + 1e-10)).detach()
	
	total_rays = rays_o_flat.shape[0]
	ray_indices = np.linspace(0, total_rays - 1, num_rays, dtype=int)
	selected_rays_o = rays_o_flat[ray_indices].cpu().numpy()  # (num_rays, 3)
	selected_rays_d = rays_d_normalized[ray_indices].cpu().numpy()  # (num_rays, 3)
	
	t_vals = np.linspace(0., 1., num_samples_per_ray)
	z_vals = near * (1. - t_vals) + far * t_vals  # (num_samples_per_ray,)
	
	nx, ny, nz = grid_shape
	lengths_np = lengths_tensor.cpu().numpy()
	grid_corners_smoke = np.array([
		[0.0, 0.0, 0.0],
		[lengths_np[0], 0.0, 0.0],
		[0.0, lengths_np[1], 0.0],
		[0.0, 0.0, lengths_np[2]],
		[lengths_np[0], lengths_np[1], 0.0],
		[lengths_np[0], 0.0, lengths_np[2]],
		[0.0, lengths_np[1], lengths_np[2]],
		[lengths_np[0], lengths_np[1], lengths_np[2]]
	])
	
	grid_corners_smoke_norm = grid_corners_smoke / lengths_np[None, :]
	grid_corners_world = coord_trans.smoke2world(torch.from_numpy(grid_corners_smoke_norm).float().to(device)).cpu().numpy()
	
	fig = plt.figure(figsize=(16, 12))
	ax = fig.add_subplot(111, projection='3d')
	
	edges = [
		[0, 1], [0, 2], [0, 3],
		[1, 4], [1, 5],
		[2, 4], [2, 6],
		[3, 5], [3, 6],
		[4, 7], [5, 7], [6, 7]
	]
	for edge in edges:
		ax.plot3D(*grid_corners_world[edge].T, 'k-', alpha=0.3, linewidth=0.5)
	
	camera_origin = selected_rays_o[0]
	ax.scatter(*camera_origin, c='red', s=200, marker='o', label='Camera Origin', zorder=10)
	
	opacity_grid_tensor = torch.from_numpy(opacity_grid).float().to(device)
	color_grid_tensor = torch.from_numpy(color_grid).float().to(device)
	
	for ray_idx in range(num_rays):
		ray_o = selected_rays_o[ray_idx]
		ray_d = selected_rays_d[ray_idx]
		
		points_world = ray_o[None, :] + ray_d[None, :] * z_vals[:, None]  # (num_samples_per_ray, 3)
		
		points_world_tensor = torch.from_numpy(points_world).float().to(device)
		points_smoke = coord_trans.world2smoke(points_world_tensor).cpu().numpy()
		points_smoke_norm = points_smoke / lengths_np[None, :]
		points_smoke_norm = np.clip(points_smoke_norm, 0.0, 1.0)
		
		grid_indices = (points_smoke_norm * np.array([nx-1, ny-1, nz-1])).astype(int)
		grid_indices = np.clip(grid_indices, 0, [nx-1, ny-1, nz-1])
		opacity_values = opacity_grid[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]]
		
		mask_inside = np.all((points_smoke_norm >= 0) & (points_smoke_norm <= 1), axis=1)
		if np.any(mask_inside):
			points_inside = points_world[mask_inside]
			opacity_inside = opacity_values[mask_inside]
			
			if len(points_inside) > 1:
				for i in range(len(points_inside) - 1):
					alpha_val = opacity_inside[i]
					color_val = [alpha_val, alpha_val, alpha_val]
					ax.plot3D(*points_inside[i:i+2].T, color=color_val, alpha=0.6, linewidth=1.0)
			
			high_opacity_mask = opacity_inside > 0.1
			if np.any(high_opacity_mask):
				high_opacity_points = points_inside[high_opacity_mask]
				high_opacity_values = opacity_inside[high_opacity_mask]
				ax.scatter(*high_opacity_points.T, c=high_opacity_values, cmap='hot', 
						  s=20, alpha=0.8, vmin=0, vmax=opacity_grid.max(), zorder=5)
	
	ax.set_xlabel('X (World Space)', fontsize=12)
	ax.set_ylabel('Y (World Space)', fontsize=12)
	ax.set_zlabel('Z (World Space)', fontsize=12)
	ax.set_title(f'Camera Rays and Opacity Grid\nCamera Origin: ({camera_origin[0]:.2f}, {camera_origin[1]:.2f}, {camera_origin[2]:.2f})', fontsize=14)
	ax.legend()
	
	all_points = np.concatenate([grid_corners_world, selected_rays_o, 
								 selected_rays_o + selected_rays_d * far], axis=0)
	max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
						  all_points[:, 1].max() - all_points[:, 1].min(),
						  all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
	mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
	mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
	mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)
	
	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f"    Saved camera rays visualization: {save_path}")


def save_vtk_image_data(data, origin, spacing, filename, scalar_name="opacity"):
	"""
	Save 3D grid data as VTK Image Data (.vti) format for ParaView.
	
	Args:
		data: 3D numpy array of shape (nx, ny, nz)
		origin: Origin point [x0, y0, z0]
		spacing: Spacing between grid points [dx, dy, dz]
		filename: Output filename (.vti)
		scalar_name: Name of the scalar field in VTK file (default: "opacity")
	"""
	# Ensure data is contiguous and float32
	data = np.ascontiguousarray(data, dtype=np.float32)
	
	# Get dimensions
	if len(data.shape) != 3:
		raise ValueError(f"Data must be 3D array, got shape {data.shape}")
	
	# VTK uses (nx, ny, nz) convention where nx is fastest changing
	# Our data is (nx, ny, nz) from grid_shape
	nx, ny, nz = data.shape
	
	# Write VTK XML Image Data format
	with open(filename, 'w') as f:
		f.write('<?xml version="1.0"?>\n')
		f.write('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
		f.write(f'  <ImageData WholeExtent="0 {nx-1} 0 {ny-1} 0 {nz-1}" ')
		f.write(f'Origin="{origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}" ')
		f.write(f'Spacing="{spacing[0]:.6f} {spacing[1]:.6f} {spacing[2]:.6f}">\n')
		f.write(f'    <Piece Extent="0 {nx-1} 0 {ny-1} 0 {nz-1}">\n')
		f.write(f'      <PointData Scalars="{scalar_name}">\n')
		f.write(f'        <DataArray type="Float32" Name="{scalar_name}" format="ascii" NumberOfComponents="1">\n')
		
		# Write data in VTK order: x varies fastest, then y, then z
		# VTK Image Data uses (x, y, z) indexing where x is fastest changing
		# Our data is (nx, ny, nz) = (x, y, z), so we iterate z, y, x (outer to inner)
		# and write data[i, j, k] where i=x, j=y, k=z
		for k in range(nz):  # z (outermost)
			for j in range(ny):  # y (middle)
				for i in range(nx):  # x (innermost, fastest changing)
					f.write(f'          {data[i, j, k]:.6e}\n')
		
		f.write('        </DataArray>\n')
		f.write('      </PointData>\n')
		f.write('    </Piece>\n')
		f.write('  </ImageData>\n')
		f.write('</VTKFile>\n')
	
	print(f"Saved VTK Image Data to {filename}")


def compute_opacity_color_grids(gaussians, velocity_model, grid_shape, coord_trans, lengths_tensor, device,
								vel_color_scale: float = 300, use_velocity_opacity: bool = True,
								min_opacity: float = 0.1, max_opacity: float = 1.0, opacity_scale: float = None,
								colormap: str = 'hsv', opacity_mode: str = 'magnitude',
								batch_size: int = 1000, sigma_threshold: float = 3.0,
								opacity_multiplier: float = 0.0):
	"""
	 GS velocity field opacity  color 
	
	Args:
		gaussians: GaussianModel object
		velocity_model: TiDFRBF velocity field
		grid_shape: shape (nx, ny, nz)
		coord_trans: coordinateobject
		lengths_tensor: length tensor，shape (3,)
		device: device
		vel_color_scale: scaling
		use_velocity_opacity: based on opacity
		min_opacity:  opacity 
		max_opacity: latest opacity 
		opacity_scale: opacity scaling
		colormap: 
		opacity_mode: opacity 
		batch_size: 
		sigma_threshold: sigma ，used for
		opacity_multiplier: opacity scaling（ 1.0）
	
	Returns:
		opacity_grid: opacity ，shape (nx, ny, nz)
		color_grid: color ，shape (nx, ny, nz, 3)
		heat_grid: heat （velocity field magnitude ），shape (nx, ny, nz)
	"""
	nx, ny, nz = grid_shape
	
	min_corner = np.zeros(3)
	max_corner = lengths_tensor.cpu().numpy()
	grid_points_smoke_np = utils_grid.generate_gridpoints(grid_shape, min_corner, max_corner)
	grid_points_smoke = torch.from_numpy(grid_points_smoke_np).float().to(device)  # (N_grid, 3)
	
	num_grid_points = grid_points_smoke.shape[0]
	grid_points_smoke_norm = grid_points_smoke / lengths_tensor.unsqueeze(0)
	grid_points_world = coord_trans.smoke2world(grid_points_smoke_norm)  # (N_grid, 3)
	
	means3D = gaussians.get_xyz.detach().to(device)  # (N_gauss, 3)
	scales = gaussians.get_scaling.detach().to(device)  # (N_gauss, 3)
	rotations = gaussians.get_rotation.detach().to(device)  # (N_gauss, 4)
	opacity = gaussians.get_opacity.detach().to(device)  # (N_gauss, 1)
	
	if opacity.dim() > 1:
		opacity = opacity.squeeze(-1)  # (N_gauss,)
	
	num_gaussians = means3D.shape[0]
	
	max_scales = torch.max(scales, dim=1)[0]  # (N_gauss,)
	max_radius = max_scales * sigma_threshold  # (N_gauss,)
	
	from scene.gaussian_model import build_scaling_rotation
	with torch.no_grad():
		L = build_scaling_rotation(scales, rotations)  # (N_gauss, 3, 3)
		cov3D = L @ L.transpose(-2, -1)  # (N_gauss, 3, 3)
		
		cov3D_reg = cov3D + torch.eye(3, device=device).unsqueeze(0) * 1e-6
		cov3D_inv = torch.inverse(cov3D_reg)  # (N_gauss, 3, 3)
	
	velocity_model.eval()
	with torch.no_grad():
		vel_pred_full = torch.zeros((num_grid_points, 3), device=device)
		num_batches = (num_grid_points + batch_size - 1) // batch_size
		for v_idx in tqdm(range(0, num_grid_points, batch_size), desc="Predicting velocity field", leave=False):
			batch_points = grid_points_smoke[v_idx:min(num_grid_points, v_idx + batch_size)].detach()
			batch_vel_pred = velocity_model(batch_points).detach()
			vel_pred_full[v_idx:min(num_grid_points, v_idx + batch_size)] = batch_vel_pred
			del batch_points, batch_vel_pred
			torch.cuda.empty_cache()
	
	with torch.no_grad():
		rgb_colors = velocity_to_rgb(
			vel_pred_full.detach(), scale=vel_color_scale, is3D=True, device=device, colormap=colormap
		).detach()  # (N_grid, 3)
	
	opacity_values = torch.zeros(num_grid_points, device=device)
	velocity_opacity_factors = torch.ones(num_grid_points, device=device)
	
	if use_velocity_opacity:
		with torch.no_grad():
			velocity_opacity_factors_raw = velocity_to_opacity(
				vel_pred_full.detach(),
				min_opacity=min_opacity,
				max_opacity=max_opacity,
				opacity_scale=opacity_scale,
				device=device,
				opacity_mode=opacity_mode
			).detach()
			if velocity_opacity_factors_raw.dim() > 1:
				velocity_opacity_factors = velocity_opacity_factors_raw.squeeze(-1)  # (N_grid,)
			else:
				velocity_opacity_factors = velocity_opacity_factors_raw  # (N_grid,)
			del velocity_opacity_factors_raw
			torch.cuda.empty_cache()
	
	opacity_batch_size = min(batch_size, 500)
	num_opacity_batches = (num_grid_points + opacity_batch_size - 1) // opacity_batch_size
	for start_idx in tqdm(range(0, num_grid_points, opacity_batch_size), desc="Computing GS opacity", leave=False):
		end_idx = min(start_idx + opacity_batch_size, num_grid_points)
		query_pts_batch = grid_points_world[start_idx:end_idx].detach().to(device)  # (batch_size, 3)
		
		with torch.no_grad():
			diff = query_pts_batch.unsqueeze(1) - means3D.unsqueeze(0)  # (batch_size, N_gauss, 3)
			
			dist = torch.norm(diff, dim=2)  # (batch_size, N_gauss)
			
			valid_mask = dist <= max_radius.unsqueeze(0)  # (batch_size, N_gauss)
			
			cov_inv_diff = torch.einsum('qij,ijk->qik', diff, cov3D_inv)  # (batch_size, N_gauss, 3)
			mahal_dist = torch.sum(diff * cov_inv_diff, dim=2)  # (batch_size, N_gauss)
			
			contribution = opacity.unsqueeze(0) * torch.exp(-0.5 * mahal_dist)  # (batch_size, N_gauss)
			
			contribution = contribution * valid_mask.float()
			
			opacity_values[start_idx:end_idx] = torch.sum(contribution, dim=1).detach()  # (batch_size,)
		
		del diff, dist, valid_mask, cov_inv_diff, mahal_dist, contribution, query_pts_batch
		torch.cuda.empty_cache()
	
	with torch.no_grad():
		if opacity_values.dim() > 1:
			opacity_values = opacity_values.squeeze(-1)
		if velocity_opacity_factors.dim() > 1:
			velocity_opacity_factors = velocity_opacity_factors.squeeze(-1)
		final_opacity = (opacity_values * velocity_opacity_factors * opacity_multiplier).detach()  # (N_grid,)
	
	with torch.no_grad():
		vel_magnitude = torch.norm(vel_pred_full, dim=1)  # (N_grid,)
		
		vel_max = vel_magnitude.max()
		vel_min = vel_magnitude.min()
		if vel_max > vel_min:
			heat_normalized = (vel_magnitude - vel_min) / (vel_max - vel_min)  # (N_grid,)
		else:
			heat_normalized = torch.ones_like(vel_magnitude) * 0.5
		
		heat_cpu = heat_normalized.cpu()
	
	with torch.no_grad():
		final_opacity_cpu = final_opacity.cpu()
		rgb_colors_cpu = rgb_colors.cpu()
	
	opacity_grid = final_opacity_cpu.numpy().reshape(grid_shape)  # (nx, ny, nz)
	color_grid = rgb_colors_cpu.numpy().reshape(nx, ny, nz, 3)  # (nx, ny, nz, 3)
	heat_grid = heat_cpu.numpy().reshape(grid_shape)  # (nx, ny, nz)
	
	del opacity_values, velocity_opacity_factors, final_opacity, final_opacity_cpu
	del rgb_colors, rgb_colors_cpu, vel_pred_full, vel_magnitude, heat_normalized, heat_cpu
	del means3D, scales, rotations, opacity, max_scales, max_radius
	del L, cov3D, cov3D_reg, cov3D_inv
	del grid_points_smoke, grid_points_world
	torch.cuda.empty_cache()
	
	return opacity_grid, color_grid, heat_grid


def ray_march_render(cam, opacity_grid, color_grid, grid_shape, coord_trans, lengths_tensor, 
					background: torch.Tensor, device, near: float = None, far: float = None,
					N_samples: int = 1024, chunk: int = 8192):
	"""
	use ray marching 
	
	Args:
		cam: Camera object
		opacity_grid: opacity ，shape (nx, ny, nz)，numpy array
		color_grid: color ，shape (nx, ny, nz, 3)，numpy array
		grid_shape: shape (nx, ny, nz)
		coord_trans: coordinateobject
		lengths_tensor: length tensor，shape (3,)
		background: ，shape (3,)
		device: device
		near: near （if None，）
		far: far （if None，）
		N_samples: 
		chunk: （）
	
	Returns:
		render_image: ，shape (H, W, 3)，torch.Tensor， [0, 1]
		transmittance_image: ，shape (H, W)，torch.Tensor， [0, 1]
	"""
	H = int(cam.image_height)
	W = int(cam.image_width)
	
	with torch.no_grad():
		rays_o, rays_d = get_rays_from_camera(cam)  # (H, W, 3), (H, W, 3)
		
		rays_o_flat = rays_o.detach().reshape(-1, 3)  # (H*W, 3)
		rays_d_flat = rays_d.detach().reshape(-1, 3)  # (H*W, 3)
		
		rays_d_norm = torch.norm(rays_d_flat, dim=-1, keepdim=True)
		rays_d_normalized = (rays_d_flat / (rays_d_norm + 1e-10)).detach()
	
	if near is None or far is None:
		smoke_corners_norm = torch.tensor([
			[0.0, 0.0, 0.0],
			[1.0, 1.0, 1.0]
		], device=device)
		
		world_corners = coord_trans.smoke2world(smoke_corners_norm)  # (2, 3)
		
		camera_center = rays_o_flat[0]
		distances_to_corners = torch.norm(world_corners - camera_center.unsqueeze(0), dim=-1)
		
		if near is None:
			near = max(0.1, distances_to_corners.min().item() * 0.5)
		if far is None:
			far = distances_to_corners.max().item() * 1.5
	
	opacity_grid_tensor = torch.from_numpy(opacity_grid).float().to(device)  # (nx, ny, nz)
	color_grid_tensor = torch.from_numpy(color_grid).float().to(device)  # (nx, ny, nz, 3)
	
	nx, ny, nz = grid_shape
	opacity_grid_5d = opacity_grid_tensor.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)  # (1, 1, nz, ny, nx)
	color_grid_5d = color_grid_tensor.permute(3, 2, 1, 0).unsqueeze(0)  # (1, 3, nz, ny, nx)
	
	N_rays = rays_o_flat.shape[0]
	all_rgb = []
	all_transmittance = []
	num_chunks = (N_rays + chunk - 1) // chunk
	
	for i in tqdm(range(0, N_rays, chunk), desc="Ray marching", leave=False, total=num_chunks):
		end_i = min(i + chunk, N_rays)
		chunk_rays_o = rays_o_flat[i:end_i]  # (chunk_size, 3)
		chunk_rays_d = rays_d_normalized[i:end_i]  # (chunk_size, 3)
		chunk_N_rays = chunk_rays_o.shape[0]
		
		t_vals = torch.linspace(0., 1., steps=N_samples, device=device)
		z_vals = near * (1. - t_vals) + far * t_vals
		z_vals = z_vals.expand([chunk_N_rays, N_samples])  # (chunk_N_rays, N_samples)
		
		with torch.no_grad():
			points_world = chunk_rays_o[..., None, :] + chunk_rays_d[..., None, :] * z_vals[..., :, None]  # (chunk_N_rays, N_samples, 3)
			points_world_flat = points_world.view(-1, 3).detach()  # (chunk_N_rays * N_samples, 3)
			
			points_smoke = coord_trans.world2smoke(points_world_flat).detach()  # (chunk_N_rays * N_samples, 3)
			
			points_smoke_norm = (points_smoke / lengths_tensor.unsqueeze(0)).detach()  # (chunk_N_rays * N_samples, 3)
			points_smoke_norm = torch.clamp(points_smoke_norm, 0.0, 1.0)
			
			grid_coords = (2.0 * points_smoke_norm - 1.0).detach()  # (chunk_N_rays * N_samples, 3)
			grid_coords = grid_coords.view(chunk_N_rays, N_samples, 1, 1, 3)  # (chunk_N_rays, N_samples, 1, 1, 3)
		
		del points_world, points_world_flat, points_smoke, points_smoke_norm
		
		with torch.no_grad():
			opacity_samples = F.grid_sample(
				opacity_grid_5d.expand(chunk_N_rays, -1, -1, -1, -1),
				grid_coords,
				mode='bilinear',
				padding_mode='zeros',
				align_corners=True
			).detach()  # (chunk_N_rays, 1, N_samples, 1, 1)
			opacity_samples = opacity_samples.squeeze(-1).squeeze(-1).squeeze(1)  # (chunk_N_rays, N_samples)
			
			color_samples = F.grid_sample(
				color_grid_5d.expand(chunk_N_rays, -1, -1, -1, -1),
				grid_coords,
				mode='bilinear',
				padding_mode='zeros',
				align_corners=True
			).detach()  # (chunk_N_rays, 3, N_samples, 1, 1)
			color_samples = color_samples.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (chunk_N_rays, N_samples, 3)
		
		del grid_coords
		
		with torch.no_grad():
			dists = z_vals[..., 1:] - z_vals[..., :-1]  # (chunk_N_rays, N_samples-1)
			dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)  # (chunk_N_rays, N_samples)
			
			opacity_safe = torch.clamp(opacity_samples, min=0.0)  # (chunk_N_rays, N_samples)
			
			alpha = (1.0 - torch.exp(-opacity_safe * dists)).detach()  # (chunk_N_rays, N_samples)
			alpha = torch.clamp(alpha, min=0.0, max=1.0)
			
			transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
			transmittance = torch.clamp(transmittance, min=0.0, max=1.0)
			
			weights = (alpha * transmittance).detach()  # (chunk_N_rays, N_samples)
			weights = torch.clamp(weights, min=0.0)
			
			rgb_map = torch.sum(weights[..., None] * color_samples, dim=-2).detach()  # (chunk_N_rays, 3)
			rgb_map = torch.clamp(rgb_map, min=0.0, max=1.0)
			
			final_transmittance = torch.prod(1. - alpha + 1e-10, dim=-1).detach()  # (chunk_N_rays,)
			final_transmittance = torch.clamp(final_transmittance, min=0.0, max=1.0)
			
			if background is not None:
				background_tensor = background.to(device) if isinstance(background, torch.Tensor) else torch.tensor(background, device=device)
				rgb_map = (rgb_map + background_tensor.unsqueeze(0) * final_transmittance[..., None]).detach()
		
		del opacity_samples, color_samples, dists, opacity_safe, alpha, transmittance, weights
		if 'background_tensor' in locals():
			del background_tensor
		
		all_rgb.append(rgb_map.detach())
		all_transmittance.append(final_transmittance.detach())
		del rgb_map, final_transmittance
		torch.cuda.empty_cache()
	
	rgb_map_full = torch.cat(all_rgb, dim=0)  # (H*W, 3)
	transmittance_map_full = torch.cat(all_transmittance, dim=0)  # (H*W,)
	
	render_image = rgb_map_full.detach().reshape(H, W, 3)  # (H, W, 3)
	transmittance_image = transmittance_map_full.detach().reshape(H, W)  # (H, W)
	
	return render_image, transmittance_image


def evaluate_velocity_colored_reconstruction(args, ckpt_dir: str, savedir: str, scale: float = None, vel_color_scale: float = 300,
											use_velocity_opacity: bool = True, min_opacity: float = 0.1, max_opacity: float = 1.0, opacity_scale: float = None,
											start_frame: int = None, end_frame: int = None,
											colormap: str = 'hsv', opacity_mode: str = 'magnitude',
											use_raymarching: bool = False, raymarching_samples: int = 1024, raymarching_chunk: int = 8192,
											opacity_multiplier: float = 1.0, save_raymarching_extra_outputs: bool = False):
	"""
	evaluatefull reconstruction process，usevelocity field
	
	This function will:：
	1.  ckpt_dir  checkpoint（frame_gaussians/, frame_velocities/, window_*_*/ ）
	2. frame 0 framestart advect， end_frame（）
	3.  start_frame -> end_frame frame（）
	4. GS velocity fieldinterpolationuse
	5. GS  opacity = original opacity *  opacity（）
	6. frame metrics（PSNR, SSIM, LPIPS）
	7. Results saved to savedir in
	
	Args:
		args: argument object
		ckpt_dir: checkpoint （contains frame_gaussians/, frame_velocities/, window_*_*/ ）
		savedir: output directory（）
		scale: resize scale（optional， args ）
		vel_color_scale: scaling（used for HSV ， colormap='hsv' use）
		use_velocity_opacity: based on opacity
		min_opacity:  opacity 
		max_opacity: latest opacity 
		opacity_scale: opacity scaling。if None， [0, 1]
		start_frame: frame index（None frame 0 framestart）
		end_frame: frame index（None frame，contains）
		colormap: ，'hsv'（）、'y_projection'（Y：->，->）、'magnitude_inferno'（->inferno colormap） 'magnitude_afmhot'（->afmhot colormap）
		opacity_mode: opacity ，'magnitude'（，） 'y_projection_v'（YV）
		use_raymarching: use ray marching （True） GS （False，）。Ray marching velocity fieldin， GS
		raymarching_samples: ray marching （ 1024）
		raymarching_chunk: ray marching （ 8192）
		opacity_multiplier: opacity scaling（ 1.0），used for opacity 
		save_raymarching_extra_outputs:  use_raymarching=True ，output（VTK、visualization、）。 False， npz file（contains opacity, color, heat）
	
	Returns:
		metrics_dict: contains metrics 
	"""
	device = set_device(args)
	
	lpips_vgg_model = LPIPS(net_type='vgg', version='0.1').to(device).eval()
	lpips_alex_model = LPIPS(net_type='alex', version='0.1').to(device).eval()
	
	if scale is None:
		scale = getattr(args, 'scale', 1)
	
	# ============================================================
	# ============================================================
	print(f"\n{'='*80}")
	print(f"Evaluating Velocity Colored Reconstruction")
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
	
	window_pattern = os.path.join(ckpt_dir, "window_*_*")
	window_dirs = glob.glob(window_pattern)
	if not window_dirs:
		raise ValueError(f"No window directories found in {ckpt_dir}")
	
	first_window_name = os.path.basename(window_dirs[0])
	match = re.match(r'window_(\d+)_(\d+)', first_window_name)
	if not match:
		raise ValueError(f"Invalid window directory name: {first_window_name}")
	
	w_start = int(match.group(1))
	w_end = int(match.group(2))
	window_size = w_end - w_start
	print(f"Detected window size: {window_size} (from {first_window_name})")
	
	if start_frame is None:
		start_frame = 0
	if end_frame is None:
		end_frame = total_frame_num
	
	start_frame = max(0, min(start_frame, total_frame_num))
	end_frame = max(start_frame, min(end_frame, total_frame_num))
	
	print(f"Render frame range: {start_frame} -> {end_frame} (total frames: {total_frame_num})")
	print(f"Advect will start from frame 0 and continue until frame {end_frame}")
	
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
	print(f"Total frames: {total_frame_num}")
	print(f"Velocity color scale: {vel_color_scale}")
	
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
	
	gaussian_state_path = os.path.join(ckpt_dir, "frame_gaussians", f"frame_000_gaussian.pth")
	if not os.path.exists(gaussian_state_path):
		raise FileNotFoundError(f"Frame 0 Gaussian state not found: {gaussian_state_path}")
	gaussian_state, _ = torch.load(gaussian_state_path)
	gaussians.restore(gaussian_state, opt)
	print(f"  Loaded initial GS from frame 0: {gaussian_state_path}")
	
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
	for t in range(total_frame_num):
		frame_to_train_cameras[t] = []
		frame_to_test_cameras[t] = []
	
	for cam_idx, cam in enumerate(train_cameras):
		if hasattr(cam, 'time') and cam.time is not None:
			if total_frame_num > 1:
				frame_idx = round(cam.time * (total_frame_num - 1))
			else:
				frame_idx = 0
			frame_idx = max(0, min(frame_idx, total_frame_num - 1))
			frame_to_train_cameras[frame_idx].append((cam_idx, cam))
	
	for cam_idx, cam in enumerate(test_cameras):
		if hasattr(cam, 'time') and cam.time is not None:
			if total_frame_num > 1:
				frame_idx = round(cam.time * (total_frame_num - 1))
			else:
				frame_idx = 0
			frame_idx = max(0, min(frame_idx, total_frame_num - 1))
			frame_to_test_cameras[frame_idx].append((cam_idx, cam))
	
	print(f"  Mapped cameras for {total_frame_num} frames")
	
	# ============================================================
	# ============================================================
	pp = PipelineParams(ArgumentParser())
	pipe = pp.extract(args)
	background = get_background_color(args, device)
	
	# ============================================================
	# ============================================================
	print(f"\n[Step 3] Processing all frames...")
	
	train_camera_psnrs = []
	train_camera_ssims = []
	train_camera_lpips_vgg = []
	train_camera_lpips_alex = []
	train_camera_frame_indices = []
	train_camera_renders = []
	train_camera_gts = []
	train_camera_info = []
	train_camera_transmittance = []
	
	test_camera_psnrs = []
	test_camera_ssims = []
	test_camera_lpips_vgg = []
	test_camera_lpips_alex = []
	test_camera_frame_indices = []
	test_camera_renders = []
	test_camera_gts = []
	test_camera_info = []
	test_camera_transmittance = []
	
	current_pos_sim = None
	xyz_world_0 = gaussians.get_xyz.detach().clone()
	xyz_smoke_0 = coord_trans.world2smoke(xyz_world_0)
	current_pos_sim = xyz_smoke_0 * lengths_tensor
	
	from velocity_training.train import _merge_inflow_to_gaussians
	
	frame_pbar = tqdm(range(end_frame), desc="Processing frames", unit="frame")
	for t in frame_pbar:
		frame_pbar.set_description(f"Processing frame {t}/{end_frame-1}")
		print(f"\n  Processing frame {t}...")
		
		should_render = (t >= start_frame)
		
		if t < total_frame_num - window_size:
			window_start = t
			window_end = t + window_size
		else:
			window_start = total_frame_num - window_size
			window_end = total_frame_num
		
		print(f"    Window: {window_start}->{window_end}")
		
		window_dir = os.path.join(ckpt_dir, f"window_{window_start}_{window_end}")
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
			prev_vel_model_path = os.path.join(ckpt_dir, "frame_velocities", f"frame_{t-1:03d}_velocity.pth")
			if not os.path.exists(prev_vel_model_path):
				current_window_dir = os.path.join(ckpt_dir, f"window_{window_start}_{window_end}")
				vel_pattern = os.path.join(current_window_dir, "ckpt", f"velrbf_frame_{t-1:03d}_ckpt_*.pth")
				vel_files = glob.glob(vel_pattern)
				if not vel_files:
					prev_window_dir = os.path.join(ckpt_dir, f"window_{t-1}_{t-1+window_size}")
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
			
			gaussian_state_path = os.path.join(ckpt_dir, "frame_gaussians", f"frame_{t:03d}_gaussian.pth")
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
			prev_vel_model_path = os.path.join(ckpt_dir, "frame_velocities", f"frame_{t-1:03d}_velocity.pth")
			if not os.path.exists(prev_vel_model_path):
				current_window_dir = os.path.join(ckpt_dir, f"window_{window_start}_{window_end}")
				vel_pattern = os.path.join(current_window_dir, "ckpt", f"velrbf_frame_{t-1:03d}_ckpt_*.pth")
				vel_files = glob.glob(vel_pattern)
				if not vel_files:
					prev_window_dir = os.path.join(ckpt_dir, f"window_{t-1}_{t-1+window_size}")
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
		
		velocity_model = None
		if t == 0:
			print(f"    Frame 0: Using original GS colors")
			if use_raymarching:
				vel_model_path = os.path.join(ckpt_dir, "frame_velocities", f"frame_000_velocity.pth")
				if not os.path.exists(vel_model_path):
					current_window_dir = os.path.join(ckpt_dir, f"window_{window_start}_{window_end}")
					vel_pattern = os.path.join(current_window_dir, "ckpt", f"velrbf_frame_000_ckpt_*.pth")
					vel_files = glob.glob(vel_pattern)
					if vel_files:
						vel_model_path = max(vel_files, key=os.path.getctime)
				if vel_model_path and os.path.exists(vel_model_path):
					velocity_model = TiDFRBF.load(vel_model_path, device=device)
					print(f"    Loaded velocity model for frame 0: {vel_model_path}")
					print(f"    Computing opacity and color grids for ray marching (frame 0)...")
					opacity_grid, color_grid, heat_grid = compute_opacity_color_grids(
						gaussians, velocity_model, grid_shape, coord_trans, lengths_tensor, device,
						vel_color_scale=vel_color_scale, use_velocity_opacity=use_velocity_opacity,
						min_opacity=min_opacity, max_opacity=max_opacity, opacity_scale=opacity_scale,
						colormap=colormap, opacity_mode=opacity_mode,
						opacity_multiplier=opacity_multiplier
					)
					print(f"    Opacity grid: shape={opacity_grid.shape}, range=[{opacity_grid.min():.4f}, {opacity_grid.max():.4f}]")
					print(f"    Color grid: shape={color_grid.shape}, range=[{color_grid.min():.4f}, {color_grid.max():.4f}]")
					print(f"    Heat grid: shape={heat_grid.shape}, range=[{heat_grid.min():.4f}, {heat_grid.max():.4f}]")
					
					eval_savedir = os.path.join(savedir, "velocity_colored_reconstruction_evaluation")
					npz_dir = os.path.join(eval_savedir, "npz_grids")
					os.makedirs(npz_dir, exist_ok=True)
					
					nx, ny, nz = grid_shape
					color_grid_np = color_grid.reshape(nx, ny, nz, 3)
					opacity_grid_np = opacity_grid.reshape(nx, ny, nz, 1)
					heat_grid_np = heat_grid.reshape(nx, ny, nz, 1)
					npz_filename = os.path.join(npz_dir, f"frame_{t:03d}_color_opacity_heat.npz")
					np.savez(npz_filename, color=color_grid_np, opacity=opacity_grid_np, heat=heat_grid_np)
					print(f"    Saved color, opacity & heat grids to NPZ: {npz_filename}")
					
					if save_raymarching_extra_outputs:
						vtk_dir = os.path.join(eval_savedir, "vtk_opacity_grids")
						os.makedirs(vtk_dir, exist_ok=True)
						
						origin = [0.0, 0.0, 0.0]
						lengths_np = lengths_tensor.cpu().numpy()
						spacing = [
							lengths_np[0] / max(nx - 1, 1),
							lengths_np[1] / max(ny - 1, 1),
							lengths_np[2] / max(nz - 1, 1)
						]
						
						vtk_filename = os.path.join(vtk_dir, f"frame_{t:03d}_opacity.vti")
						save_vtk_image_data(opacity_grid, origin, spacing, vtk_filename, scalar_name="opacity")
						print(f"    Saved opacity grid to VTK format: {vtk_filename}")
						
						mid_x = nx // 2
						mid_y = ny // 2
						mid_z = nz // 2
						
						vis_dir = os.path.join(eval_savedir, "grid_visualizations")
						os.makedirs(vis_dir, exist_ok=True)
						
						opacity_slice_x = opacity_grid[mid_x, :, :]  # (ny, nz)
						color_slice_x = color_grid[mid_x, :, :, :]  # (ny, nz, 3)
						opacity_min_x, opacity_max_x = opacity_slice_x.min(), opacity_slice_x.max()
						if opacity_max_x > opacity_min_x:
							opacity_img_x = (opacity_slice_x - opacity_min_x) / (opacity_max_x - opacity_min_x)
						else:
							opacity_img_x = opacity_slice_x * 0
						opacity_img_x = np.clip(opacity_img_x, 0, 1)
						opacity_img_x_uint8 = (opacity_img_x * 255).astype(np.uint8)
						opacity_path_x = os.path.join(vis_dir, f"frame_{t:03d}_opacity_midslice_x.png")
						imageio.imwrite(opacity_path_x, opacity_img_x_uint8)
						color_img_x = np.clip(color_slice_x, 0, 1)
						color_img_x_uint8 = (color_img_x * 255).astype(np.uint8)
						color_path_x = os.path.join(vis_dir, f"frame_{t:03d}_color_midslice_x.png")
						imageio.imwrite(color_path_x, color_img_x_uint8)
						
						opacity_slice_y = opacity_grid[:, mid_y, :]  # (nx, nz)
						color_slice_y = color_grid[:, mid_y, :, :]  # (nx, nz, 3)
						opacity_min_y, opacity_max_y = opacity_slice_y.min(), opacity_slice_y.max()
						if opacity_max_y > opacity_min_y:
							opacity_img_y = (opacity_slice_y - opacity_min_y) / (opacity_max_y - opacity_min_y)
						else:
							opacity_img_y = opacity_slice_y * 0
						opacity_img_y = np.clip(opacity_img_y, 0, 1)
						opacity_img_y_uint8 = (opacity_img_y * 255).astype(np.uint8)
						opacity_path_y = os.path.join(vis_dir, f"frame_{t:03d}_opacity_midslice_y.png")
						imageio.imwrite(opacity_path_y, opacity_img_y_uint8)
						color_img_y = np.clip(color_slice_y, 0, 1)
						color_img_y_uint8 = (color_img_y * 255).astype(np.uint8)
						color_path_y = os.path.join(vis_dir, f"frame_{t:03d}_color_midslice_y.png")
						imageio.imwrite(color_path_y, color_img_y_uint8)
						
						opacity_slice_z = opacity_grid[:, :, mid_z]  # (nx, ny)
						color_slice_z = color_grid[:, :, mid_z, :]  # (nx, ny, 3)
						opacity_min_z, opacity_max_z = opacity_slice_z.min(), opacity_slice_z.max()
						if opacity_max_z > opacity_min_z:
							opacity_img_z = (opacity_slice_z - opacity_min_z) / (opacity_max_z - opacity_min_z)
						else:
							opacity_img_z = opacity_slice_z * 0
						opacity_img_z = np.clip(opacity_img_z, 0, 1)
						opacity_img_z_uint8 = (opacity_img_z * 255).astype(np.uint8)
						opacity_path_z = os.path.join(vis_dir, f"frame_{t:03d}_opacity_midslice_z.png")
						imageio.imwrite(opacity_path_z, opacity_img_z_uint8)
						color_img_z = np.clip(color_slice_z, 0, 1)
						color_img_z_uint8 = (color_img_z * 255).astype(np.uint8)
						color_path_z = os.path.join(vis_dir, f"frame_{t:03d}_color_midslice_z.png")
						imageio.imwrite(color_path_z, color_img_z_uint8)
						
						print(f"    Saved grid visualizations (X, Y, Z midslices):")
						print(f"      X: {opacity_path_x}, {color_path_x}")
						print(f"      Y: {opacity_path_y}, {color_path_y}")
						print(f"      Z: {opacity_path_z}, {color_path_z}")
					color_wrapped_gaussians = None
					override_color = None
					override_opacity = None
				else:
					print(f"    Warning: No velocity model found for frame 0, cannot use ray marching")
					color_wrapped_gaussians = gaussians
					override_color = None
					override_opacity = None
					opacity_grid = None
					color_grid = None
					heat_grid = None
			else:
				color_wrapped_gaussians = gaussians
				override_color = None
				override_opacity = None
				opacity_grid = None
				color_grid = None
				heat_grid = None
		else:
			vel_model_path = os.path.join(ckpt_dir, "frame_velocities", f"frame_{t-1:03d}_velocity.pth")
			if not os.path.exists(vel_model_path):
				current_window_dir = os.path.join(ckpt_dir, f"window_{window_start}_{window_end}")
				vel_pattern = os.path.join(current_window_dir, "ckpt", f"velrbf_frame_{t-1:03d}_ckpt_*.pth")
				vel_files = glob.glob(vel_pattern)
				if not vel_files:
					prev_window_dir = os.path.join(ckpt_dir, f"window_{t-1}_{t-1+window_size}")
					vel_pattern = os.path.join(prev_window_dir, "ckpt", f"velrbf_frame_{t-1:03d}_ckpt_*.pth")
					vel_files = glob.glob(vel_pattern)
				if vel_files:
					vel_model_path = max(vel_files, key=os.path.getctime)
			
			if vel_model_path and os.path.exists(vel_model_path):
				velocity_model = TiDFRBF.load(vel_model_path, device=device)
				print(f"    Loaded velocity model for color calculation: {vel_model_path}")
				
				if use_raymarching:
					print(f"    Computing opacity and color grids for ray marching...")
					opacity_grid, color_grid, heat_grid = compute_opacity_color_grids(
						gaussians, velocity_model, grid_shape, coord_trans, lengths_tensor, device,
						vel_color_scale=vel_color_scale, use_velocity_opacity=use_velocity_opacity,
						min_opacity=min_opacity, max_opacity=max_opacity, opacity_scale=opacity_scale,
						colormap=colormap, opacity_mode=opacity_mode,
						opacity_multiplier=opacity_multiplier
					)
					print(f"    Opacity grid: shape={opacity_grid.shape}, range=[{opacity_grid.min():.4f}, {opacity_grid.max():.4f}]")
					print(f"    Color grid: shape={color_grid.shape}, range=[{color_grid.min():.4f}, {color_grid.max():.4f}]")
					print(f"    Heat grid: shape={heat_grid.shape}, range=[{heat_grid.min():.4f}, {heat_grid.max():.4f}]")
					
					eval_savedir = os.path.join(savedir, "velocity_colored_reconstruction_evaluation")
					npz_dir = os.path.join(eval_savedir, "npz_grids")
					os.makedirs(npz_dir, exist_ok=True)
					
					nx, ny, nz = grid_shape
					color_grid_np = color_grid.reshape(nx, ny, nz, 3)
					opacity_grid_np = opacity_grid.reshape(nx, ny, nz, 1)
					heat_grid_np = heat_grid.reshape(nx, ny, nz, 1)
					npz_filename = os.path.join(npz_dir, f"frame_{t:03d}_color_opacity_heat.npz")
					np.savez(npz_filename, color=color_grid_np, opacity=opacity_grid_np, heat=heat_grid_np)
					print(f"    Saved color, opacity & heat grids to NPZ: {npz_filename}")
					
					if save_raymarching_extra_outputs:
						vtk_dir = os.path.join(eval_savedir, "vtk_opacity_grids")
						os.makedirs(vtk_dir, exist_ok=True)
						
						origin = [0.0, 0.0, 0.0]
						lengths_np = lengths_tensor.cpu().numpy()
						spacing = [
							lengths_np[0] / max(nx - 1, 1),
							lengths_np[1] / max(ny - 1, 1),
							lengths_np[2] / max(nz - 1, 1)
						]
						
						vtk_filename = os.path.join(vtk_dir, f"frame_{t:03d}_opacity.vti")
						save_vtk_image_data(opacity_grid, origin, spacing, vtk_filename, scalar_name="opacity")
						print(f"    Saved opacity grid to VTK format: {vtk_filename}")
						
						mid_x = nx // 2
						mid_y = ny // 2
						mid_z = nz // 2
						
						vis_dir = os.path.join(eval_savedir, "grid_visualizations")
						os.makedirs(vis_dir, exist_ok=True)
						
						opacity_slice_x = opacity_grid[mid_x, :, :]  # (ny, nz)
						color_slice_x = color_grid[mid_x, :, :, :]  # (ny, nz, 3)
						opacity_min_x, opacity_max_x = opacity_slice_x.min(), opacity_slice_x.max()
						if opacity_max_x > opacity_min_x:
							opacity_img_x = (opacity_slice_x - opacity_min_x) / (opacity_max_x - opacity_min_x)
						else:
							opacity_img_x = opacity_slice_x * 0
						opacity_img_x = np.clip(opacity_img_x, 0, 1)
						opacity_img_x_uint8 = (opacity_img_x * 255).astype(np.uint8)
						opacity_path_x = os.path.join(vis_dir, f"frame_{t:03d}_opacity_midslice_x.png")
						imageio.imwrite(opacity_path_x, opacity_img_x_uint8)
						color_img_x = np.clip(color_slice_x, 0, 1)
						color_img_x_uint8 = (color_img_x * 255).astype(np.uint8)
						color_path_x = os.path.join(vis_dir, f"frame_{t:03d}_color_midslice_x.png")
						imageio.imwrite(color_path_x, color_img_x_uint8)
						
						opacity_slice_y = opacity_grid[:, mid_y, :]  # (nx, nz)
						color_slice_y = color_grid[:, mid_y, :, :]  # (nx, nz, 3)
						opacity_min_y, opacity_max_y = opacity_slice_y.min(), opacity_slice_y.max()
						if opacity_max_y > opacity_min_y:
							opacity_img_y = (opacity_slice_y - opacity_min_y) / (opacity_max_y - opacity_min_y)
						else:
							opacity_img_y = opacity_slice_y * 0
						opacity_img_y = np.clip(opacity_img_y, 0, 1)
						opacity_img_y_uint8 = (opacity_img_y * 255).astype(np.uint8)
						opacity_path_y = os.path.join(vis_dir, f"frame_{t:03d}_opacity_midslice_y.png")
						imageio.imwrite(opacity_path_y, opacity_img_y_uint8)
						color_img_y = np.clip(color_slice_y, 0, 1)
						color_img_y_uint8 = (color_img_y * 255).astype(np.uint8)
						color_path_y = os.path.join(vis_dir, f"frame_{t:03d}_color_midslice_y.png")
						imageio.imwrite(color_path_y, color_img_y_uint8)
						
						opacity_slice_z = opacity_grid[:, :, mid_z]  # (nx, ny)
						color_slice_z = color_grid[:, :, mid_z, :]  # (nx, ny, 3)
						opacity_min_z, opacity_max_z = opacity_slice_z.min(), opacity_slice_z.max()
						if opacity_max_z > opacity_min_z:
							opacity_img_z = (opacity_slice_z - opacity_min_z) / (opacity_max_z - opacity_min_z)
						else:
							opacity_img_z = opacity_slice_z * 0
						opacity_img_z = np.clip(opacity_img_z, 0, 1)
						opacity_img_z_uint8 = (opacity_img_z * 255).astype(np.uint8)
						opacity_path_z = os.path.join(vis_dir, f"frame_{t:03d}_opacity_midslice_z.png")
						imageio.imwrite(opacity_path_z, opacity_img_z_uint8)
						color_img_z = np.clip(color_slice_z, 0, 1)
						color_img_z_uint8 = (color_img_z * 255).astype(np.uint8)
						color_path_z = os.path.join(vis_dir, f"frame_{t:03d}_color_midslice_z.png")
						imageio.imwrite(color_path_z, color_img_z_uint8)
						
						print(f"    Saved grid visualizations (X, Y, Z midslices):")
						print(f"      X: {opacity_path_x}, {color_path_x}")
						print(f"      Y: {opacity_path_y}, {color_path_y}")
						print(f"      Z: {opacity_path_z}, {color_path_z}")
					color_wrapped_gaussians = None
					override_color = None
					override_opacity = None
				else:
					xyz_world_current = gaussians.get_xyz.detach().clone()
					xyz_smoke_current = coord_trans.world2smoke(xyz_world_current)
					gaussian_points_smoke = xyz_smoke_current * lengths_tensor
					
					with torch.no_grad():
						velocities_at_points = interpolate_velocity_to_gaussian_points(
							velocity_model, gaussian_points_smoke.detach(), grid_shape, lengths_tensor, device
						).detach()
					
					with torch.no_grad():
						rgb_colors = velocity_to_rgb(
							velocities_at_points.detach(), scale=vel_color_scale, is3D=True, device=device, colormap=colormap
						).detach()
					
					override_opacity = None
					if use_velocity_opacity:
						with torch.no_grad():
							velocity_opacity_factor = velocity_to_opacity(
								velocities_at_points.detach(), 
								min_opacity=min_opacity, 
								max_opacity=max_opacity, 
								opacity_scale=opacity_scale,
								device=device,
								opacity_mode=opacity_mode
							).detach()
							original_opacity = gaussians.get_opacity.detach()  # (N, 1)
							override_opacity = (original_opacity * velocity_opacity_factor).detach()
						print(f"    Computed opacity from velocity magnitude (factor range: [{velocity_opacity_factor.min().item():.3f}, {velocity_opacity_factor.max().item():.3f}])")
						print(f"    Final opacity range: [{override_opacity.min().item():.3f}, {override_opacity.max().item():.3f}]")
					
					color_wrapped_gaussians = gaussians
					override_color = rgb_colors
					print(f"    Converted velocity to RGB color for {len(velocities_at_points)} points")
					opacity_grid = None
					color_grid = None
					heat_grid = None
			else:
				print(f"    Warning: Velocity model not found for color calculation, using original colors")
				color_wrapped_gaussians = gaussians
				override_color = None
				override_opacity = None
				opacity_grid = None
				color_grid = None
				heat_grid = None
		
		if should_render and (not use_raymarching or save_raymarching_extra_outputs):
			train_cameras_for_frame = frame_to_train_cameras.get(t, [])
			test_cameras_for_frame = frame_to_test_cameras.get(t, [])
			
			print(f"    Train cameras: {len(train_cameras_for_frame)}, Test cameras: {len(test_cameras_for_frame)}")
			
			total_cameras = len(train_cameras_for_frame) + len(test_cameras_for_frame)
			cam_pbar = tqdm(enumerate(train_cameras_for_frame), total=len(train_cameras_for_frame), desc=f"  Rendering train cameras (frame {t})", leave=False)
			for cam_enum_idx, (cam_idx, cam) in cam_pbar:
				with torch.no_grad():
					if use_raymarching and opacity_grid is not None and color_grid is not None:
						render_result, transmittance_result = ray_march_render(
							cam, opacity_grid, color_grid, grid_shape, coord_trans, lengths_tensor,
							background, device, near=None, far=None,
							N_samples=raymarching_samples, chunk=raymarching_chunk
						)  # (H, W, 3), (H, W)
						render_result = render_result.permute(2, 0, 1)  # (3, H, W)
					else:
						render_result = render(cam, color_wrapped_gaussians, pipe, background, stage="coarse", 
												override_color=override_color, override_opacity=override_opacity)["render"]
						transmittance_result = None
					gt_image = cam.original_image.cuda()
					
					render_tensor = render_result.detach().contiguous().unsqueeze(0)
					gt_tensor = gt_image.detach().contiguous().unsqueeze(0)
					
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
					
					render_np = render_result.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
					gt_np = gt_image.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
					render_img = (render_np * 255).astype(np.uint8)
					gt_img = (gt_np * 255).astype(np.uint8)
					
					train_camera_renders.append(render_img)
					train_camera_gts.append(gt_img)
					train_camera_info.append((t, cam_idx))
					
					if transmittance_result is not None:
						transmittance_np = transmittance_result.detach().clamp(0, 1).cpu().numpy()
						transmittance_img = (transmittance_np * 255).astype(np.uint8)
						train_camera_transmittance.append(transmittance_img)
					else:
						train_camera_transmittance.append(None)
			
			test_cam_pbar = tqdm(enumerate(test_cameras_for_frame), total=len(test_cameras_for_frame), desc=f"  Rendering test cameras (frame {t})", leave=False)
			for cam_enum_idx, (cam_idx, cam) in test_cam_pbar:
				with torch.no_grad():
					if use_raymarching and opacity_grid is not None and color_grid is not None:
						rays_o_temp, rays_d_temp = get_rays_from_camera(cam)
						rays_o_flat_temp = rays_o_temp.detach().reshape(-1, 3)
						smoke_corners_norm = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device)
						world_corners = coord_trans.smoke2world(smoke_corners_norm)
						camera_center = rays_o_flat_temp[0]
						distances_to_corners = torch.norm(world_corners - camera_center.unsqueeze(0), dim=-1)
						near_vis = max(0.1, distances_to_corners.min().item() * 0.5)
						far_vis = distances_to_corners.max().item() * 1.5
						
						if cam_enum_idx == 0:
							eval_savedir = os.path.join(savedir, "velocity_colored_reconstruction_evaluation")
							vis_dir = os.path.join(eval_savedir, "camera_rays_visualization")
							os.makedirs(vis_dir, exist_ok=True)
							vis_path = os.path.join(vis_dir, f"frame_{t:03d}_cam_{cam_idx:03d}_test_rays.png")
							visualize_camera_rays_and_grid(
								cam, opacity_grid, color_grid, grid_shape, coord_trans, lengths_tensor,
								near_vis, far_vis, vis_path, num_rays=50, num_samples_per_ray=100
							)
						
						render_result, transmittance_result = ray_march_render(
							cam, opacity_grid, color_grid, grid_shape, coord_trans, lengths_tensor,
							background, device, near=None, far=None,
							N_samples=raymarching_samples, chunk=raymarching_chunk
						)  # (H, W, 3), (H, W)
						render_result = render_result.permute(2, 0, 1)  # (3, H, W)
					else:
						render_result = render(cam, color_wrapped_gaussians, pipe, background, stage="coarse", 
												override_color=override_color, override_opacity=override_opacity)["render"]
						transmittance_result = None
					gt_image = cam.original_image.cuda()
					
					render_tensor = render_result.detach().contiguous().unsqueeze(0)
					gt_tensor = gt_image.detach().contiguous().unsqueeze(0)
					
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
					
					render_np = render_result.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
					gt_np = gt_image.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
					render_img = (render_np * 255).astype(np.uint8)
					gt_img = (gt_np * 255).astype(np.uint8)
					
					test_camera_renders.append(render_img)
					test_camera_gts.append(gt_img)
					test_camera_info.append((t, cam_idx))
					
					if transmittance_result is not None:
						transmittance_np = transmittance_result.detach().clamp(0, 1).cpu().numpy()
						transmittance_img = (transmittance_np * 255).astype(np.uint8)
						test_camera_transmittance.append(transmittance_img)
					else:
						test_camera_transmittance.append(None)
		else:
			print(f"    Skipping render (frame {t} not in range [{start_frame}, {end_frame}))")
		
	
	# ============================================================
	# ============================================================
	print(f"\n[Step 4] Computing metrics and saving results...")
	
	eval_savedir = os.path.join(savedir, "velocity_colored_reconstruction_evaluation")
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
	print(f"Velocity Colored Reconstruction Evaluation Results")
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
		f.write(f"Velocity Colored Reconstruction Evaluation Results\n")
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
	for idx, (render_img, gt_img, transmittance_img, (frame_idx, cam_idx)) in enumerate(zip(train_camera_renders, train_camera_gts, train_camera_transmittance, train_camera_info)):
		render_path = os.path.join(train_images_dir, f"cam_{cam_idx:03d}_frame_{frame_idx:03d}_render.png")
		gt_path = os.path.join(train_images_dir, f"cam_{cam_idx:03d}_frame_{frame_idx:03d}_gt.png")
		imageio.imwrite(render_path, render_img)
		imageio.imwrite(gt_path, gt_img)
		
		if transmittance_img is not None:
			transmittance_path = os.path.join(train_images_dir, f"cam_{cam_idx:03d}_frame_{frame_idx:03d}_transmittance.png")
			imageio.imwrite(transmittance_path, transmittance_img)
	
	print(f"  Saved {len(train_camera_renders)} train view images")
	if any(t is not None for t in train_camera_transmittance):
		print(f"  Saved {sum(1 for t in train_camera_transmittance if t is not None)} train view transmittance images")
	
	test_images_dir = os.path.join(eval_savedir, "images", "test")
	os.makedirs(test_images_dir, exist_ok=True)
	for idx, (render_img, gt_img, transmittance_img, (frame_idx, cam_idx)) in enumerate(zip(test_camera_renders, test_camera_gts, test_camera_transmittance, test_camera_info)):
		render_path = os.path.join(test_images_dir, f"cam_{cam_idx:03d}_frame_{frame_idx:03d}_render.png")
		gt_path = os.path.join(test_images_dir, f"cam_{cam_idx:03d}_frame_{frame_idx:03d}_gt.png")
		imageio.imwrite(render_path, render_img)
		imageio.imwrite(gt_path, gt_img)
		
		if transmittance_img is not None:
			transmittance_path = os.path.join(test_images_dir, f"cam_{cam_idx:03d}_frame_{frame_idx:03d}_transmittance.png")
			imageio.imwrite(transmittance_path, transmittance_img)
	
	print(f"  Saved {len(test_camera_renders)} test view images")
	if any(t is not None for t in test_camera_transmittance):
		print(f"  Saved {sum(1 for t in test_camera_transmittance if t is not None)} test view transmittance images")
	
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
