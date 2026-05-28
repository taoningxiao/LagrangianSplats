import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from velocity_common.coordinate_transform import CoordinateTransform


def get_rays(H, W, K, c2w):
	"""
	
	Args:
		H: 
		W: 
		K:  [3, 3] (numpy array or tensor)
		c2w: cameraworld coordinates [4, 4] (numpy array or tensor)
	Returns:
		rays_o: [H, W, 3] 
		rays_d: [H, W, 3] direction
	"""
	device = c2w.device if hasattr(c2w, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	if isinstance(K, np.ndarray):
		K = torch.from_numpy(K).float().to(device)
	else:
		K = K.float().to(device)
	
	if isinstance(c2w, np.ndarray):
		c2w = torch.from_numpy(c2w).float().to(device)
	else:
		c2w = c2w.float().to(device)
	
	i, j = torch.meshgrid(torch.linspace(0, int(W)-1, int(W), device=device), 
						 torch.linspace(0, int(H)-1, int(H), device=device), 
						 indexing='ij')
	i = i.t()
	j = j.t()
	dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
	rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)
	rays_o = c2w[:3,-1].expand(rays_d.shape)
	return rays_o, rays_d


def get_rays_from_camera(cam):
	"""
	 Camera object rays， GS 
	
	Args:
		cam: Camera object（scene.cameras.Camera）
	
	Returns:
		rays_o: [H, W, 3]  (World Space)
		rays_d: [H, W, 3] direction (World Space, )
	"""
	H = int(cam.image_height)
	W = int(cam.image_width)
	device = "cuda"
	
	fx = W / (2.0 * math.tan(cam.FoVx / 2.0))
	fy = H / (2.0 * math.tan(cam.FoVy / 2.0))
	cx = W / 2.0
	cy = H / 2.0
	
	K = torch.tensor([
		[fx, 0, cx],
		[0, fy, cy],
		[0, 0, 1]
	], dtype=torch.float32, device=device)
	
	w2c = cam.world_view_transform.detach().cpu().numpy() if torch.is_tensor(cam.world_view_transform) else np.array(cam.world_view_transform)
	w2c = w2c.T
	c2w = np.linalg.inv(w2c)
	c2w = torch.from_numpy(c2w).float().to(device)
	
	rays_o, rays_d = get_rays(H, W, K, c2w)

	print("foVx: ", cam.FoVx, "foVy: ", cam.FoVy)
	print("K: ", K)
	
	rays_d = -rays_d
	
	return rays_o, rays_d


def ray_inflow_region_intersection(rays_o, rays_d, coord_trans, lengths_tensor, device="cuda", inflow_region_min=None, inflow_region_max=None):
	"""
	 inflow region （ World Space in）
	
	Args:
		rays_o: [H, W, 3]  (World Space)
		rays_d: [H, W, 3] direction (World Space, )
		coord_trans: CoordinateTransform object
		lengths_tensor: [3] tensor，Sim Space 
		device: device
		inflow_region_min: bbox  [x_min, y_min, z_min]（ lengths_tensor ）
		inflow_region_max: bbox latest [x_max, y_max, z_max]（ lengths_tensor ）
	
	Returns:
		intersection_mask: [H, W] bool tensor，True  region 
	"""
	H, W = rays_o.shape[:2]
	
	if inflow_region_min is None:
		inflow_region_min = [0.0, 0.1, 0.0]
	if inflow_region_max is None:
		inflow_region_max = [1.0, 0.3, 1.0]
	
	if isinstance(inflow_region_min, (list, tuple)):
		inflow_region_min = torch.tensor(inflow_region_min, device=device, dtype=torch.float32)
	if isinstance(inflow_region_max, (list, tuple)):
		inflow_region_max = torch.tensor(inflow_region_max, device=device, dtype=torch.float32)
	
	aabb_min_sim = inflow_region_min * lengths_tensor  # [3]
	aabb_max_sim = inflow_region_max * lengths_tensor  # [3]
	
	x_coords = torch.stack([aabb_min_sim[0], aabb_max_sim[0]])
	y_coords = torch.stack([aabb_min_sim[1], aabb_max_sim[1]])
	z_coords = torch.stack([aabb_min_sim[2], aabb_max_sim[2]])
	xx, yy, zz = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
	bbox_vertices_sim = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)  # [8, 3]
	
	# Sim Space -> Smoke Space -> World Space
	bbox_vertices_smoke = bbox_vertices_sim / lengths_tensor.unsqueeze(0)  # [8, 3]
	bbox_vertices_world = coord_trans.smoke2world(bbox_vertices_smoke)  # [8, 3]
	print("mask world positions: ", bbox_vertices_world)
	print("mask smoke positions: ", bbox_vertices_smoke)
	
	aabb_min_world = torch.min(bbox_vertices_world, dim=0)[0]  # [3]
	aabb_max_world = torch.max(bbox_vertices_world, dim=0)[0]  # [3]
	
	rays_o_flat = rays_o.view(-1, 3)  # [H*W, 3]
	rays_d_flat = rays_d.view(-1, 3)  # [H*W, 3]
	
	inv_d = 1.0 / (rays_d_flat + 1e-8)
	t0 = (aabb_min_world.unsqueeze(0) - rays_o_flat) * inv_d  # [H*W, 3]
	t1 = (aabb_max_world.unsqueeze(0) - rays_o_flat) * inv_d  # [H*W, 3]
	
	t_min = torch.minimum(t0, t1)  # [H*W, 3]
	t_max = torch.maximum(t0, t1)  # [H*W, 3]
	
	t_enter = torch.max(t_min, dim=-1)[0]  # [H*W]
	t_exit = torch.min(t_max, dim=-1)[0]   # [H*W]
	
	intersects = (t_enter < t_exit) & (t_exit > 0)
	
	intersection_mask = intersects.view(H, W)
	
	return intersection_mask


def visualize_rays_and_inflow_region(rays_o, rays_d, coord_trans, lengths_tensor, 
                                     inflow_region_min, inflow_region_max, 
                                     save_path, num_rays_to_plot=100):
	"""
	visualization rays  inflow region  Sim Coord inthree-view plot
	
	Args:
		rays_o: [H, W, 3]  (World Space)
		rays_d: [H, W, 3] direction (World Space)
		coord_trans: CoordinateTransform object
		lengths_tensor: [3] tensor，Sim Space 
		inflow_region_min: [3] tensor，bbox （ lengths_tensor ）
		inflow_region_max: [3] tensor，bbox latest（ lengths_tensor ）
		save_path: save path
		num_rays_to_plot:  ray count（，）
	"""
	device = rays_o.device
	H, W = rays_o.shape[:2]
	
	if torch.is_tensor(rays_o):
		rays_o_np = rays_o.detach().cpu().numpy()
	else:
		rays_o_np = rays_o
	if torch.is_tensor(rays_d):
		rays_d_np = rays_d.detach().cpu().numpy()
	else:
		rays_d_np = rays_d
	if torch.is_tensor(lengths_tensor):
		lengths_tensor_np = lengths_tensor.detach().cpu().numpy()
	else:
		lengths_tensor_np = lengths_tensor
	if torch.is_tensor(inflow_region_min):
		inflow_region_min_np = inflow_region_min.detach().cpu().numpy()
	else:
		inflow_region_min_np = np.array(inflow_region_min)
	if torch.is_tensor(inflow_region_max):
		inflow_region_max_np = inflow_region_max.detach().cpu().numpy()
	else:
		inflow_region_max_np = np.array(inflow_region_max)
	
	rays_o_flat = rays_o_np.reshape(-1, 3)  # [H*W, 3]
	rays_d_flat = rays_d_np.reshape(-1, 3)  # [H*W, 3]
	
	# World -> Smoke -> Sim
	rays_o_smoke = coord_trans.world2smoke(torch.from_numpy(rays_o_flat).to(device)).detach().cpu().numpy()
	rays_end_world = rays_o_flat + rays_d_flat
	rays_end_smoke = coord_trans.world2smoke(torch.from_numpy(rays_end_world).to(device)).detach().cpu().numpy()
	rays_d_smoke = rays_end_smoke - rays_o_smoke
	rays_d_smoke_norm = np.linalg.norm(rays_d_smoke, axis=-1, keepdims=True)
	rays_d_smoke = rays_d_smoke / (rays_d_smoke_norm + 1e-8)
	
	rays_o_sim = rays_o_smoke * lengths_tensor_np  # [H*W, 3]
	rays_d_sim = rays_d_smoke * lengths_tensor_np  # [H*W, 3]
	rays_d_sim_norm = np.linalg.norm(rays_d_sim, axis=-1, keepdims=True)
	rays_d_sim = rays_d_sim / (rays_d_sim_norm + 1e-8)
	
	aabb_min_sim = inflow_region_min_np * lengths_tensor_np
	aabb_max_sim = inflow_region_max_np * lengths_tensor_np
	
	indices = np.random.choice(H * W, min(num_rays_to_plot, H * W), replace=False)
	rays_o_plot = rays_o_sim[indices]  # [N, 3]
	rays_d_plot = rays_d_sim[indices]  # [N, 3]
	
	ray_length = np.max(lengths_tensor_np) * 2.0
	rays_end_plot = rays_o_plot + rays_d_plot * ray_length
	
	fig = plt.figure(figsize=(18, 6))
	
	views = [
		{'title': 'Front View (X-Y plane)', 'x': 0, 'y': 1, 'xlabel': 'X (Sim)', 'ylabel': 'Y (Sim)'},
		{'title': 'Side View (Y-Z plane)', 'x': 1, 'y': 2, 'xlabel': 'Y (Sim)', 'ylabel': 'Z (Sim)'},
		{'title': 'Top View (X-Z plane)', 'x': 0, 'y': 2, 'xlabel': 'X (Sim)', 'ylabel': 'Z (Sim)'}
	]
	
	for view_idx, view in enumerate(views):
		ax = fig.add_subplot(1, 3, view_idx + 1)
		
		if view_idx == 0:
			bbox_rect = patches.Rectangle(
				(aabb_min_sim[0], aabb_min_sim[1]),
				aabb_max_sim[0] - aabb_min_sim[0],
				aabb_max_sim[1] - aabb_min_sim[1],
				linewidth=2, edgecolor='red', facecolor='red', alpha=0.2, label='Inflow Region'
			)
			ax.add_patch(bbox_rect)
		elif view_idx == 1:
			bbox_rect = patches.Rectangle(
				(aabb_min_sim[1], aabb_min_sim[2]),
				aabb_max_sim[1] - aabb_min_sim[1],
				aabb_max_sim[2] - aabb_min_sim[2],
				linewidth=2, edgecolor='red', facecolor='red', alpha=0.2, label='Inflow Region'
			)
			ax.add_patch(bbox_rect)
		else:
			bbox_rect = patches.Rectangle(
				(aabb_min_sim[0], aabb_min_sim[2]),
				aabb_max_sim[0] - aabb_min_sim[0],
				aabb_max_sim[2] - aabb_min_sim[2],
				linewidth=2, edgecolor='red', facecolor='red', alpha=0.2, label='Inflow Region'
			)
			ax.add_patch(bbox_rect)
		
		for i in range(len(rays_o_plot)):
			ax.plot(
				[rays_o_plot[i, view['x']], rays_end_plot[i, view['x']]],
				[rays_o_plot[i, view['y']], rays_end_plot[i, view['y']]],
				'b-', alpha=0.3, linewidth=0.5
			)
		
		ax.scatter(rays_o_plot[:, view['x']], rays_o_plot[:, view['y']], 
		          c='blue', s=10, alpha=0.6, label='Ray Origins' if view_idx == 0 else '')
		
		sim_bbox_rect = patches.Rectangle(
			(0, 0),
			lengths_tensor_np[view['x']],
			lengths_tensor_np[view['y']],
			linewidth=1, edgecolor='gray', facecolor='none', linestyle='--', label='Sim Space' if view_idx == 0 else ''
		)
		ax.add_patch(sim_bbox_rect)
		
		ax.set_xlabel(view['xlabel'], fontsize=12)
		ax.set_ylabel(view['ylabel'], fontsize=12)
		ax.set_title(view['title'], fontsize=12, fontweight='bold')
		ax.grid(True, alpha=0.3)
		ax.set_aspect('equal', adjustable='box')
		if view_idx == 0:
			ax.legend(loc='upper right', fontsize=10)
	
	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f"Rays and inflow region visualization saved to: {save_path}")

