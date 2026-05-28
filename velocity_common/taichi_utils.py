import torch
import taichi as ti

ti.init(arch=ti.cuda)


@ti.kernel
def scatter_grad_kernel(
	p_pos: ti.types.ndarray(),    # [N, 3] Normalized coordinates [0, 1]
	p_grad: ti.types.ndarray(),   # [N, 3] Gradient on particles
	g_grad: ti.types.ndarray(),   # [D, H, W, 3] Gradient buffer for grid
	D: int, H: int, W: int
):
	for i in range(p_pos.shape[0]):
		# Map [0, 1] to grid coordinates [0, W-1], [0, H-1], [0, D-1]
		# Input p_pos is assumed to be (x, y, z)
		# Grid shape is (D, H, W) -> (z, y, x)
		
		pos_x = p_pos[i, 0] * (W - 1)
		pos_y = p_pos[i, 1] * (H - 1)
		pos_z = p_pos[i, 2] * (D - 1)
		
		# Base coordinates (floor)
		x_f = ti.floor(pos_x)
		y_f = ti.floor(pos_y)
		z_f = ti.floor(pos_z)
		
		x_i = int(x_f)
		y_i = int(y_f)
		z_i = int(z_f)
		
		# Fractions
		fx = pos_x - x_f
		fy = pos_y - y_f
		fz = pos_z - z_f
		
		# 2x2x2 Splatting loop
		for dz in ti.static(range(2)):
			for dy in ti.static(range(2)):
				for dx in ti.static(range(2)):
					# Compute trilinear weight
					w_x = dx * fx + (1 - dx) * (1 - fx)
					w_y = dy * fy + (1 - dy) * (1 - fy)
					w_z = dz * fz + (1 - dz) * (1 - fz)
					weight = w_x * w_y * w_z
					
					# Boundary check
					idx_x = ti.max(0, ti.min(x_i + dx, W - 1))
					idx_y = ti.max(0, ti.min(y_i + dy, H - 1))
					idx_z = ti.max(0, ti.min(z_i + dz, D - 1))
					
					# Atomic Add Gradients to Grid
					for c in ti.static(range(3)):
						ti.atomic_add(g_grad[idx_z, idx_y, idx_x, c], weight * p_grad[i, c])


def scatter_grad_to_grid_taichi(grad_particles, particle_coords_norm, grid_shape):
	"""
	Python wrapper for Taichi scatter kernel.
	Args:
		grad_particles: [N, 3] Torch tensor, gradients on particles.
		particle_coords_norm: [N, 3] Torch tensor, coordinates in [0, 1].
		grid_shape: (D, H, W) tuple.
	Returns:
		grad_grid: [1, 3, D, H, W] Torch tensor (compatible with PyTorch gradient format).
	"""
	D, H, W = grid_shape
	# Ensure contiguous and on CUDA
	if not grad_particles.is_contiguous(): grad_particles = grad_particles.contiguous()
	if not particle_coords_norm.is_contiguous(): particle_coords_norm = particle_coords_norm.contiguous()
	
	# Alloc zero-filled buffer for grid gradients
	# Note: TiDFRBF / PyTorch convention might expect (3, D, H, W) or (D, H, W, 3).
	# grid_sample expects (N, C, D, H, W).
	# We will compute in (D, H, W, 3) for easier Taichi indexing, then permute.
	grad_grid_taichi = torch.zeros((D, H, W, 3), device=grad_particles.device, dtype=torch.float32)
	
	scatter_grad_kernel(particle_coords_norm, grad_particles, grad_grid_taichi, D, H, W)
	
	# Permute to [1, 3, D, H, W] to match what grid_sample's backward would produce for the 'input'
	return grad_grid_taichi.permute(3, 0, 1, 2).unsqueeze(0)

