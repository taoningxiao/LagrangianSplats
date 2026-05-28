import numpy as np
import torch

def generate_gridpoints(shape: tuple, min_corner: np.ndarray, max_corner: np.ndarray, endpoint=True) -> np.ndarray:
	"""
	grid pointscoordinate
	
	Args:
		shape: shape，(Nx, Ny)  (Nx, Ny, Nz)
		       shape[0] = X direction
		       shape[1] = Y direction
		       shape[2] = Z direction (3D )
		min_corner: coordinate [x_min, y_min, z_min]
		max_corner: latestcoordinate [x_max, y_max, z_max]
		endpoint: contains
	
	Returns:
		points: shape (N, dim) ，where N = prod(shape)
		         C order ， (Z) 
		        reshape  shape + (dim,) ，arr[i, j, k, :] coordinate (X[i], Y[j], Z[k])
	"""
	X = np.linspace(min_corner[0], max_corner[0], shape[0], endpoint=endpoint)
	Y = np.linspace(min_corner[1], max_corner[1], shape[1], endpoint=endpoint)

	if len(shape) == 2:
		X, Y = np.meshgrid(X, Y, indexing='ij')
		points = np.stack((X.flatten(), Y.flatten()), axis=1)
	else:
		Z = np.linspace(min_corner[2], max_corner[2], shape[2], endpoint=endpoint)
		X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')
		points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)

	return points

def compute_gradients(grid: np.ndarray, dx: float) -> np.ndarray:
	"""
	scalar
	
	Args:
		grid: scalar，shape (Nx, Ny, Nz)  (Nx, Ny)
		      frame i frame i direction（x=0, y=1, z=2）
		dx: 
	
	Returns:
		，shape grid.shape + (dim,)
		grads[..., i] = ∂grid/∂(frameidirection)
	"""
	dim = len(grid.shape)
	grads = np.zeros(grid.shape + (dim,))

	for i in range(dim):
		grads[..., i] = np.gradient(grid, dx, axis=i)

	return grads

def compute_divergence(vel_field: np.ndarray, dx: float) -> np.ndarray:
	"""
	velocity field
	
	Args:
		vel_field: velocity field，shape (Nx, Ny, Nz, 3)  (Nx, Ny, 3)
		           where vel_field[..., 0] = vx, vel_field[..., 1] = vy, vel_field[..., 2] = vz
		           frame i frame i direction（x=0, y=1, z=2）
		dx: 
	
	Returns:
		，shape (Nx, Ny, Nz)  (Nx, Ny)
		div = ∂vx/∂x + ∂vy/∂y + ∂vz/∂z
	"""
	dim = len(vel_field.shape) - 1
	div = np.zeros(vel_field.shape[:-1])
	
	for i in range(dim):
		div += np.gradient(vel_field[..., i], dx, axis=i)
	
	return div

def compute_divergence_torch(vel_field: torch.Tensor, dx: float) -> torch.Tensor:
	"""
	velocity field (PyTorch，)
	
	Args:
		vel_field: velocity field，shape (Nx, Ny, Nz, 3)  (Nx, Ny, 3)
		           where vel_field[..., 0] = vx, vel_field[..., 1] = vy, vel_field[..., 2] = vz
		           frame i frame i direction（x=0, y=1, z=2）
		dx: 
	
	Returns:
		，shape (Nx, Ny, Nz)  (Nx, Ny)
		div = ∂vx/∂x + ∂vy/∂y + ∂vz/∂z
	"""
	import torch.nn.functional as F
	
	dim = len(vel_field.shape) - 1
	div = torch.zeros(vel_field.shape[:-1], device=vel_field.device, dtype=vel_field.dtype)
	
	for i in range(dim):
		grad = torch.gradient(vel_field[..., i], dim=i, spacing=dx)[0]
		div += grad
	
	return div

def perturb_points(points: np.ndarray, grid_shape: tuple, min_corner: np.ndarray, max_corner: np.ndarray, perturbation_scale: float = 0.1) -> np.ndarray:
	"""
	，
	
	Args:
		points: original，shape (N, 3)
		grid_shape: shape (Nx, Ny, Nz)
		min_corner: coordinate
		max_corner: latestcoordinate
		perturbation_scale: ，
	
	Returns:
		，shape (N, 3)
	"""
	dx = (max_corner[0] - min_corner[0]) / grid_shape[0]
	dy = (max_corner[1] - min_corner[1]) / grid_shape[1]
	dz = (max_corner[2] - min_corner[2]) / grid_shape[2]
	
	perturbation = np.random.normal(0, 1, points.shape)
	perturbation[:, 0] *= dx * perturbation_scale
	perturbation[:, 1] *= dy * perturbation_scale
	perturbation[:, 2] *= dz * perturbation_scale
	
	perturbed_points = points + perturbation
	
	perturbed_points = np.clip(perturbed_points, min_corner, max_corner)
	
	return perturbed_points

def trilinear_interpolation(field: np.ndarray, points: np.ndarray, min_corner: np.ndarray, max_corner: np.ndarray) -> np.ndarray:
	"""
	3Dinterpolation
	
	Args:
		field: 3D，shape (Nx, Ny, Nz)  (Nx, Ny, Nz, C)
		points: interpolation，shape (N, 3)，coordinate [min_corner, max_corner]
		min_corner: coordinate
		max_corner: latestcoordinate
	
	Returns:
		interpolation，shape (N,)  (N, C)
	"""
	grid_shape = field.shape[:3]
	points_scaled = (points - min_corner) / (max_corner - min_corner)
	points_scaled = points_scaled * (np.array(grid_shape) - 1)
	
	points_int = np.floor(points_scaled).astype(int)
	points_frac = points_scaled - points_int
	
	points_int = np.clip(points_int, 0, np.array(grid_shape) - 2)
	
	x0, y0, z0 = points_int[:, 0], points_int[:, 1], points_int[:, 2]
	x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
	
	if len(field.shape) == 3:
		c000 = field[x0, y0, z0]
		c001 = field[x0, y0, z1]
		c010 = field[x0, y1, z0]
		c011 = field[x0, y1, z1]
		c100 = field[x1, y0, z0]
		c101 = field[x1, y0, z1]
		c110 = field[x1, y1, z0]
		c111 = field[x1, y1, z1]
	else:
		c000 = field[x0, y0, z0]
		c001 = field[x0, y0, z1]
		c010 = field[x0, y1, z0]
		c011 = field[x0, y1, z1]
		c100 = field[x1, y0, z0]
		c101 = field[x1, y0, z1]
		c110 = field[x1, y1, z0]
		c111 = field[x1, y1, z1]
	
	xd = points_frac[:, 0:1] if len(field.shape) == 4 else points_frac[:, 0]
	yd = points_frac[:, 1:2] if len(field.shape) == 4 else points_frac[:, 1]
	zd = points_frac[:, 2:3] if len(field.shape) == 4 else points_frac[:, 2]
	
	c00 = c000 * (1 - xd) + c100 * xd
	c01 = c001 * (1 - xd) + c101 * xd
	c10 = c010 * (1 - xd) + c110 * xd
	c11 = c011 * (1 - xd) + c111 * xd
	
	c0 = c00 * (1 - yd) + c10 * yd
	c1 = c01 * (1 - yd) + c11 * yd
	
	c = c0 * (1 - zd) + c1 * zd
	
	return c