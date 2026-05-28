import os
import torch
import torch.nn as nn
import numpy as np
import taichi as ti
from typing import Union, List, Tuple
from velocity_common.rbf import WendlandC4, WendlandC2, Poly6

ti.init(arch=ti.cuda, default_fp=ti.f32)

class TiDFRBF(nn.Module):
    """
     Taichi  DFRBF 
    use PyTorch autograd.Function 
    
    """
    
    def __init__(self, radial_func, init_centers, h, device='cuda', use_spatial_hash=True):
        """
         TiDFRBF 
        
        Args:
            radial_func: radial functioninstance
            init_centers: centers，shape (n_kernels, dim)
            h: ，scalar
            device: device
            use_spatial_hash: with spatial-hash acceleration，True
        """
        super(TiDFRBF, self).__init__()
        
        self.radial_func = radial_func
        self.dim = init_centers.shape[1]
        self.n_kernels = len(init_centers)
        self.device = device
        self.use_spatial_hash = use_spatial_hash
        self.name = f'TiDFRBF_{type(radial_func).__name__}_{"Hash" if use_spatial_hash else "NoHash"}'
        
        if isinstance(h, (int, float)):
            init_radii = np.ones(self.n_kernels) * h
        elif isinstance(h, (list, tuple, np.ndarray)):
            init_radii = np.array(h)
        elif torch.is_tensor(h):
            init_radii = h.detach().cpu().numpy()
        else:
            raise TypeError(f"Unsupported type for h: {type(h)}")
        
        init_weights = np.zeros((self.n_kernels, self.dim))
        
        self.centers = nn.Parameter(torch.tensor(init_centers, dtype=torch.float32, device=device))
        self.radii = nn.Parameter(torch.tensor(init_radii, dtype=torch.float32, device=device))
        self.weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32, device=device))
        
        self.min_radius = 1e-6
        
        self.radial_func_type_map = {
            "WendlandC4": 0,
            "WendlandC2": 1, 
            "Poly6": 2
        }
        self.radial_func_type = self.radial_func_type_map.get(type(radial_func).__name__, 0)
        
        self._hash_table_initialized = False
        self._last_centers_hash = None
        self._last_radii_hash = None
        
        print(f'[*] TiDFRBF initialized: {self.n_kernels} kernels, {self.dim}D, radial function: {type(radial_func).__name__}')
    
    def _needs_hash_update(self):
        """check"""
        if not self._hash_table_initialized:
            return True
        
        current_centers_hash = hash(self.centers.data.cpu().numpy().tobytes())
        if current_centers_hash != self._last_centers_hash:
            return True
        
        current_radii_hash = hash(self.radii.data.cpu().numpy().tobytes())
        if current_radii_hash != self._last_radii_hash:
            return True
        
        return False
    
    def _update_spatial_hash(self):
        """"""
        
        centers_np = self.centers.detach().cpu().numpy()
        self.min_coords = np.min(centers_np, axis=0)
        self.max_coords = np.max(centers_np, axis=0)
        
        self.grid_size = self.radii.max().item()
        
        self.grid_dims = np.ceil((self.max_coords - self.min_coords) / self.grid_size).astype(np.int32)
        
        kernel_grid_indices = np.floor((centers_np - self.min_coords) / self.grid_size).astype(np.int32)
        for i in range(len(kernel_grid_indices)):
            for j in range(len(kernel_grid_indices[i])):
                kernel_grid_indices[i, j] = max(0, min(kernel_grid_indices[i, j], self.grid_dims[j] - 1))
        
        grid_to_kernels = {}
        for k, grid_idx in enumerate(kernel_grid_indices):
            grid_key = tuple(grid_idx)
            if grid_key not in grid_to_kernels:
                grid_to_kernels[grid_key] = []
            grid_to_kernels[grid_key].append(k)
        
        self.total_grids = np.prod(self.grid_dims)
        
        grid_linear_indices = {}
        linear_idx = 0
        if self.dim == 2:
            for i in range(self.grid_dims[0]):
                for j in range(self.grid_dims[1]):
                    grid_key = (i, j)
                    grid_linear_indices[grid_key] = linear_idx
                    linear_idx += 1
        else:  # 3D case
            for i in range(self.grid_dims[0]):
                for j in range(self.grid_dims[1]):
                    for k in range(self.grid_dims[2]):
                        grid_key = (i, j, k)
                        grid_linear_indices[grid_key] = linear_idx
                        linear_idx += 1
        
        grid_kernel_indices = []
        grid_kernel_nbs = []
        
        for linear_idx in range(self.total_grids):
            if self.dim == 2:
                grid_x = linear_idx // self.grid_dims[1]
                grid_y = linear_idx % self.grid_dims[1]
                grid_key = (grid_x, grid_y)
            else:  # 3D case
                grid_x = linear_idx // (self.grid_dims[1] * self.grid_dims[2])
                grid_y = (linear_idx % (self.grid_dims[1] * self.grid_dims[2])) // self.grid_dims[2]
                grid_z = linear_idx % self.grid_dims[2]
                grid_key = (grid_x, grid_y, grid_z)
            
            grid_kernel_indices.append(len(grid_kernel_nbs))
            
            if grid_key in grid_to_kernels:
                grid_kernel_nbs.extend(grid_to_kernels[grid_key])
        
        grid_kernel_indices.append(len(grid_kernel_nbs))
        
        self.grid_kernel_indices = np.array(grid_kernel_indices, dtype=np.int32)
        self.grid_kernel_nbs = np.array(grid_kernel_nbs, dtype=np.int32)
        
        self._hash_table_initialized = True
        self._last_centers_hash = hash(self.centers.data.cpu().numpy().tobytes())
        self._last_radii_hash = hash(self.radii.data.cpu().numpy().tobytes())
        
    
    def forward_hash(self, query_points):
        """
        forward pass（with spatial-hash acceleration）
        
        Args:
            query_points: torch.Tensor, shape=(N, dim), query pointscoordinate
            
        Returns:
            vels: torch.Tensor, shape=(N, dim), velocities at the query points
        """
        if self._needs_hash_update():
            self._update_spatial_hash()
        
        query_points = query_points.to(self.device)
        
        constrained_radii = torch.clamp(self.radii, min=self.min_radius)
        
        if self.dim == 2:
            return DFRBFOperator2D.apply(query_points, self.centers, constrained_radii, self.weights,
                                       self.grid_kernel_indices, self.grid_kernel_nbs,
                                       self.min_coords, self.grid_dims, self.grid_size,
                                       self.radial_func_type)
        else:
            return DFRBFOperator3D.apply(query_points, self.centers, constrained_radii, self.weights,
                                       self.grid_kernel_indices, self.grid_kernel_nbs,
                                       self.min_coords, self.grid_dims, self.grid_size,
                                       self.radial_func_type)
    
    def forward(self, query_points):
        """
        forward pass（without spatial hashing，directly evaluating all kernels）
        used forkernel
        
        Args:
            query_points: torch.Tensor, shape=(N, dim), query pointscoordinate
            
        Returns:
            vels: torch.Tensor, shape=(N, dim), velocities at the query points
        """
        query_points = query_points.to(self.device)
        
        constrained_radii = torch.clamp(self.radii, min=self.min_radius)
        
        if self.dim == 2:
            return DFRBFOperator2DNoHash.apply(query_points, self.centers, constrained_radii, self.weights,
                                             self.radial_func_type)
        else:
            return DFRBFOperator3DNoHash.apply(query_points, self.centers, constrained_radii, self.weights,
                                             self.radial_func_type)
    
    def divergence(self, points):
        """Compute divergence（DFRBF ）"""
        return torch.zeros(len(points), dtype=points.dtype, device=points.device)
    
    def vorticity(self, query_points):
        """
        Compute vorticity（curl of the velocity field）
        
        Args:
            query_points: torch.Tensor, shape=(N, dim), query pointscoordinate
            
        Returns:
            vorticity: torch.Tensor, shape=(N, 3) for 3D or (N,) for 2D, vorticity at the query points
        """
        query_points = query_points.to(self.device)
        
        constrained_radii = torch.clamp(self.radii, min=self.min_radius)
        
        if self.dim == 2:
            return DFRBFVorticityOperator2D.apply(query_points, self.centers, constrained_radii, self.weights,
                                                self.radial_func_type)
        else:
            return DFRBFVorticityOperator3D.apply(query_points, self.centers, constrained_radii, self.weights,
                                                self.radial_func_type)
    
    def save(self, filepath):
        """Save the model to disk."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        state_dict = {
            'model_state_dict': self.state_dict(),
            'dim': self.dim,
            'n_kernels': self.n_kernels,
            'radial_func_name': type(self.radial_func).__name__,
            'init_centers': self.centers.data.clone(),
            'init_radii': self.radii.data.clone(),
            'init_weights': self.weights.data.clone()
        }
        
        torch.save(state_dict, filepath)
        print(f"TiDFRBF model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, device='cuda'):
        """Load the model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
        
        state_dict = torch.load(filepath, map_location=device)
        radial_func_name = state_dict['radial_func_name']
        radial_func = cls._get_radial_func_by_name(radial_func_name)
        
        model = cls(
            radial_func=radial_func,
            init_centers=state_dict['init_centers'],
            h=state_dict['init_radii'],
            device=device
        )
        
        model.load_state_dict(state_dict['model_state_dict'])
        print(f"TiDFRBF model loaded from {filepath} with radial function: {radial_func_name}")
        return model
    
    def union(self, other):
        """
        Merge two TiDFRBF instances.
        
        Args:
            other: TiDFRBF object，object TiDFRBF instance
            
        Returns:
            TiDFRBF: merged TiDFRBF instance
            
        Raises:
            ValueError: objectconfiguration is incompatible
        """
        if not isinstance(other, TiDFRBF):
            raise ValueError("Can only merge with another TiDFRBF instance")
        
        if self.dim != other.dim:
            raise ValueError(f"dimension mismatch: current instance is {self.dim}D，other instance is {other.dim}D")
        
        if type(self.radial_func).__name__ != type(other.radial_func).__name__:
            raise ValueError(f"radial functiontype mismatch: current instance is {type(self.radial_func).__name__}，"
                           f"other instance is {type(other.radial_func).__name__}")
        
        if self.device != other.device:
            raise ValueError(f"device: object {self.device}，object {other.device}")
        
        if self.use_spatial_hash != other.use_spatial_hash:
            raise ValueError(f"spatial-hash setting mismatch: current instance is {self.use_spatial_hash}，"
                           f"other instance is {other.use_spatial_hash}")
        
        combined_centers = torch.cat([self.centers.data, other.centers.data], dim=0)
        combined_radii = torch.cat([self.radii.data, other.radii.data], dim=0)
        combined_weights = torch.cat([self.weights.data, other.weights.data], dim=0)
        
        combined_model = TiDFRBF(
            radial_func=self.radial_func,
            init_centers=combined_centers.cpu().numpy(),
            h=combined_radii.cpu().numpy(),
            device=self.device,
            use_spatial_hash=self.use_spatial_hash
        )
        
        with torch.no_grad():
            combined_model.weights.data = combined_weights
        
        print(f'[*] TiDFRBF merge complete: {self.n_kernels} + {other.n_kernels} = {combined_model.n_kernels} kernels')
        
        return combined_model
    
    @staticmethod
    def _get_radial_func_by_name(radial_func_name):
        """Create a radial-function instance by name."""
        radial_func_map = {
            'Poly6': Poly6,
            'WendlandC2': WendlandC2,
            'WendlandC4': WendlandC4,
        }
        
        if radial_func_name not in radial_func_map:
            raise ValueError(f"Unknown radial function name: {radial_func_name}. "
                           f"Supported functions: {list(radial_func_map.keys())}")
        
        return radial_func_map[radial_func_name]()

@ti.func
def dfrbf_kernel_2d(x, r:ti.f32, radial_func_type):
    """2D DFRBF kernel function - only supportsWendlandC4"""
    t = (1 - r) ** 4
    c1 = 30 * t
    c2 = (1 + 4 * r - 5 * r * r) * t
    
    m1 = (c2 * 1 - c1 * r * r) * ti.Matrix.identity(ti.f32, 2)
    m2 = c1 * x.outer_product(x)
    return m1 + m2

@ti.func
def dfrbf_kernel_3d(x, r:ti.f32, radial_func_type):
    # t = ti.f32(0.0)
    # c1 = ti.f32(0.0)
    # c2 = ti.f32(0.0)
    
    # if radial_func_type == 0:  # WendlandC4
    #     t = (1 - r) ** 4
    #     c1 = 30 * t
    #     c2 = (1 + 4 * r - 5 * r * r) * t
    # elif radial_func_type == 1:  # WendlandC2
    #     t = (1 - r) ** 3
    #     c1 = 12 * t
    #     c2 = (1 + 3 * r - 12 * r * r) * t
    # elif radial_func_type == 2:  # Poly6
    #     t = (1 - r * r) ** 3
    #     c1 = 24 * t
    #     c2 = 6 * t * t
    # else:
    #     t = (1 - r) ** 4
    #     c1 = 30 * t
    #     c2 = (1 + 4 * r - 5 * r * r) * t
    
    # m1 = (c2 * (3 - 1) - c1 * r * r) * ti.Matrix.identity(ti.f32, 3)
    # m2 = c1 * x.outer_product(x)
    # return t * (m1 + m2)
    t = (1 - r) ** 4
    c1 = 30 * t
    c2 = (1 + 4 * r - 5 * r * r) * t
    
    m1 = (c2 * 2 - c1 * r * r) * ti.Matrix.identity(ti.f32, 3)
    m2 = c1 * x.outer_product(x)
    return m1 + m2

@ti.kernel
def _forward_2d_batch_hash_kernel(query_points: ti.types.ndarray(dtype=ti.math.vec2),
                                  vels: ti.types.ndarray(dtype=ti.math.vec2),
                                  centers: ti.types.ndarray(dtype=ti.math.vec2),
                                  radii: ti.types.ndarray(dtype=ti.f32),
                                  weights: ti.types.ndarray(dtype=ti.math.vec2),
                                  grid_kernel_indices: ti.types.ndarray(dtype=ti.i32),
                                  grid_kernel_nbs: ti.types.ndarray(dtype=ti.i32),
                                  min_coords: ti.types.ndarray(dtype=ti.f32),
                                  grid_dims: ti.types.ndarray(dtype=ti.i32),
                                  grid_size: ti.f32, radial_func_type: ti.i32):
    """with spatial-hash acceleration2D DFRBFforward pass"""
    for q in range(query_points.shape[0]):
        point = query_points[q]
        
        grid_idx = ti.Vector([0, 0], dt=ti.i32)
        
        dx = point[0] - min_coords[0]
        grid_x_float = dx / grid_size
        grid_x_floor = ti.floor(grid_x_float)
        grid_x_int = ti.int32(grid_x_floor)
        grid_x_clamped = ti.max(0, ti.min(grid_x_int, grid_dims[0] - 1))
        grid_idx[0] = grid_x_clamped
        
        dy = point[1] - min_coords[1]
        grid_y_float = dy / grid_size
        grid_y_floor = ti.floor(grid_y_float)
        grid_y_int = ti.int32(grid_y_floor)
        grid_y_clamped = ti.max(0, ti.min(grid_y_int, grid_dims[1] - 1))
        grid_idx[1] = grid_y_clamped
        
        for di in range(-1, 2):
            for dj in range(-1, 2):
                neighbor_grid = ti.Vector([grid_idx[0] + di, grid_idx[1] + dj], dt=ti.i32)
                
                if (neighbor_grid[0] >= 0 and neighbor_grid[0] < grid_dims[0] and
                    neighbor_grid[1] >= 0 and neighbor_grid[1] < grid_dims[1]):
                    
                    neighbor_linear_idx = neighbor_grid[0] * grid_dims[1] + neighbor_grid[1]
                    
                    for i in range(grid_kernel_indices[neighbor_linear_idx], 
                                  grid_kernel_indices[neighbor_linear_idx + 1]):
                        k = grid_kernel_nbs[i]
                        center = centers[k]
                        radius = radii[k]
                        
                        x = (point - center) / radius
                        r = x.norm()
                        
                        if r < 1.0:
                            t = ti.f32(0.0)
                            c1 = ti.f32(0.0)
                            c2 = ti.f32(0.0)
                            
                            if radial_func_type == 0:  # WendlandC4
                                t = (1 - r) ** 4
                                c1 = 30 * t
                                c2 = (1 + 4 * r - 5 * r * r) * t
                            elif radial_func_type == 1:  # WendlandC2
                                t = (1 - r) ** 3
                                c1 = 12 * t
                                c2 = (1 + 3 * r - 12 * r * r) * t
                            elif radial_func_type == 2:  # Poly6
                                t = (1 - r * r) ** 3
                                c1 = 24 * t
                                c2 = 6 * t * t
                            else:
                                t = (1 - r) ** 4
                                c1 = 30 * t
                                c2 = (1 + 4 * r - 5 * r * r) * t
                            
                            weight = weights[k]
                            res1 = (c2 * (2 - 1) - c1 * r * r) * weight
                            res2 = c1 * (x * weight).sum() * x
                            vels[q] += res1 + res2

@ti.kernel
def _forward_3d_batch_hash_kernel(query_points: ti.types.ndarray(dtype=ti.math.vec3),
                                  vels: ti.types.ndarray(dtype=ti.math.vec3),
                                  centers: ti.types.ndarray(dtype=ti.math.vec3),
                                  radii: ti.types.ndarray(dtype=ti.f32),
                                  weights: ti.types.ndarray(dtype=ti.math.vec3),
                                  grid_kernel_indices: ti.types.ndarray(dtype=ti.i32),
                                  grid_kernel_nbs: ti.types.ndarray(dtype=ti.i32),
                                  min_coords: ti.types.ndarray(dtype=ti.f32),
                                  grid_dims: ti.types.ndarray(dtype=ti.i32),
                                  grid_size: ti.f32, radial_func_type: ti.i32):
    """with spatial-hash acceleration3D DFRBFforward pass"""
    for q in range(query_points.shape[0]):
        point = query_points[q]
        
        grid_idx = ti.Vector([0, 0, 0], dt=ti.i32)
        
        dx = point[0] - min_coords[0]
        grid_x_float = dx / grid_size
        grid_x_floor = ti.floor(grid_x_float)
        grid_x_int = ti.int32(grid_x_floor)
        grid_x_clamped = ti.max(0, ti.min(grid_x_int, grid_dims[0] - 1))
        grid_idx[0] = grid_x_clamped
        
        dy = point[1] - min_coords[1]
        grid_y_float = dy / grid_size
        grid_y_floor = ti.floor(grid_y_float)
        grid_y_int = ti.int32(grid_y_floor)
        grid_y_clamped = ti.max(0, ti.min(grid_y_int, grid_dims[1] - 1))
        grid_idx[1] = grid_y_clamped
        
        dz = point[2] - min_coords[2]
        grid_z_float = dz / grid_size
        grid_z_floor = ti.floor(grid_z_float)
        grid_z_int = ti.int32(grid_z_floor)
        grid_z_clamped = ti.max(0, ti.min(grid_z_int, grid_dims[2] - 1))
        grid_idx[2] = grid_z_clamped
        
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    neighbor_grid = ti.Vector([grid_idx[0] + di, grid_idx[1] + dj, grid_idx[2] + dk], dt=ti.i32)
                    
                    if (neighbor_grid[0] >= 0 and neighbor_grid[0] < grid_dims[0] and
                        neighbor_grid[1] >= 0 and neighbor_grid[1] < grid_dims[1] and
                        neighbor_grid[2] >= 0 and neighbor_grid[2] < grid_dims[2]):
                        
                        neighbor_linear_idx = (neighbor_grid[0] * grid_dims[1] * grid_dims[2] + 
                                              neighbor_grid[1] * grid_dims[2] + 
                                              neighbor_grid[2])
                        
                        for i in range(grid_kernel_indices[neighbor_linear_idx], 
                                      grid_kernel_indices[neighbor_linear_idx + 1]):
                            k = grid_kernel_nbs[i]
                            center = centers[k]
                            radius = radii[k]
                            
                            x = (point - center) / radius
                            r = x.norm()
                            
                            if r < 1.0:
                                t = ti.f32(0.0)
                                c1 = ti.f32(0.0)
                                c2 = ti.f32(0.0)
                                
                                if radial_func_type == 0:  # WendlandC4
                                    t = (1 - r) ** 4
                                    c1 = 30 * t
                                    c2 = (1 + 4 * r - 5 * r * r) * t
                                elif radial_func_type == 1:  # WendlandC2
                                    t = (1 - r) ** 3
                                    c1 = 12 * t
                                    c2 = (1 + 3 * r - 12 * r * r) * t
                                elif radial_func_type == 2:  # Poly6
                                    t = (1 - r * r) ** 3
                                    c1 = 24 * t
                                    c2 = 6 * t * t
                                else:
                                    t = (1 - r) ** 4
                                    c1 = 30 * t
                                    c2 = (1 + 4 * r - 5 * r * r) * t
                                
                                weight = weights[k]
                                res1 = (c2 * (3 - 1) - c1 * r * r) * weight
                                res2 = c1 * (x * weight).sum() * x
                                vels[q] += res1 + res2

@ti.kernel
def _backward_2d_batch_hash_kernel(query_points: ti.types.ndarray(dtype=ti.math.vec2),
                                   grad_output: ti.types.ndarray(dtype=ti.math.vec2),
                                   centers: ti.types.ndarray(dtype=ti.math.vec2),
                                   radii: ti.types.ndarray(dtype=ti.f32),
                                   weights: ti.types.ndarray(dtype=ti.math.vec2),
                                   grad_centers: ti.types.ndarray(dtype=ti.math.vec2),
                                   grad_radii: ti.types.ndarray(dtype=ti.f32),
                                   grad_weights: ti.types.ndarray(dtype=ti.math.vec2),
                                   grid_kernel_indices: ti.types.ndarray(dtype=ti.i32),
                                   grid_kernel_nbs: ti.types.ndarray(dtype=ti.i32),
                                   min_coords: ti.types.ndarray(dtype=ti.f32),
                                   grid_dims: ti.types.ndarray(dtype=ti.i32),
                                   grid_size: ti.f32, radial_func_type: ti.i32):
    """with spatial-hash acceleration2D DFRBFbackward pass - """
    for q in range(query_points.shape[0]):
        point = query_points[q]
        grad_vel = grad_output[q]
        
        grid_idx = ti.Vector([0, 0], dt=ti.i32)
        
        dx = point[0] - min_coords[0]
        grid_x_float = dx / grid_size
        grid_x_floor = ti.floor(grid_x_float)
        grid_x_int = ti.int32(grid_x_floor)
        grid_x_clamped = ti.max(0, ti.min(grid_x_int, grid_dims[0] - 1))
        grid_idx[0] = grid_x_clamped
        
        dy = point[1] - min_coords[1]
        grid_y_float = dy / grid_size
        grid_y_floor = ti.floor(grid_y_float)
        grid_y_int = ti.int32(grid_y_floor)
        grid_y_clamped = ti.max(0, ti.min(grid_y_int, grid_dims[1] - 1))
        grid_idx[1] = grid_y_clamped
        
        for di in range(-1, 2):
            for dj in range(-1, 2):
                neighbor_grid = ti.Vector([grid_idx[0] + di, grid_idx[1] + dj], dt=ti.i32)
                
                if (neighbor_grid[0] >= 0 and neighbor_grid[0] < grid_dims[0] and
                    neighbor_grid[1] >= 0 and neighbor_grid[1] < grid_dims[1]):
                    
                    neighbor_linear_idx = neighbor_grid[0] * grid_dims[1] + neighbor_grid[1]
                    
                    for i in range(grid_kernel_indices[neighbor_linear_idx], 
                                  grid_kernel_indices[neighbor_linear_idx + 1]):
                        k = grid_kernel_nbs[i]
                        center = centers[k]
                        radius = radii[k]
                        
                        x = (point - center) / radius
                        r = x.norm() + 1e-8
                        
                        if r < 1.0:
                            t = ti.f32(0.0)
                            c1 = ti.f32(0.0)
                            c2 = ti.f32(0.0)
                            
                            # if radial_func_type == 0:  # WendlandC4
                            t = (1 - r) ** 4
                            c1 = 30 * t
                            c2 = (1 + 4 * r - 5 * r * r) * t
                            # else:
                            #     raise ValueError(f"Invalid radial function type: {radial_func_type}")
                            
                            weight = weights[k]
                            grad_weight = (c2 * (2 - 1) - c1 * r * r) * grad_vel + c1 * (x * grad_vel).sum() * x
                            grad_weights[k] += grad_weight
                            
                            # grad_center = ∂L/∂v * ∂v/∂x * ∂x/∂c
                            # ∂x/∂c = -1/radius * I
                            
                            d = 2  # 2D
                            C = (1 - r) ** 3
                            A = -20 * (d - 1) * r + 10 * (d + 5) * (3 * r - 1) * r
                            B = 120
                            
                            term1_vec = C * (A * weight - B * (x * weight).sum() * x)
                            term1 = term1_vec.outer_product(x / r)
                            
                            xw_outer = x.outer_product(weight)
                            xw_dot = (x * weight).sum()
                            term2 = 30 * (1 - r) ** 4 * (xw_outer + xw_dot * ti.Matrix.identity(ti.f32, 2))
                            
                            dv_dx = term1 + term2
                            
                            dx_dc = -ti.Matrix.identity(ti.f32, 2) / radius
                            
                            grad_center = (grad_vel @ dv_dx) @ dx_dc
                            grad_centers[k] += grad_center
                            
                            # grad_r = ∂L/∂v * ∂v/∂x * ∂x/∂r
                            # ∂v/∂x = C(AI - Bxx^T)w x^T/r + 30(1-r)^4(xw^T + (x·w)I)
                            # ∂x/∂r = -x/r
                            
                            dx_dr = -x / radius
                            
                            grad_radius = (grad_vel @ dv_dx) @ dx_dr
                            grad_radii[k] += grad_radius

@ti.kernel
def _backward_3d_batch_hash_kernel(query_points: ti.types.ndarray(dtype=ti.math.vec3),
                                   grad_output: ti.types.ndarray(dtype=ti.math.vec3),
                                   centers: ti.types.ndarray(dtype=ti.math.vec3),
                                   radii: ti.types.ndarray(dtype=ti.f32),
                                   weights: ti.types.ndarray(dtype=ti.math.vec3),
                                   grad_centers: ti.types.ndarray(dtype=ti.math.vec3),
                                   grad_radii: ti.types.ndarray(dtype=ti.f32),
                                   grad_weights: ti.types.ndarray(dtype=ti.math.vec3),
                                   grid_kernel_indices: ti.types.ndarray(dtype=ti.i32),
                                   grid_kernel_nbs: ti.types.ndarray(dtype=ti.i32),
                                   min_coords: ti.types.ndarray(dtype=ti.f32),
                                   grid_dims: ti.types.ndarray(dtype=ti.i32),
                                   grid_size: ti.f32, radial_func_type: ti.i32):
    """with spatial-hash acceleration3D DFRBFbackward pass - """
    for q in range(query_points.shape[0]):
        point = query_points[q]
        grad_vel = grad_output[q]
        
        grid_idx = ti.Vector([0, 0, 0], dt=ti.i32)
        
        dx = point[0] - min_coords[0]
        grid_x_float = dx / grid_size
        grid_x_floor = ti.floor(grid_x_float)
        grid_x_int = ti.int32(grid_x_floor)
        grid_x_clamped = ti.max(0, ti.min(grid_x_int, grid_dims[0] - 1))
        grid_idx[0] = grid_x_clamped
        
        dy = point[1] - min_coords[1]
        grid_y_float = dy / grid_size
        grid_y_floor = ti.floor(grid_y_float)
        grid_y_int = ti.int32(grid_y_floor)
        grid_y_clamped = ti.max(0, ti.min(grid_y_int, grid_dims[1] - 1))
        grid_idx[1] = grid_y_clamped
        
        dz = point[2] - min_coords[2]
        grid_z_float = dz / grid_size
        grid_z_floor = ti.floor(grid_z_float)
        grid_z_int = ti.int32(grid_z_floor)
        grid_z_clamped = ti.max(0, ti.min(grid_z_int, grid_dims[2] - 1))
        grid_idx[2] = grid_z_clamped
        
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    neighbor_grid = ti.Vector([grid_idx[0] + di, grid_idx[1] + dj, grid_idx[2] + dk], dt=ti.i32)
                    
                    if (neighbor_grid[0] >= 0 and neighbor_grid[0] < grid_dims[0] and
                        neighbor_grid[1] >= 0 and neighbor_grid[1] < grid_dims[1] and
                        neighbor_grid[2] >= 0 and neighbor_grid[2] < grid_dims[2]):
                        
                        neighbor_linear_idx = (neighbor_grid[0] * grid_dims[1] * grid_dims[2] + 
                                              neighbor_grid[1] * grid_dims[2] + 
                                              neighbor_grid[2])
                        
                        for i in range(grid_kernel_indices[neighbor_linear_idx], 
                                      grid_kernel_indices[neighbor_linear_idx + 1]):
                            k = grid_kernel_nbs[i]
                            center = centers[k]
                            radius = radii[k]
                            
                            x = (point - center) / radius
                            r = x.norm() + 1e-8
                            
                            if r < 1.0:
                                t = ti.f32(0.0)
                                c1 = ti.f32(0.0)
                                c2 = ti.f32(0.0)
                                

                                t = (1 - r) ** 4
                                c1 = 30 * t
                                c2 = (1 + 4 * r - 5 * r * r) * t
                                
                                weight = weights[k]
                                grad_weight = (c2 * (3 - 1) - c1 * r * r) * grad_vel + c1 * (x * grad_vel).sum() * x
                                grad_weights[k] += grad_weight
                                
                                # grad_center = ∂L/∂v * ∂v/∂x * ∂x/∂c
                                # ∂x/∂c = -1/radius * I
                                
                                d = 3  # 3D
                                C = (1 - r) ** 3
                                A = -20 * (d - 1) * r + 10 * (d + 5) * (3 * r - 1) * r
                                B = 120
                                
                                term1_vec = C * (A * weight - B * (x * weight).sum() * x)
                                term1 = term1_vec.outer_product(x / r)
                                
                                xw_outer = x.outer_product(weight)
                                xw_dot = (x * weight).sum()
                                term2 = 30 * (1 - r) ** 4 * (xw_outer + xw_dot * ti.Matrix.identity(ti.f32, 3))
                                
                                dv_dx = term1 + term2
                                
                                dx_dc = -ti.Matrix.identity(ti.f32, 3) / radius
                                
                                grad_center = (grad_vel @ dv_dx) @ dx_dc
                                grad_centers[k] += grad_center
                                
                                # grad_r = ∂L/∂v * ∂v/∂x * ∂x/∂r
                                # ∂v/∂x = C(AI - Bxx^T)w x^T/r + 30(1-r)^4(xw^T + (x·w)I)
                                # ∂x/∂r = -x/r
                                
                                dx_dr = -x / radius
                                
                                grad_radius = (grad_vel @ dv_dx) @ dx_dr
                                grad_radii[k] += grad_radius

@ti.kernel
def _vorticity_2d_batch_no_hash_kernel(query_points: ti.types.ndarray(dtype=ti.math.vec2),
                                     vorticities: ti.types.ndarray(dtype=ti.f32),
                                     centers: ti.types.ndarray(dtype=ti.math.vec2),
                                     radii: ti.types.ndarray(dtype=ti.f32),
                                     weights: ti.types.ndarray(dtype=ti.math.vec2),
                                     radial_func_type: ti.i32):
    """2D DFRBFvorticity - 2Dvorticityscalar"""
    for q in range(query_points.shape[0]):
        query_point = query_points[q]
        vorticity = ti.f32(0.0)
        
        for k in range(centers.shape[0]):
            center = centers[k]
            radius = radii[k]
            weight = weights[k]
            
            x = (query_point - center) / radius
            r = x.norm()
            
            if r < 1e-6:
                continue
            
            if r < 1.0:
                if radial_func_type == 0:  # WendlandC4
                    h_r = 120 * r * (1 - r)**3 * (2 * r - 1)
                    cross_product = x[0] * weight[1] - x[1] * weight[0]
                    vorticity += h_r * cross_product / (r * radius)
        
        vorticities[q] = vorticity

@ti.kernel
def _vorticity_3d_batch_no_hash_kernel(query_points: ti.types.ndarray(dtype=ti.math.vec3),
                                     vorticities: ti.types.ndarray(dtype=ti.math.vec3),
                                     centers: ti.types.ndarray(dtype=ti.math.vec3),
                                     radii: ti.types.ndarray(dtype=ti.f32),
                                     weights: ti.types.ndarray(dtype=ti.math.vec3),
                                     radial_func_type: ti.i32):
    """3D DFRBFvorticity - 3Dvorticityvector"""
    for q in range(query_points.shape[0]):
        query_point = query_points[q]
        vorticity = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        
        for k in range(centers.shape[0]):
            center = centers[k]
            radius = radii[k]
            weight = weights[k]
            
            x = (query_point - center) / radius
            r = x.norm()
            
            if r < 1e-6:
                continue
            
            if r < 1.0:
                if radial_func_type == 0:  # WendlandC4
                    h_r = 30 * r * (1 - r) ** 3 * (9 * r - 5)
                    cross_product = x.cross(weight)
                    vorticity += h_r * cross_product / (r * radius)
        
        vorticities[q] = vorticity

@ti.kernel
def _backward_vorticity_2d_batch_no_hash_kernel(query_points: ti.types.ndarray(dtype=ti.math.vec2),
                                              grad_output: ti.types.ndarray(dtype=ti.f32),
                                              centers: ti.types.ndarray(dtype=ti.math.vec2),
                                              radii: ti.types.ndarray(dtype=ti.f32),
                                              weights: ti.types.ndarray(dtype=ti.math.vec2),
                                              grad_centers: ti.types.ndarray(dtype=ti.math.vec2),
                                              grad_radii: ti.types.ndarray(dtype=ti.f32),
                                              grad_weights: ti.types.ndarray(dtype=ti.math.vec2),
                                              grad_query_points: ti.types.ndarray(dtype=ti.math.vec2),
                                              radial_func_type: ti.i32):
    """2D DFRBFvorticitybackward pass"""
    for q in range(query_points.shape[0]):
        query_point = query_points[q]
        grad_vorticity = grad_output[q]
        
        for k in range(centers.shape[0]):
            center = centers[k]
            radius = radii[k]
            weight = weights[k]
            
            x = (query_point - center) / radius
            r = x.norm()
            
            if r < 1e-6:
                continue
            
            if r < 1.0:
                if radial_func_type == 0:  # WendlandC4
                    h_r = 120 * r * (1 - r) ** 3 * (2 * r - 1)
                    
                    # ∂(x×w)/∂w[0] = -x[1], ∂(x×w)/∂w[1] = x[0]
                    coeff = h_r / (r * radius) * grad_vorticity
                    grad_weight = ti.Vector([0.0, 0.0], dt=ti.f32)
                    grad_weight[0] = -coeff * x[1]
                    grad_weight[1] = coeff * x[0]
                    grad_weights[k] += grad_weight
                    
                    d = 2
                    
                    # term1 = 120*(1-r)^2[-(d+6)r+(d+3)r](x×weight)*1/r * x
                    cross_product = x[0] * weight[1] - x[1] * weight[0]
                    term1_coeff = 120 * (1 - r) ** 2 * (-(d + 6) * r + (d + 3)) * cross_product / r
                    term1 = term1_coeff * x
                    
                    # term2 = -30*(1-r)^3*[(d+6)r-(d+2)][w]_x
                    term2_coeff = -30 * (1 - r) ** 3 * ((d + 6) * r - (d + 2))
                    term2 = ti.Vector([-weight[1], weight[0]], dt=ti.f32) * term2_coeff
                    
                    domega_dx = (term1 + term2) / radius
                    
                    # ∂x/∂x_c = -1/radius * I
                    # grad_vorticity * domega_dx * (-1/radius * I) = -grad_vorticity * domega_dx / radius
                    grad_center = -grad_vorticity * domega_dx / radius
                    grad_centers[k] += grad_center
                    
                    grad_query_point = grad_vorticity * domega_dx / radius
                    grad_query_points[q] += grad_query_point
                    
                    radius_term1 = -grad_vorticity * domega_dx.dot(x) / radius
                    
                    cross_product = x[0] * weight[1] - x[1] * weight[0]
                    radius_term2 = -30 * (1 - r) ** 3 / (radius * radius) * ((d + 6) * r - (d + 2)) * cross_product * grad_vorticity
                    
                    grad_radius = radius_term1 + radius_term2
                    grad_radii[k] += grad_radius

@ti.kernel
def _backward_vorticity_3d_batch_no_hash_kernel(query_points: ti.types.ndarray(dtype=ti.math.vec3),
                                              grad_output: ti.types.ndarray(dtype=ti.math.vec3),
                                              centers: ti.types.ndarray(dtype=ti.math.vec3),
                                              radii: ti.types.ndarray(dtype=ti.f32),
                                              weights: ti.types.ndarray(dtype=ti.math.vec3),
                                              grad_centers: ti.types.ndarray(dtype=ti.math.vec3),
                                              grad_radii: ti.types.ndarray(dtype=ti.f32),
                                              grad_weights: ti.types.ndarray(dtype=ti.math.vec3),
                                              grad_query_points: ti.types.ndarray(dtype=ti.math.vec3),
                                              radial_func_type: ti.i32):
    """3D DFRBFvorticitybackward pass"""
    for q in range(query_points.shape[0]):
        query_point = query_points[q]
        grad_vorticity = grad_output[q]
        
        for k in range(centers.shape[0]):
            center = centers[k]
            radius = radii[k]
            weight = weights[k]
            
            x = (query_point - center) / radius
            r = x.norm()
            
            if r < 1e-6:
                continue
            
            if r < 1.0:
                if radial_func_type == 0:  # WendlandC4
                    h_r = 30 * r * (1 - r) ** 3 * (9 * r - 5)
                    
                    coeff = h_r / (r * radius)
                    grad_weight = grad_vorticity.cross(x) * coeff
                    grad_weights[k] += grad_weight
                    
                    d = 3
                    
                    # term1 = 120*(1-r)^2[-(d+6)r+(d+3)r](x×weight)*1/r * x^T
                    cross_product_vec = x.cross(weight)
                    term1_coeff = 120 * (1 - r) ** 2 * (-(d + 6) * r + (d + 3)) / r
                    term1 = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dt=ti.f32)
                    for i in range(3):
                        for j in range(3):
                            term1[i, j] = term1_coeff * cross_product_vec[i] * x[j]
                    
                    # term2 = -30*(1-r)^3*[(d+6)r-(d+2)][w]_x
                    term2_coeff = -30 * (1 - r) ** 3 * ((d + 6) * r - (d + 2))
                    term2 = ti.Matrix([[0.0, -weight[2], weight[1]], 
                                     [weight[2], 0.0, -weight[0]], 
                                     [-weight[1], weight[0], 0.0]], dt=ti.f32) * term2_coeff
                    
                    domega_dx = (term1 + term2) / radius
                    
                    # ∂x/∂x_c = -1/radius * I
                    # grad_vorticity * domega_dx * (-1/radius * I) = -grad_vorticity * domega_dx / radius
                    grad_center = -(grad_vorticity @ domega_dx) / radius
                    grad_centers[k] += grad_center
                    
                    grad_query_point = (grad_vorticity @ domega_dx) / radius
                    grad_query_points[q] += grad_query_point
                    
                    radius_term1 = -(grad_vorticity @ domega_dx).dot(x) / radius
                    
                    cross_product_vec = x.cross(weight)
                    term2_coeff = -30 * (1 - r) ** 3 / (radius * radius) * ((d + 6) * r - (d + 2))
                    radius_term2 = grad_vorticity.dot(term2_coeff * cross_product_vec)
                    
                    grad_radius = radius_term1 + radius_term2
                    grad_radii[k] += grad_radius

@ti.kernel
def _forward_2d_batch_no_hash_kernel(query_points: ti.types.ndarray(dtype=ti.math.vec2),
                                    vels: ti.types.ndarray(dtype=ti.math.vec2),
                                    centers: ti.types.ndarray(dtype=ti.math.vec2),
                                    radii: ti.types.ndarray(dtype=ti.f32),
                                    weights: ti.types.ndarray(dtype=ti.math.vec2),
                                    radial_func_type: ti.i32):
    """without spatial hashing2D DFRBFforward pass - directly evaluating all kernels"""
    for q in range(query_points.shape[0]):
        query_point = query_points[q]
        vel = ti.Vector([0.0, 0.0], dt=ti.f32)
        
        for k in range(centers.shape[0]):
            center = centers[k]
            radius = radii[k]
            weight = weights[k]
            
            x = (query_point - center) / radius
            r = x.norm()
            
            if r < 1.0:
                M = dfrbf_kernel_2d(x, r, radial_func_type)
                vel += M @ weight
        
        vels[q] = vel

@ti.kernel
def _forward_3d_batch_no_hash_kernel(query_points: ti.types.ndarray(dtype=ti.math.vec3),
                                    vels: ti.types.ndarray(dtype=ti.math.vec3),
                                    centers: ti.types.ndarray(dtype=ti.math.vec3),
                                    radii: ti.types.ndarray(dtype=ti.f32),
                                    weights: ti.types.ndarray(dtype=ti.math.vec3),
                                    radial_func_type: ti.i32):
    """without spatial hashing3D DFRBFforward pass - directly evaluating all kernels"""
    for q in range(query_points.shape[0]):
        query_point = query_points[q]
        vel = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        
        for k in range(centers.shape[0]):
            center = centers[k]
            radius = radii[k]
            weight = weights[k]
            
            x = (query_point - center) / radius
            r = x.norm()
            
            if r < 1.0:
                M = dfrbf_kernel_3d(x, r, radial_func_type)
                vel += M @ weight
        
        vels[q] = vel

@ti.kernel
def _backward_2d_batch_no_hash_kernel(query_points: ti.types.ndarray(dtype=ti.math.vec2),
                                     grad_output: ti.types.ndarray(dtype=ti.math.vec2),
                                     centers: ti.types.ndarray(dtype=ti.math.vec2),
                                     radii: ti.types.ndarray(dtype=ti.f32),
                                     weights: ti.types.ndarray(dtype=ti.math.vec2),
                                     grad_centers: ti.types.ndarray(dtype=ti.math.vec2),
                                     grad_radii: ti.types.ndarray(dtype=ti.f32),
                                     grad_weights: ti.types.ndarray(dtype=ti.math.vec2),
                                     grad_query_points: ti.types.ndarray(dtype=ti.math.vec2),
                                     radial_func_type: ti.i32):
    """without spatial hashing2D DFRBFbackward pass - use"""
    for q in range(query_points.shape[0]):
        query_point = query_points[q]
        grad_vel = grad_output[q]
        
        for k in range(centers.shape[0]):
            center = centers[k]
            radius = radii[k]
            weight = weights[k]
            
            x = (query_point - center) / radius
            r = x.norm() + 1e-8
            
            if r < 1.0:
                if radial_func_type == 0:
                    # t = (1 - r) ** 4
                    # c1 = 30 * t
                    # c2 = (1 + 4 * r - 5 * r * r) * t
                    
                    # grad_weight = (c2 * (2 - 1) - c1 * r * r) * grad_vel + c1 * (x * grad_vel).sum() * x
                    # grad_weights[k] += grad_weight
                    M = dfrbf_kernel_2d(x, r, radial_func_type)
                    grad_weight = M @ grad_vel
                    grad_weights[k] += grad_weight
                    
                    # grad_center = ∂L/∂v * ∂v/∂x * ∂x/∂c
                    # ∂x/∂c = -1/radius * I
                    
                    d = 2  # 2D
                    C = (1 - r) ** 3
                    A = 30 * r * ((d + 5) * r - (d + 1))
                    B = 120
                    
                    term1_vec = C * (A * weight - B * (x * weight).sum() * x)
                    term1 = term1_vec.outer_product(x / r)
                    
                    xw_outer = x.outer_product(weight)
                    xw_dot = (x * weight).sum()
                    term2 = 30 * (1 - r) ** 4 * (xw_outer + xw_dot * ti.Matrix.identity(ti.f32, 2))
                    
                    dv_dx = term1 + term2
                    
                    grad_center = -(dv_dx.transpose() @ grad_vel) / radius
                    grad_centers[k] += grad_center
                    
                    grad_query_point = (grad_vel @ dv_dx) / radius
                    grad_query_points[q] += grad_query_point
                    
                    # grad_radius = ∂L/∂v * ∂v/∂r * ∂r/∂radius
                    # ∂r/∂radius = -r/radius
                    # ∂v/∂r = ∂v/∂x * ∂x/∂r = ∂v/∂x * (-x/radius)
                    dx_dr = -x / radius
                    
                    grad_radius = (grad_vel @ dv_dx) @ dx_dr
                    grad_radii[k] += grad_radius

@ti.kernel
def _backward_3d_batch_no_hash_kernel(query_points: ti.types.ndarray(dtype=ti.math.vec3),
                                     grad_output: ti.types.ndarray(dtype=ti.math.vec3),
                                     centers: ti.types.ndarray(dtype=ti.math.vec3),
                                     radii: ti.types.ndarray(dtype=ti.f32),
                                     weights: ti.types.ndarray(dtype=ti.math.vec3),
                                     grad_centers: ti.types.ndarray(dtype=ti.math.vec3),
                                     grad_radii: ti.types.ndarray(dtype=ti.f32),
                                     grad_weights: ti.types.ndarray(dtype=ti.math.vec3),
                                     grad_query_points: ti.types.ndarray(dtype=ti.math.vec3),
                                     radial_func_type: ti.i32):
    """without spatial hashing3D DFRBFbackward pass - use"""
    for q in range(query_points.shape[0]):
        query_point = query_points[q]
        grad_vel = grad_output[q]
        
        for k in range(centers.shape[0]):
            center = centers[k]
            radius = radii[k]
            weight = weights[k]
            
            x = (query_point - center) / radius
            r = x.norm() + 1e-8
            
            if r < 1.0:
                if radial_func_type == 0:
                    # t = (1 - r) ** 4
                    # c1 = 30 * t
                    # c2 = (1 + 4 * r - 5 * r * r) * t
                    
                    # grad_weight = (c2 * (3 - 1) - c1 * r * r) * grad_vel + c1 * (x * grad_vel).sum() * x
                    # grad_weights[k] += grad_weight
                    M = dfrbf_kernel_3d(x, r, radial_func_type)
                    grad_weight = M.transpose() @ grad_vel
                    grad_weights[k] += grad_weight
                    
                    # grad_center = ∂L/∂v * ∂v/∂x * ∂x/∂c
                    # ∂x/∂c = -1/radius * I
                    
                    d = 3  # 3D
                    C = (1 - r) ** 3
                    A = 30 * r * ((d + 5) * r - (d + 1))
                    B = 120
                    
                    term1_vec = C * (A * weight - B * (x * weight).sum() * x)
                    term1 = term1_vec.outer_product(x / r)
                    
                    xw_outer = x.outer_product(weight)
                    xw_dot = (x * weight).sum()
                    term2 = 30 * (1 - r) ** 4 * (xw_outer + xw_dot * ti.Matrix.identity(ti.f32, 3))
                    
                    dv_dx = term1 + term2
                    
                    grad_center = -(dv_dx.transpose() @ grad_vel) / radius
                    grad_centers[k] += grad_center
                    
                    grad_query_point = (grad_vel @ dv_dx) / radius
                    grad_query_points[q] += grad_query_point
                    
                    # grad_radius = ∂L/∂v * ∂v/∂r * ∂r/∂radius
                    # ∂r/∂radius = -r/radius
                    # ∂v/∂r = ∂v/∂x * ∂x/∂r = ∂v/∂x * (-x/radius)
                    dx_dr = -x / radius
                    
                    grad_radius = (grad_vel @ dv_dx) @ dx_dr
                    grad_radii[k] += grad_radius

class DFRBFOperator2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_points, centers, radii, weights, 
                grid_kernel_indices, grid_kernel_nbs, min_coords, grid_dims, 
                grid_size, radial_func_type):
        """forward pass"""
        ctx.save_for_backward(query_points, centers, radii, weights)
        ctx.grid_kernel_indices = grid_kernel_indices
        ctx.grid_kernel_nbs = grid_kernel_nbs
        ctx.min_coords = min_coords
        ctx.grid_dims = grid_dims
        ctx.grid_size = grid_size
        ctx.radial_func_type = radial_func_type
        
        N = query_points.shape[0]
        vels = torch.zeros((N, 2), dtype=torch.float32, device=query_points.device)
        
        query_points = query_points.contiguous()
        vels = vels.contiguous()
        
        _forward_2d_batch_hash_kernel(query_points, vels, centers, radii, weights,
                                     grid_kernel_indices, grid_kernel_nbs,
                                     min_coords, grid_dims, grid_size, radial_func_type)
        return vels

    @staticmethod
    def backward(ctx, grad_output):
        """backward pass - use"""
        query_points, centers, radii, weights = ctx.saved_tensors
        
        grad_centers = torch.zeros_like(centers)
        grad_radii = torch.zeros_like(radii)
        grad_weights = torch.zeros_like(weights)
        
        query_points = query_points.contiguous()
        grad_output = grad_output.contiguous()
        grad_centers = grad_centers.contiguous()
        grad_radii = grad_radii.contiguous()
        grad_weights = grad_weights.contiguous()
        
        _backward_2d_batch_hash_kernel(query_points, grad_output, centers, radii, weights,
                                      grad_centers, grad_radii, grad_weights,
                                      ctx.grid_kernel_indices, ctx.grid_kernel_nbs,
                                      ctx.min_coords, ctx.grid_dims, ctx.grid_size, ctx.radial_func_type)
        
        min_radius = 1e-6
        should_block = (radii < min_radius) & (grad_radii > 0)
        grad_radii = torch.where(should_block, torch.zeros_like(grad_radii), grad_radii)
        
        return None, grad_centers, grad_radii, grad_weights, None

class DFRBFVorticityOperator2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_points, centers, radii, weights, radial_func_type):
        """forward pass"""
        ctx.save_for_backward(query_points, centers, radii, weights)
        ctx.radial_func_type = radial_func_type
        
        N = query_points.shape[0]
        vorticities = torch.zeros((N,), dtype=torch.float32, device=query_points.device)
        
        query_points = query_points.contiguous()
        vorticities = vorticities.contiguous()
        
        _vorticity_2d_batch_no_hash_kernel(query_points, vorticities, centers, radii, weights, radial_func_type)
        return vorticities

    @staticmethod
    def backward(ctx, grad_output):
        """backward pass"""
        query_points, centers, radii, weights = ctx.saved_tensors
        radial_func_type = ctx.radial_func_type
        
        grad_centers = torch.zeros_like(centers)
        grad_radii = torch.zeros_like(radii)
        grad_weights = torch.zeros_like(weights)
        grad_query_points = torch.zeros_like(query_points)
        
        query_points = query_points.contiguous()
        grad_output = grad_output.contiguous()
        grad_centers = grad_centers.contiguous()
        grad_radii = grad_radii.contiguous()
        grad_weights = grad_weights.contiguous()
        grad_query_points = grad_query_points.contiguous()
        
        _backward_vorticity_2d_batch_no_hash_kernel(query_points, grad_output, centers, radii, weights,
                                                  grad_centers, grad_radii, grad_weights, grad_query_points, radial_func_type)
        
        min_radius = 1e-6
        should_block = (radii < min_radius) & (grad_radii > 0)
        grad_radii = torch.where(should_block, torch.zeros_like(grad_radii), grad_radii)
        
        return grad_query_points, grad_centers, grad_radii, grad_weights, None

class DFRBFVorticityOperator3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_points, centers, radii, weights, radial_func_type):
        """forward pass"""
        ctx.save_for_backward(query_points, centers, radii, weights)
        ctx.radial_func_type = radial_func_type
        
        N = query_points.shape[0]
        vorticities = torch.zeros((N, 3), dtype=torch.float32, device=query_points.device)
        
        query_points = query_points.contiguous()
        vorticities = vorticities.contiguous()
        
        _vorticity_3d_batch_no_hash_kernel(query_points, vorticities, centers, radii, weights, radial_func_type)
        return vorticities

    @staticmethod
    def backward(ctx, grad_output):
        """backward pass"""
        query_points, centers, radii, weights = ctx.saved_tensors
        radial_func_type = ctx.radial_func_type
        
        grad_centers = torch.zeros_like(centers)
        grad_radii = torch.zeros_like(radii)
        grad_weights = torch.zeros_like(weights)
        grad_query_points = torch.zeros_like(query_points)
        
        query_points = query_points.contiguous()
        grad_output = grad_output.contiguous()
        grad_centers = grad_centers.contiguous()
        grad_radii = grad_radii.contiguous()
        grad_weights = grad_weights.contiguous()
        grad_query_points = grad_query_points.contiguous()
        
        _backward_vorticity_3d_batch_no_hash_kernel(query_points, grad_output, centers, radii, weights,
                                                  grad_centers, grad_radii, grad_weights, grad_query_points, radial_func_type)
        
        min_radius = 1e-6
        should_block = (radii < min_radius) & (grad_radii > 0)
        grad_radii = torch.where(should_block, torch.zeros_like(grad_radii), grad_radii)
        
        return grad_query_points, grad_centers, grad_radii, grad_weights, None, None, None, None, None, None

class DFRBFOperator3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_points, centers, radii, weights, 
                grid_kernel_indices, grid_kernel_nbs, min_coords, grid_dims, 
                grid_size, radial_func_type):
        """forward pass"""
        ctx.save_for_backward(query_points, centers, radii, weights)
        ctx.grid_kernel_indices = grid_kernel_indices
        ctx.grid_kernel_nbs = grid_kernel_nbs
        ctx.min_coords = min_coords
        ctx.grid_dims = grid_dims
        ctx.grid_size = grid_size
        ctx.radial_func_type = radial_func_type
        
        N = query_points.shape[0]
        vels = torch.zeros((N, 3), dtype=torch.float32, device=query_points.device)
        
        query_points = query_points.contiguous()
        vels = vels.contiguous()
        
        _forward_3d_batch_hash_kernel(query_points, vels, centers, radii, weights,
                                     grid_kernel_indices, grid_kernel_nbs,
                                     min_coords, grid_dims, grid_size, radial_func_type)
        return vels

    @staticmethod
    def backward(ctx, grad_output):
        """backward pass - use"""
        query_points, centers, radii, weights = ctx.saved_tensors
        
        grad_centers = torch.zeros_like(centers)
        grad_radii = torch.zeros_like(radii)
        grad_weights = torch.zeros_like(weights)
        
        query_points = query_points.contiguous()
        grad_output = grad_output.contiguous()
        grad_centers = grad_centers.contiguous()
        grad_radii = grad_radii.contiguous()
        grad_weights = grad_weights.contiguous()
        
        _backward_3d_batch_hash_kernel(query_points, grad_output, centers, radii, weights,
                                      grad_centers, grad_radii, grad_weights,
                                      ctx.grid_kernel_indices, ctx.grid_kernel_nbs,
                                      ctx.min_coords, ctx.grid_dims, ctx.grid_size, ctx.radial_func_type)
        
        min_radius = 1e-6
        should_block = (radii < min_radius) & (grad_radii > 0)
        grad_radii = torch.where(should_block, torch.zeros_like(grad_radii), grad_radii)
        
        return None, grad_centers, grad_radii, grad_weights, None

class DFRBFOperator2DNoHash(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_points, centers, radii, weights, radial_func_type):
        """forward pass"""
        ctx.save_for_backward(query_points, centers, radii, weights)
        ctx.radial_func_type = radial_func_type
        
        N = query_points.shape[0]
        vels = torch.zeros((N, 2), dtype=torch.float32, device=query_points.device)
        
        query_points = query_points.contiguous()
        vels = vels.contiguous()
        
        _forward_2d_batch_no_hash_kernel(query_points, vels, centers, radii, weights, radial_func_type)
        return vels

    @staticmethod
    def backward(ctx, grad_output):
        """backward pass"""
        query_points, centers, radii, weights = ctx.saved_tensors
        radial_func_type = ctx.radial_func_type
        
        N = query_points.shape[0]
        n_kernels = centers.shape[0]
        
        grad_centers = torch.zeros_like(centers)
        grad_radii = torch.zeros_like(radii)
        grad_weights = torch.zeros_like(weights)
        grad_query_points = torch.zeros_like(query_points)
        
        query_points = query_points.contiguous()
        grad_output = grad_output.contiguous()
        grad_centers = grad_centers.contiguous()
        grad_radii = grad_radii.contiguous()
        grad_weights = grad_weights.contiguous()
        grad_query_points = grad_query_points.contiguous()
        
        _backward_2d_batch_no_hash_kernel(query_points, grad_output, centers, radii, weights,
                                        grad_centers, grad_radii, grad_weights, grad_query_points, radial_func_type)
        
        min_radius = 1e-6
        should_block = (radii < min_radius) & (grad_radii > 0)
        grad_radii = torch.where(should_block, torch.zeros_like(grad_radii), grad_radii)
        
        return grad_query_points, grad_centers, grad_radii, grad_weights, None

class DFRBFOperator3DNoHash(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_points, centers, radii, weights, radial_func_type):
        """forward pass"""
        ctx.save_for_backward(query_points, centers, radii, weights)
        ctx.radial_func_type = radial_func_type
        
        N = query_points.shape[0]
        vels = torch.zeros((N, 3), dtype=torch.float32, device=query_points.device)
        
        query_points = query_points.contiguous()
        vels = vels.contiguous()
        
        _forward_3d_batch_no_hash_kernel(query_points, vels, centers, radii, weights, radial_func_type)
        return vels

    @staticmethod
    def backward(ctx, grad_output):
        """backward pass"""
        query_points, centers, radii, weights = ctx.saved_tensors
        radial_func_type = ctx.radial_func_type
        
        N = query_points.shape[0]
        n_kernels = centers.shape[0]
        
        grad_centers = torch.zeros_like(centers)
        grad_radii = torch.zeros_like(radii)
        grad_weights = torch.zeros_like(weights)
        grad_query_points = torch.zeros_like(query_points)
        
        query_points = query_points.contiguous()
        grad_output = grad_output.contiguous()
        grad_centers = grad_centers.contiguous()
        grad_radii = grad_radii.contiguous()
        grad_weights = grad_weights.contiguous()
        grad_query_points = grad_query_points.contiguous()
        
        _backward_3d_batch_no_hash_kernel(query_points, grad_output, centers, radii, weights,
                                        grad_centers, grad_radii, grad_weights, grad_query_points, radial_func_type)
        
        min_radius = 1e-6
        should_block = (radii < min_radius) & (grad_radii > 0)
        grad_radii = torch.where(should_block, torch.zeros_like(grad_radii), grad_radii)
        
        return grad_query_points, grad_centers, grad_radii, grad_weights, None
