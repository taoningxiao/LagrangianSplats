import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
from tqdm import tqdm
# from scene.grid import HashHexPlane

# Import TiDFRBF for kernel-based velocity field
try:
    from velocity_common.dfrbf import TiDFRBF
    from velocity_common.rbf import WendlandC4
    from simple_knn._C import distCUDA2
    TIDFRBF_AVAILABLE = True
except ImportError as e:
    TIDFRBF_AVAILABLE = False
    print(f"Warning: TiDFRBF not available: {e}")

try:
    import taichi as ti
    TI_AVAILABLE = True
except ImportError:
    TI_AVAILABLE = False
    print("Warning: Taichi not available, falling back to PyTorch implementation")


# Taichi kernels for efficient GPU parallel computation
if TI_AVAILABLE:
    ti.init(arch=ti.cuda)
    
    @ti.kernel
    def sample_velocity_kernel(
        xyz_norm: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
        time_norm: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
        velocity_grid: ti.types.ndarray(dtype=ti.f32, ndim=5),  # (T, 3, D, H, W)
        velocity_out: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
        num_time_steps: ti.i32,
        d_res: ti.i32, h_res: ti.i32, w_res: ti.i32
    ):
        """Parallel velocity-field sampling（trilinearinterpolation）"""
        N = xyz_norm.shape[0]
        for i in ti.ndrange(N):
            x = xyz_norm[i, 0]
            y = xyz_norm[i, 1]
            z = xyz_norm[i, 2]
            
            d_coord = (x + 1.0) * 0.5 * (d_res - 1.0)
            h_coord = (y + 1.0) * 0.5 * (h_res - 1.0)
            w_coord = (z + 1.0) * 0.5 * (w_res - 1.0)
            
            d0 = ti.max(0, ti.min(d_res - 1, ti.int32(ti.floor(d_coord))))
            d1 = ti.min(d_res - 1, d0 + 1)
            h0 = ti.max(0, ti.min(h_res - 1, ti.int32(ti.floor(h_coord))))
            h1 = ti.min(h_res - 1, h0 + 1)
            w0 = ti.max(0, ti.min(w_res - 1, ti.int32(ti.floor(w_coord))))
            w1 = ti.min(w_res - 1, w0 + 1)
            
            dd = d_coord - d0
            dh = h_coord - h0
            dw = w_coord - w0
            
            w000 = (1.0 - dd) * (1.0 - dh) * (1.0 - dw)
            w001 = (1.0 - dd) * (1.0 - dh) * dw
            w010 = (1.0 - dd) * dh * (1.0 - dw)
            w011 = (1.0 - dd) * dh * dw
            w100 = dd * (1.0 - dh) * (1.0 - dw)
            w101 = dd * (1.0 - dh) * dw
            w110 = dd * dh * (1.0 - dw)
            w111 = dd * dh * dw
            
            t_norm = time_norm[i]
            t_floor = ti.max(0, ti.min(num_time_steps - 1, ti.int32(ti.floor(t_norm))))
            t_ceil = ti.min(num_time_steps - 1, t_floor + 1)
            t_alpha = t_norm - t_floor
            t_weight_floor = 1.0 - t_alpha
            t_weight_ceil = t_alpha
            
            for c in range(3):
                vel_floor = (velocity_grid[t_floor, c, d0, h0, w0] * w000 +
                            velocity_grid[t_floor, c, d0, h0, w1] * w001 +
                            velocity_grid[t_floor, c, d0, h1, w0] * w010 +
                            velocity_grid[t_floor, c, d0, h1, w1] * w011 +
                            velocity_grid[t_floor, c, d1, h0, w0] * w100 +
                            velocity_grid[t_floor, c, d1, h0, w1] * w101 +
                            velocity_grid[t_floor, c, d1, h1, w0] * w110 +
                            velocity_grid[t_floor, c, d1, h1, w1] * w111)
                
                vel_ceil = (velocity_grid[t_ceil, c, d0, h0, w0] * w000 +
                           velocity_grid[t_ceil, c, d0, h0, w1] * w001 +
                           velocity_grid[t_ceil, c, d0, h1, w0] * w010 +
                           velocity_grid[t_ceil, c, d0, h1, w1] * w011 +
                           velocity_grid[t_ceil, c, d1, h0, w0] * w100 +
                           velocity_grid[t_ceil, c, d1, h0, w1] * w101 +
                           velocity_grid[t_ceil, c, d1, h1, w0] * w110 +
                           velocity_grid[t_ceil, c, d1, h1, w1] * w111)
                
                velocity_out[i, c] = vel_floor * t_weight_floor + vel_ceil * t_weight_ceil
    
    @ti.kernel
    def backward_grad_kernel(
        xyz_norm: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
        time_norm: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
        grad_velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
        grad_velocity_grid: ti.types.ndarray(dtype=ti.f32, ndim=5),  # (T, 3, D, H, W)
        num_time_steps: ti.i32,
        d_res: ti.i32, h_res: ti.i32, w_res: ti.i32
    ):
        """Parallel gradient scatter to the velocity grid"""
        N = xyz_norm.shape[0]
        for i in ti.ndrange(N):
            x = xyz_norm[i, 0]
            y = xyz_norm[i, 1]
            z = xyz_norm[i, 2]
            
            d_coord = (x + 1.0) * 0.5 * (d_res - 1.0)
            h_coord = (y + 1.0) * 0.5 * (h_res - 1.0)
            w_coord = (z + 1.0) * 0.5 * (w_res - 1.0)
            
            d0 = ti.max(0, ti.min(d_res - 1, ti.int32(ti.floor(d_coord))))
            d1 = ti.min(d_res - 1, d0 + 1)
            h0 = ti.max(0, ti.min(h_res - 1, ti.int32(ti.floor(h_coord))))
            h1 = ti.min(h_res - 1, h0 + 1)
            w0 = ti.max(0, ti.min(w_res - 1, ti.int32(ti.floor(w_coord))))
            w1 = ti.min(w_res - 1, w0 + 1)
            
            dd = d_coord - d0
            dh = h_coord - h0
            dw = w_coord - w0
            
            w000 = (1.0 - dd) * (1.0 - dh) * (1.0 - dw)
            w001 = (1.0 - dd) * (1.0 - dh) * dw
            w010 = (1.0 - dd) * dh * (1.0 - dw)
            w011 = (1.0 - dd) * dh * dw
            w100 = dd * (1.0 - dh) * (1.0 - dw)
            w101 = dd * (1.0 - dh) * dw
            w110 = dd * dh * (1.0 - dw)
            w111 = dd * dh * dw
            
            t_norm = time_norm[i]
            t_floor = ti.max(0, ti.min(num_time_steps - 1, ti.int32(ti.floor(t_norm))))
            t_ceil = ti.min(num_time_steps - 1, t_floor + 1)
            t_alpha = t_norm - t_floor
            t_weight_floor = 1.0 - t_alpha
            t_weight_ceil = t_alpha
            
            for c in range(3):
                grad_v_c = grad_velocity[i, c]
                
                ti.atomic_add(grad_velocity_grid[t_floor, c, d0, h0, w0], grad_v_c * t_weight_floor * w000)
                ti.atomic_add(grad_velocity_grid[t_floor, c, d0, h0, w1], grad_v_c * t_weight_floor * w001)
                ti.atomic_add(grad_velocity_grid[t_floor, c, d0, h1, w0], grad_v_c * t_weight_floor * w010)
                ti.atomic_add(grad_velocity_grid[t_floor, c, d0, h1, w1], grad_v_c * t_weight_floor * w011)
                ti.atomic_add(grad_velocity_grid[t_floor, c, d1, h0, w0], grad_v_c * t_weight_floor * w100)
                ti.atomic_add(grad_velocity_grid[t_floor, c, d1, h0, w1], grad_v_c * t_weight_floor * w101)
                ti.atomic_add(grad_velocity_grid[t_floor, c, d1, h1, w0], grad_v_c * t_weight_floor * w110)
                ti.atomic_add(grad_velocity_grid[t_floor, c, d1, h1, w1], grad_v_c * t_weight_floor * w111)
                
                ti.atomic_add(grad_velocity_grid[t_ceil, c, d0, h0, w0], grad_v_c * t_weight_ceil * w000)
                ti.atomic_add(grad_velocity_grid[t_ceil, c, d0, h0, w1], grad_v_c * t_weight_ceil * w001)
                ti.atomic_add(grad_velocity_grid[t_ceil, c, d0, h1, w0], grad_v_c * t_weight_ceil * w010)
                ti.atomic_add(grad_velocity_grid[t_ceil, c, d0, h1, w1], grad_v_c * t_weight_ceil * w011)
                ti.atomic_add(grad_velocity_grid[t_ceil, c, d1, h0, w0], grad_v_c * t_weight_ceil * w100)
                ti.atomic_add(grad_velocity_grid[t_ceil, c, d1, h0, w1], grad_v_c * t_weight_ceil * w101)
                ti.atomic_add(grad_velocity_grid[t_ceil, c, d1, h1, w0], grad_v_c * t_weight_ceil * w110)
                ti.atomic_add(grad_velocity_grid[t_ceil, c, d1, h1, w1], grad_v_c * t_weight_ceil * w111)

    @ti.kernel
    def sample_velocity_kernel_kernel(
        xyz_norm: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
        time_floor: ti.types.ndarray(dtype=ti.i32, ndim=1),  # (N,)
        time_ceil: ti.types.ndarray(dtype=ti.i32, ndim=1),  # (N,)
        time_alpha: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
        velocity_grids: ti.types.ndarray(dtype=ti.f32, ndim=5),
        velocity_out: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
        d_res: ti.i32, h_res: ti.i32, w_res: ti.i32
    ):
        """Parallel velocity-field sampling（trilinearinterpolation + interpolation）- used forVelocityKernel"""
        N = xyz_norm.shape[0]
        for i in ti.ndrange(N):
            x = xyz_norm[i, 0]
            y = xyz_norm[i, 1]
            z = xyz_norm[i, 2]
            
            d_coord = (x + 1.0) * 0.5 * (d_res - 1.0)
            h_coord = (y + 1.0) * 0.5 * (h_res - 1.0)
            w_coord = (z + 1.0) * 0.5 * (w_res - 1.0)
            
            d0 = ti.max(0, ti.min(d_res - 1, ti.int32(ti.floor(d_coord))))
            d1 = ti.min(d_res - 1, d0 + 1)
            h0 = ti.max(0, ti.min(h_res - 1, ti.int32(ti.floor(h_coord))))
            h1 = ti.min(h_res - 1, h0 + 1)
            w0 = ti.max(0, ti.min(w_res - 1, ti.int32(ti.floor(w_coord))))
            w1 = ti.min(w_res - 1, w0 + 1)
            
            dd = d_coord - d0
            dh = h_coord - h0
            dw = w_coord - w0
            
            w000 = (1.0 - dd) * (1.0 - dh) * (1.0 - dw)
            w001 = (1.0 - dd) * (1.0 - dh) * dw
            w010 = (1.0 - dd) * dh * (1.0 - dw)
            w011 = (1.0 - dd) * dh * dw
            w100 = dd * (1.0 - dh) * (1.0 - dw)
            w101 = dd * (1.0 - dh) * dw
            w110 = dd * dh * (1.0 - dw)
            w111 = dd * dh * dw
            
            t_floor = time_floor[i]
            t_ceil = time_ceil[i]
            t_alpha_val = time_alpha[i]
            t_weight_floor = 1.0 - t_alpha_val
            t_weight_ceil = t_alpha_val
            
            for c in range(3):
                vel_floor = (velocity_grids[t_floor, c, d0, h0, w0] * w000 +
                            velocity_grids[t_floor, c, d0, h0, w1] * w001 +
                            velocity_grids[t_floor, c, d0, h1, w0] * w010 +
                            velocity_grids[t_floor, c, d0, h1, w1] * w011 +
                            velocity_grids[t_floor, c, d1, h0, w0] * w100 +
                            velocity_grids[t_floor, c, d1, h0, w1] * w101 +
                            velocity_grids[t_floor, c, d1, h1, w0] * w110 +
                            velocity_grids[t_floor, c, d1, h1, w1] * w111)
                
                vel_ceil = (velocity_grids[t_ceil, c, d0, h0, w0] * w000 +
                           velocity_grids[t_ceil, c, d0, h0, w1] * w001 +
                           velocity_grids[t_ceil, c, d0, h1, w0] * w010 +
                           velocity_grids[t_ceil, c, d0, h1, w1] * w011 +
                           velocity_grids[t_ceil, c, d1, h0, w0] * w100 +
                           velocity_grids[t_ceil, c, d1, h0, w1] * w101 +
                           velocity_grids[t_ceil, c, d1, h1, w0] * w110 +
                           velocity_grids[t_ceil, c, d1, h1, w1] * w111)
                
                velocity_out[i, c] = vel_floor * t_weight_floor + vel_ceil * t_weight_ceil
    
    @ti.kernel
    def backward_grad_kernel_kernel(
        xyz_norm: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
        time_floor: ti.types.ndarray(dtype=ti.i32, ndim=1),  # (N,)
        time_ceil: ti.types.ndarray(dtype=ti.i32, ndim=1),  # (N,)
        time_weight_floor: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
        time_weight_ceil: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
        grad_velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
        grad_velocity_grids: ti.types.ndarray(dtype=ti.f32, ndim=5),
        d_res: ti.i32, h_res: ti.i32, w_res: ti.i32
    ):
        """Parallel gradient scatter to the velocity grid - used forVelocityKernel"""
        N = xyz_norm.shape[0]
        for i in ti.ndrange(N):
            x = xyz_norm[i, 0]
            y = xyz_norm[i, 1]
            z = xyz_norm[i, 2]
            
            d_coord = (x + 1.0) * 0.5 * (d_res - 1.0)
            h_coord = (y + 1.0) * 0.5 * (h_res - 1.0)
            w_coord = (z + 1.0) * 0.5 * (w_res - 1.0)
            
            d0 = ti.max(0, ti.min(d_res - 1, ti.int32(ti.floor(d_coord))))
            d1 = ti.min(d_res - 1, d0 + 1)
            h0 = ti.max(0, ti.min(h_res - 1, ti.int32(ti.floor(h_coord))))
            h1 = ti.min(h_res - 1, h0 + 1)
            w0 = ti.max(0, ti.min(w_res - 1, ti.int32(ti.floor(w_coord))))
            w1 = ti.min(w_res - 1, w0 + 1)
            
            dd = d_coord - d0
            dh = h_coord - h0
            dw = w_coord - w0
            
            w000 = (1.0 - dd) * (1.0 - dh) * (1.0 - dw)
            w001 = (1.0 - dd) * (1.0 - dh) * dw
            w010 = (1.0 - dd) * dh * (1.0 - dw)
            w011 = (1.0 - dd) * dh * dw
            w100 = dd * (1.0 - dh) * (1.0 - dw)
            w101 = dd * (1.0 - dh) * dw
            w110 = dd * dh * (1.0 - dw)
            w111 = dd * dh * dw
            
            t_floor = time_floor[i]
            t_ceil = time_ceil[i]
            t_weight_floor_val = time_weight_floor[i]
            t_weight_ceil_val = time_weight_ceil[i]
            
            for c in range(3):
                grad_v_c = grad_velocity[i, c]
                
                ti.atomic_add(grad_velocity_grids[t_floor, c, d0, h0, w0], grad_v_c * t_weight_floor_val * w000)
                ti.atomic_add(grad_velocity_grids[t_floor, c, d0, h0, w1], grad_v_c * t_weight_floor_val * w001)
                ti.atomic_add(grad_velocity_grids[t_floor, c, d0, h1, w0], grad_v_c * t_weight_floor_val * w010)
                ti.atomic_add(grad_velocity_grids[t_floor, c, d0, h1, w1], grad_v_c * t_weight_floor_val * w011)
                ti.atomic_add(grad_velocity_grids[t_floor, c, d1, h0, w0], grad_v_c * t_weight_floor_val * w100)
                ti.atomic_add(grad_velocity_grids[t_floor, c, d1, h0, w1], grad_v_c * t_weight_floor_val * w101)
                ti.atomic_add(grad_velocity_grids[t_floor, c, d1, h1, w0], grad_v_c * t_weight_floor_val * w110)
                ti.atomic_add(grad_velocity_grids[t_floor, c, d1, h1, w1], grad_v_c * t_weight_floor_val * w111)
                
                ti.atomic_add(grad_velocity_grids[t_ceil, c, d0, h0, w0], grad_v_c * t_weight_ceil_val * w000)
                ti.atomic_add(grad_velocity_grids[t_ceil, c, d0, h0, w1], grad_v_c * t_weight_ceil_val * w001)
                ti.atomic_add(grad_velocity_grids[t_ceil, c, d0, h1, w0], grad_v_c * t_weight_ceil_val * w010)
                ti.atomic_add(grad_velocity_grids[t_ceil, c, d0, h1, w1], grad_v_c * t_weight_ceil_val * w011)
                ti.atomic_add(grad_velocity_grids[t_ceil, c, d1, h0, w0], grad_v_c * t_weight_ceil_val * w100)
                ti.atomic_add(grad_velocity_grids[t_ceil, c, d1, h0, w1], grad_v_c * t_weight_ceil_val * w101)
                ti.atomic_add(grad_velocity_grids[t_ceil, c, d1, h1, w0], grad_v_c * t_weight_ceil_val * w110)
                ti.atomic_add(grad_velocity_grids[t_ceil, c, d1, h1, w1], grad_v_c * t_weight_ceil_val * w111)


class VelocityAdvectionFunction(torch.autograd.Function):
    """
    Custom autograd Function for velocity advection using Taichi GPU kernels.
    """
    @staticmethod
    def forward(ctx, xyz_t0, velocity_grid, xyz_min, xyz_max, time, dt, num_time_steps, grid_resolution):
        """
        Args:
            xyz_t0: (N, 3) t=0position
            velocity_grid: (num_time_steps, 3, d, h, w) velocity grid
            xyz_min: (3,) bbox
            xyz_max: (3,) bboxlatest
            time: (N, 1) target time
            dt: time steps
            num_time_steps: time steps
            grid_resolution: [d, h, w] grid resolution
        
        Returns:
            xyz_t: (N, 3) tposition
        """
        device = xyz_t0.device
        N = xyz_t0.shape[0]
        d_res, h_res, w_res = grid_resolution[0], grid_resolution[1], grid_resolution[2]
        
        ctx.save_for_backward(xyz_t0, velocity_grid, xyz_min, xyz_max, time)
        ctx.dt = dt
        ctx.num_time_steps = num_time_steps
        ctx.grid_resolution = grid_resolution
        ctx.N = N
        
        xyz_t = xyz_t0.clone()
        current_time = torch.zeros((N,), device=device)  # (N,) instead of (N, 1)
        
        num_steps = max(1, int(math.ceil(time.max().item() / dt)))
        actual_dt = (time / num_steps).squeeze(-1)  # (N,)
        
        intermediate_positions = []
        intermediate_times = []
        intermediate_velocities = []
        
        if TI_AVAILABLE:
            velocity_out = torch.zeros((N, 3), device=device)
            
            for step in range(num_steps):
                intermediate_positions.append(xyz_t.clone())
                intermediate_times.append(current_time.clone())
                
                xyz_norm = (xyz_t - xyz_min) / (xyz_max - xyz_min) * 2.0 - 1.0  # (N, 3)
                time_norm = current_time.clamp(0.0, 1.0) * (num_time_steps - 1)  # (N,)
                
                sample_velocity_kernel(
                    xyz_norm, time_norm, velocity_grid,
                    velocity_out, num_time_steps,
                    d_res, h_res, w_res
                )
                
                velocity = velocity_out.clone()
                intermediate_velocities.append(velocity.clone())
                
                xyz_t = xyz_t + velocity * actual_dt.unsqueeze(-1)
                current_time = current_time + actual_dt
        else:
            for step in range(num_steps):
                intermediate_positions.append(xyz_t.clone())
                intermediate_times.append(current_time.clone())
                
                xyz_norm = (xyz_t - xyz_min) / (xyz_max - xyz_min) * 2.0 - 1.0  # (N, 3)
                time_norm = current_time.clamp(0.0, 1.0) * (num_time_steps - 1)  # (N,)
                
                time_floor = torch.floor(time_norm).long().clamp(0, num_time_steps - 1)  # (N,)
                time_ceil = torch.ceil(time_norm).long().clamp(0, num_time_steps - 1)  # (N,)
                time_alpha = time_norm - time_floor.float()  # (N,)
                
                xyz_norm_grid = xyz_norm.reshape(1, 1, 1, N, 3).flip(-1)
                
                unique_times = torch.unique(torch.cat([time_floor, time_ceil]))
                vel_dict = {}
                for t_val in unique_times:
                    t_val = t_val.item()
                    vel_field = velocity_grid[t_val:t_val+1]
                    vel_sample = F.grid_sample(
                        vel_field, xyz_norm_grid,
                        mode='bilinear', align_corners=True, padding_mode='border'
                    )
                    vel_dict[t_val] = vel_sample.squeeze().permute(1, 0)
                
                vel_floor = torch.stack([vel_dict[time_floor[i].item()][i] for i in range(N)])
                vel_ceil = torch.stack([vel_dict[time_ceil[i].item()][i] for i in range(N)])
                velocity = vel_floor * (1 - time_alpha.unsqueeze(-1)) + vel_ceil * time_alpha.unsqueeze(-1)
                
                intermediate_velocities.append(velocity.clone())
                xyz_t = xyz_t + velocity * actual_dt.unsqueeze(-1)
                current_time = current_time + actual_dt
        
        ctx.intermediate_positions = intermediate_positions
        ctx.intermediate_times = intermediate_times
        ctx.intermediate_velocities = intermediate_velocities
        ctx.actual_dt = actual_dt
        
        return xyz_t
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        analytic-gradient backpropagation - useTaichi
        
        grad_output: (N, 3) xyz_t
        """
        xyz_t0, velocity_grid, xyz_min, xyz_max, time = ctx.saved_tensors
        device = xyz_t0.device
        N = ctx.N
        dt = ctx.dt
        num_time_steps = ctx.num_time_steps
        grid_resolution = ctx.grid_resolution
        d_res, h_res, w_res = grid_resolution[0], grid_resolution[1], grid_resolution[2]
        
        grad_xyz_t0 = grad_output.clone()
        grad_velocity_grid = torch.zeros_like(velocity_grid)
        
        grad_xyz = grad_output.clone()
        actual_dt = ctx.actual_dt
        
        if TI_AVAILABLE:
            for step_idx in reversed(range(len(ctx.intermediate_positions))):
                xyz_t = ctx.intermediate_positions[step_idx]
                current_time = ctx.intermediate_times[step_idx]
                
                grad_velocity = grad_xyz * actual_dt.unsqueeze(-1)  # (N, 3)
                
                xyz_norm = (xyz_t - xyz_min) / (xyz_max - xyz_min) * 2.0 - 1.0  # (N, 3)
                time_norm = current_time.clamp(0.0, 1.0) * (num_time_steps - 1)  # (N,)
                
                backward_grad_kernel(
                    xyz_norm, time_norm, grad_velocity,
                    grad_velocity_grid, num_time_steps,
                    d_res, h_res, w_res
                )
                
        else:
            grid_resolution_tensor = torch.tensor(grid_resolution, device=device, dtype=torch.long)
            
            for step_idx in reversed(range(len(ctx.intermediate_positions))):
                xyz_t = ctx.intermediate_positions[step_idx]
                current_time = ctx.intermediate_times[step_idx]
                velocity = ctx.intermediate_velocities[step_idx]
                
                grad_velocity = grad_xyz * actual_dt.unsqueeze(-1)
                
                xyz_norm = (xyz_t - xyz_min) / (xyz_max - xyz_min) * 2.0 - 1.0
                time_norm = current_time.clamp(0.0, 1.0) * (num_time_steps - 1)
                
                time_floor = torch.floor(time_norm).long().clamp(0, num_time_steps - 1)
                time_ceil = torch.ceil(time_norm).long().clamp(0, num_time_steps - 1)
                time_alpha = time_norm - time_floor.float()
                time_weight_floor = 1.0 - time_alpha
                time_weight_ceil = time_alpha
                
                grid_coords = (xyz_norm + 1.0) / 2.0 * (grid_resolution_tensor.float() - 1.0)
                
                d_coord = grid_coords[:, 0]
                h_coord = grid_coords[:, 1]
                w_coord = grid_coords[:, 2]
                
                d0 = torch.floor(d_coord).long().clamp(0, d_res - 1)
                d1 = (d0 + 1).clamp(0, d_res - 1)
                h0 = torch.floor(h_coord).long().clamp(0, h_res - 1)
                h1 = (h0 + 1).clamp(0, h_res - 1)
                w0 = torch.floor(w_coord).long().clamp(0, w_res - 1)
                w1 = (w0 + 1).clamp(0, w_res - 1)
                
                dd = d_coord - d0.float()
                dh = h_coord - h0.float()
                dw = w_coord - w0.float()
                
                w000 = (1 - dd) * (1 - dh) * (1 - dw)
                w001 = (1 - dd) * (1 - dh) * dw
                w010 = (1 - dd) * dh * (1 - dw)
                w011 = (1 - dd) * dh * dw
                w100 = dd * (1 - dh) * (1 - dw)
                w101 = dd * (1 - dh) * dw
                w110 = dd * dh * (1 - dw)
                w111 = dd * dh * dw
                
                corners = [
                    (d0, h0, w0, w000), (d0, h0, w1, w001),
                    (d0, h1, w0, w010), (d0, h1, w1, w011),
                    (d1, h0, w0, w100), (d1, h0, w1, w101),
                    (d1, h1, w0, w110), (d1, h1, w1, w111)
                ]
                
                grad_velocity_grid_flat = grad_velocity_grid.view(num_time_steps, 3, -1)
                
                for t_idx, t_weight in [(time_floor, time_weight_floor), 
                                        (time_ceil, time_weight_ceil)]:
                    for di, hi, wi, spatial_weight in corners:
                        total_weight = t_weight * spatial_weight
                        spatial_flat_indices = di * (h_res * w_res) + hi * w_res + wi
                        
                        for c in range(3):
                            grad_contrib = grad_velocity[:, c] * total_weight
                            unique_times = torch.unique(t_idx)
                            for t_val in unique_times:
                                mask = (t_idx == t_val)
                                if mask.any():
                                    t_spatial_indices = spatial_flat_indices[mask]
                                    t_grad_contrib = grad_contrib[mask]
                                    grad_velocity_grid_flat[t_val, c].index_add_(
                                        0, t_spatial_indices, t_grad_contrib
                                    )
        
        grad_xyz_t0 = grad_xyz
        
        return grad_xyz_t0, grad_velocity_grid, None, None, None, None, None, None


class VelocityGrid(nn.Module):
    """
    velocity grid：stores30time stepsvelocity field，eachgrid resolution75x150x75x3
    """
    def __init__(self, num_time_steps=30, grid_resolution=[75, 150, 75], xyz_min=None, xyz_max=None):
        super(VelocityGrid, self).__init__()
        self.num_time_steps = num_time_steps
        self.grid_resolution = grid_resolution  # [d, h, w]
        self.channels = 3
        
        self.velocity_grid = nn.Parameter(
            torch.randn([num_time_steps, self.channels, *grid_resolution]) * 0.01
        )
        
        if xyz_min is not None and xyz_max is not None:
            self.set_aabb(xyz_max, xyz_min)
        else:
            self.register_buffer('xyz_min', None)
            self.register_buffer('xyz_max', None)
    
    def set_aabb(self, xyz_max, xyz_min):
        """Set the world-space bounding box."""
        if isinstance(xyz_max, np.ndarray):
            xyz_max = torch.tensor(xyz_max, dtype=torch.float32, device="cuda")
        elif isinstance(xyz_max, torch.Tensor):
            xyz_max = xyz_max.to("cuda")
        if isinstance(xyz_min, np.ndarray):
            xyz_min = torch.tensor(xyz_min, dtype=torch.float32, device="cuda")
        elif isinstance(xyz_min, torch.Tensor):
            xyz_min = xyz_min.to("cuda")
        self.register_buffer('xyz_min', xyz_min)
        self.register_buffer('xyz_max', xyz_max)
    
    def sample_velocity(self, xyz, time):
        """
        Sample velocity at the given positions and times.
        
        Args:
            xyz: (N, 3) world coordinatesposition
            time: (N, 1) ，[0, 1]
        
        Returns:
            velocity: (N, 3) velocity vector
        """
        if self.xyz_min is None or self.xyz_max is None:
            raise ValueError("AABB not set. Call set_aabb() first.")
        
        device = xyz.device
        N = xyz.shape[0]
        
        xyz_norm = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)  # (N, 3) in [0, 1]
        xyz_norm = xyz_norm * 2.0 - 1.0  # (N, 3) in [-1, 1]
        
        time_norm = time.clamp(0.0, 1.0) * (self.num_time_steps - 1)  # (N, 1) in [0, num_time_steps-1]
        
        time_floor = torch.floor(time_norm).long().clamp(0, self.num_time_steps - 1)  # (N, 1)
        time_ceil = torch.ceil(time_norm).long().clamp(0, self.num_time_steps - 1)  # (N, 1)
        time_alpha = time_norm - time_floor.float()  # (N, 1)
        
        xyz_norm_grid = xyz_norm.reshape(1, 1, 1, N, 3)  # (1, 1, 1, N, 3)
        xyz_norm_grid = xyz_norm_grid.flip(-1)
        
        unique_floor_times = torch.unique(time_floor.squeeze())
        unique_ceil_times = torch.unique(time_ceil.squeeze())
        
        vel_dict_floor = {}
        vel_dict_ceil = {}
        
        for t_val in unique_floor_times:
            t_val = t_val.item()
            vel_field = self.velocity_grid[t_val:t_val+1]  # [1, 3, d, h, w]
            vel_sample = F.grid_sample(
                vel_field,
                xyz_norm_grid,
                mode='bilinear',
                align_corners=True,
                padding_mode='border'
            )  # [1, 3, 1, 1, N]
            vel_dict_floor[t_val] = vel_sample.squeeze().permute(1, 0)  # (N, 3)
        
        for t_val in unique_ceil_times:
            t_val = t_val.item()
            vel_field = self.velocity_grid[t_val:t_val+1]  # [1, 3, d, h, w]
            vel_sample = F.grid_sample(
                vel_field,
                xyz_norm_grid,
                mode='bilinear',
                align_corners=True,
                padding_mode='border'
            )  # [1, 3, 1, 1, N]
            vel_dict_ceil[t_val] = vel_sample.squeeze().permute(1, 0)  # (N, 3)
        
        vel_floor = torch.zeros((N, 3), device=device)
        vel_ceil = torch.zeros((N, 3), device=device)
        
        for i in range(N):
            t_floor = time_floor[i, 0].item()
            t_ceil = time_ceil[i, 0].item()
            vel_floor[i] = vel_dict_floor[t_floor][i]
            vel_ceil[i] = vel_dict_ceil[t_ceil][i]
        
        velocity = vel_floor * (1 - time_alpha) + vel_ceil * time_alpha  # (N, 3)
        
        return velocity
    
    def advect_positions(self, xyz_t0, time, dt=1e-3):
        """
        Advect frame-0 positions to time t with the velocity field.
        Use a custom Function for analytic-gradient backpropagation.
        
        Args:
            xyz_t0: (N, 3) t=0position（world coordinates）
            time: (N, 1) target time，[0, 1]
            dt: time steps
        
        Returns:
            xyz_t: (N, 3) tposition
        """
        return VelocityAdvectionFunction.apply(
            xyz_t0,
            self.velocity_grid,
            self.xyz_min,
            self.xyz_max,
            time,
            dt,
            self.num_time_steps,
            self.grid_resolution
        )


def generate_kernel_centers(xyz_min, xyz_max, n_kernels, device='cuda'):
    """
    Randomly generate kernel centers inside the bounding box.
    
    Args:
        xyz_min: (3,) bbox
        xyz_max: (3,) bboxlatest
        n_kernels: kernelcount
        device: device
    
    Returns:
        centers: (n_kernels, 3) kernelcenters
    """
    if isinstance(xyz_min, np.ndarray):
        xyz_min = torch.tensor(xyz_min, dtype=torch.float32, device=device)
    if isinstance(xyz_max, np.ndarray):
        xyz_max = torch.tensor(xyz_max, dtype=torch.float32, device=device)
    
    centers = torch.rand(n_kernels, 3, device=device) * (xyz_max - xyz_min) + xyz_min
    return centers


def compute_h_from_kernels(kernel_centers, device='cuda'):
    """
    Use the average nearest-neighbor distance between kernel centers as h.
    
    Args:
        kernel_centers: (n_kernels, 3) kernelcenters
        device: device
    
    Returns:
        h: scalar，average nearest-neighbor distance
    """
    if isinstance(kernel_centers, np.ndarray):
        kernel_centers = torch.tensor(kernel_centers, dtype=torch.float32, device=device)
    elif isinstance(kernel_centers, torch.Tensor):
        kernel_centers = kernel_centers.to(device)
    
    dist2 = torch.clamp_min(distCUDA2(kernel_centers), 0.0000001)  # (n_kernels,)
    
    h = torch.sqrt(dist2).mean().item()
    
    return h


class VelocityKernelAdvectionFunction(torch.autograd.Function):
    """
    Custom autograd Function for kernel-based velocity advection with gradients to TiDFRBF parameters.
    Evaluate TiDFRBF on grid vertices, then interpolate velocities at advected points.
    """
    @staticmethod
    def forward(ctx, xyz_t0, vel_models, xyz_min, xyz_max, time, dt, num_time_steps, grid_resolution):
        """
        Args:
            xyz_t0: (N, 3) t=0position
            vel_models: ModuleList of TiDFRBF models (num_time_steps)
            xyz_min: (3,) bbox
            xyz_max: (3,) bboxlatest
            time: (N, 1) target time，[0, 1]
            dt: time steps
            num_time_steps: time steps
            grid_resolution: [d, h, w] grid resolution
        
        Returns:
            xyz_t: (N, 3) tposition
        """
        device = xyz_t0.device
        N = xyz_t0.shape[0]
        d_res, h_res, w_res = grid_resolution[0], grid_resolution[1], grid_resolution[2]
        
        ctx.save_for_backward(xyz_t0, xyz_min, xyz_max, time)
        ctx.vel_models = vel_models
        ctx.dt = dt
        ctx.num_time_steps = num_time_steps
        ctx.grid_resolution = grid_resolution
        ctx.N = N
        
        d_coords = torch.linspace(0, 1, d_res, device=device)  # (d_res,)
        h_coords = torch.linspace(0, 1, h_res, device=device)  # (h_res,)
        w_coords = torch.linspace(0, 1, w_res, device=device)  # (w_res,)
        
        d_coords_world = d_coords * (xyz_max[0] - xyz_min[0]) + xyz_min[0]
        h_coords_world = h_coords * (xyz_max[1] - xyz_min[1]) + xyz_min[1]
        w_coords_world = w_coords * (xyz_max[2] - xyz_min[2]) + xyz_min[2]
        
        grid_d, grid_h, grid_w = torch.meshgrid(d_coords_world, h_coords_world, w_coords_world, indexing='ij')
        grid_points = torch.stack([grid_d, grid_h, grid_w], dim=-1)  # (d_res, h_res, w_res, 3)
        grid_points_flat = grid_points.reshape(-1, 3)  # (d_res * h_res * w_res, 3)
        
        xyz_t = xyz_t0.clone()
        current_time = torch.zeros((N,), device=device)  # (N,)
        
        num_steps = max(1, int(math.ceil(time.max().item() / dt)))
        actual_dt = (time / num_steps).squeeze(-1)  # (N,)
        
        intermediate_positions = []
        intermediate_times = []
        intermediate_velocities = []
        intermediate_velocity_grids = []
        
        for step in range(num_steps):
            intermediate_positions.append(xyz_t.clone())
            intermediate_times.append(current_time.clone())
            
            time_normalized = current_time.clamp(0.0, 1.0)  # (N,)
            
            xyz_norm = (xyz_t - xyz_min) / (xyz_max - xyz_min)  # (N, 3) in [0, 1]
            xyz_norm = xyz_norm * 2.0 - 1.0  # (N, 3) in [-1, 1]
            
            time_norm = time_normalized * (num_time_steps - 1)  # (N,)
            
            time_floor = torch.floor(time_norm).long().clamp(0, num_time_steps - 1)  # (N,)
            time_ceil = torch.ceil(time_norm).long().clamp(0, num_time_steps - 1)  # (N,)
            time_alpha = time_norm - time_floor.float()  # (N,)
            
            time_floor_i32 = time_floor.int()  # (N,) int32 for Taichi
            time_ceil_i32 = time_ceil.int()  # (N,) int32 for Taichi
            
            xyz_norm_grid = xyz_norm.reshape(1, 1, 1, N, 3)  # (1, 1, 1, N, 3)
            xyz_norm_grid = xyz_norm_grid.flip(-1)
            
            unique_floor_times = torch.unique(time_floor)
            unique_ceil_times = torch.unique(time_ceil)
            unique_times = torch.unique(torch.cat([unique_floor_times, unique_ceil_times]))
            
            velocity_grids_dict = {}
            
            for t_val in unique_times:
                t_val = t_val.item()
                vel_model = vel_models[t_val]
                
                with torch.enable_grad():
                    grid_velocities = vel_model(grid_points_flat)  
                    
                    grid_velocities_grid = grid_velocities.reshape(d_res, h_res, w_res, 3).permute(3, 0, 1, 2).unsqueeze(0)
                
                velocity_grids_dict[t_val] = grid_velocities_grid
            
            if TI_AVAILABLE:
                velocity_grids_all = torch.zeros((num_time_steps, 3, d_res, h_res, w_res), device=device)
                for t_val in unique_times:
                    t_val = t_val.item()
                    velocity_grids_all[t_val] = velocity_grids_dict[t_val].squeeze(0)  # [3, d, h, w]
                
                velocity_out = torch.zeros((N, 3), device=device)
                sample_velocity_kernel_kernel(
                    xyz_norm, time_floor_i32, time_ceil_i32, time_alpha,
                    velocity_grids_all, velocity_out,
                    d_res, h_res, w_res
                )
                velocity = velocity_out
            else:
                vel_dict_floor = {}
                vel_dict_ceil = {}
                
                xyz_norm_grid = xyz_norm.reshape(1, 1, 1, N, 3)  # (1, 1, 1, N, 3)
                xyz_norm_grid = xyz_norm_grid.flip(-1)
                
                for t_val in unique_times:
                    t_val = t_val.item()
                    grid_velocities_grid = velocity_grids_dict[t_val]
                    
                    vel_sample = F.grid_sample(
                        grid_velocities_grid,
                        xyz_norm_grid,
                        mode='bilinear',
                        align_corners=True,
                        padding_mode='border'
                    )  # [1, 3, 1, 1, N]
                    
                    vel_sample = vel_sample.squeeze().permute(1, 0)  # (N, 3)
                    
                    if t_val in unique_floor_times:
                        vel_dict_floor[t_val] = vel_sample
                    if t_val in unique_ceil_times:
                        vel_dict_ceil[t_val] = vel_sample
                
                vel_floor = torch.zeros((N, 3), device=device)
                vel_ceil = torch.zeros((N, 3), device=device)
                
                for i in range(N):
                    t_floor = time_floor[i].item()
                    t_ceil = time_ceil[i].item()
                    vel_floor[i] = vel_dict_floor[t_floor][i]
                    vel_ceil[i] = vel_dict_ceil[t_ceil][i]
                
                velocity = vel_floor * (1 - time_alpha.unsqueeze(-1)) + vel_ceil * time_alpha.unsqueeze(-1)  # (N, 3)
            
            intermediate_velocities.append(velocity.clone())
            intermediate_velocity_grids.append(velocity_grids_dict)
            
            xyz_t = xyz_t + velocity * actual_dt.unsqueeze(-1)
            current_time = current_time + actual_dt
        
        ctx.intermediate_positions = intermediate_positions
        ctx.intermediate_times = intermediate_times
        ctx.intermediate_velocities = intermediate_velocities
        ctx.intermediate_velocity_grids = intermediate_velocity_grids
        ctx.actual_dt = actual_dt
        
        return xyz_t
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        backward pass - advect，TiDFRBF
        
        grad_output: (N, 3) xyz_t
        """
        xyz_t0, xyz_min, xyz_max, time = ctx.saved_tensors
        device = xyz_t0.device
        N = ctx.N
        dt = ctx.dt
        num_time_steps = ctx.num_time_steps
        grid_resolution = ctx.grid_resolution
        vel_models = ctx.vel_models
        actual_dt = ctx.actual_dt
        d_res, h_res, w_res = grid_resolution[0], grid_resolution[1], grid_resolution[2]
        
        grad_xyz = grad_output.clone()
        grid_resolution_tensor = torch.tensor(grid_resolution, device=device, dtype=torch.long)
        
        for step_idx in reversed(range(len(ctx.intermediate_positions))):
            xyz_t = ctx.intermediate_positions[step_idx]
            current_time = ctx.intermediate_times[step_idx]
            velocity_stored = ctx.intermediate_velocities[step_idx]
            velocity_grids_dict = ctx.intermediate_velocity_grids[step_idx]
            
            grad_velocity = grad_xyz * actual_dt.unsqueeze(-1)  # (N, 3)
            
            xyz_norm = (xyz_t - xyz_min) / (xyz_max - xyz_min) * 2.0 - 1.0  # (N, 3)
            time_norm = current_time.clamp(0.0, 1.0) * (num_time_steps - 1)  # (N,)
            
            time_floor = torch.floor(time_norm).long().clamp(0, num_time_steps - 1)  # (N,)
            time_ceil = torch.ceil(time_norm).long().clamp(0, num_time_steps - 1)  # (N,)
            time_alpha = time_norm - time_floor.float()  # (N,)
            time_weight_floor = 1.0 - time_alpha
            time_weight_ceil = time_alpha
            
            time_floor_i32 = time_floor.int()  # (N,) int32 for Taichi
            time_ceil_i32 = time_ceil.int()  # (N,) int32 for Taichi
            
            grad_velocity_grid_dict = {}
            unique_times_in_step = list(velocity_grids_dict.keys())
            for t_val in unique_times_in_step:
                grad_velocity_grid_dict[t_val] = torch.zeros((1, 3, d_res, h_res, w_res), device=device)
            
            if TI_AVAILABLE:
                grad_velocity_grids_all = torch.zeros((num_time_steps, 3, d_res, h_res, w_res), device=device)
                
                backward_grad_kernel_kernel(
                    xyz_norm, time_floor_i32, time_ceil_i32, time_weight_floor, time_weight_ceil,
                    grad_velocity, grad_velocity_grids_all,
                    d_res, h_res, w_res
                )
                
                for t_val in unique_times_in_step:
                    grad_velocity_grid_dict[t_val] = grad_velocity_grids_all[t_val:t_val+1]
            else:
                grid_coords = (xyz_norm + 1.0) / 2.0 * (grid_resolution_tensor.float() - 1.0)
                
                d_coord = grid_coords[:, 0]
                h_coord = grid_coords[:, 1]
                w_coord = grid_coords[:, 2]
                
                d0 = torch.floor(d_coord).long().clamp(0, d_res - 1)
                d1 = (d0 + 1).clamp(0, d_res - 1)
                h0 = torch.floor(h_coord).long().clamp(0, h_res - 1)
                h1 = (h0 + 1).clamp(0, h_res - 1)
                w0 = torch.floor(w_coord).long().clamp(0, w_res - 1)
                w1 = (w0 + 1).clamp(0, w_res - 1)
                
                dd = d_coord - d0.float()
                dh = h_coord - h0.float()
                dw = w_coord - w0.float()
                
                w000 = (1 - dd) * (1 - dh) * (1 - dw)
                w001 = (1 - dd) * (1 - dh) * dw
                w010 = (1 - dd) * dh * (1 - dw)
                w011 = (1 - dd) * dh * dw
                w100 = dd * (1 - dh) * (1 - dw)
                w101 = dd * (1 - dh) * dw
                w110 = dd * dh * (1 - dw)
                w111 = dd * dh * dw
                
                corners = [
                    (d0, h0, w0, w000), (d0, h0, w1, w001),
                    (d0, h1, w0, w010), (d0, h1, w1, w011),
                    (d1, h0, w0, w100), (d1, h0, w1, w101),
                    (d1, h1, w0, w110), (d1, h1, w1, w111)
                ]
                
                for t_idx, t_weight in [(time_floor, time_weight_floor), 
                                        (time_ceil, time_weight_ceil)]:
                    for di, hi, wi, spatial_weight in corners:
                        total_weight = t_weight * spatial_weight
                        spatial_flat_indices = di * (h_res * w_res) + hi * w_res + wi
                        
                        for c in range(3):
                            grad_contrib = grad_velocity[:, c] * total_weight
                            unique_times = torch.unique(t_idx)
                            for t_val in unique_times:
                                t_val = t_val.item()
                                mask = (t_idx == t_val)
                                if mask.any() and t_val in grad_velocity_grid_dict:
                                    t_spatial_indices = spatial_flat_indices[mask]
                                    t_grad_contrib = grad_contrib[mask]
                                    
                                    grad_velocity_grid_flat = grad_velocity_grid_dict[t_val].view(1, 3, -1)
                                    grad_velocity_grid_flat[0, c].index_add_(
                                        0, t_spatial_indices, t_grad_contrib
                                    )
            
            for t_val, grad_velocity_grid in grad_velocity_grid_dict.items():
                if grad_velocity_grid.abs().sum() > 0:
                    vel_model = vel_models[t_val]
                    
                    # grad_velocity_grid_flat = grad_velocity_grid.view(3, -1).permute(1, 0)  # (d_res * h_res * w_res, 3)
                    
                    # grid_velocities_flat = grid_velocities_stored.view(3, -1).permute(1, 0)  # (d_res * h_res * w_res, 3)
                    
                    # vel_params = [p for p in vel_model.parameters() if p.requires_grad]
                    # import pdb; pdb.set_trace()
                    # if len(vel_params) > 0 and grid_velocities_flat.requires_grad:
                    #     print("backprop to velocity kernel")
                    #     torch.autograd.grad(
                    #         outputs=grid_velocities_flat,
                    #         inputs=vel_params,  # centers, radii, weights
                    #         grad_outputs=grad_velocity_grid_flat,
                    #         retain_graph=True,
                    #         create_graph=False,
                    #         only_inputs=False
                    #     )
                    #     print("backprop to velocity kernel done")
                    # else:
                    #     print("no velocity kernel to backprop")
                    grid_velocities_stored = velocity_grids_dict[t_val] 
                    
                    
                    target_tensor = grid_velocities_stored
                    grad_tensor = grad_velocity_grid
                    
                    if target_tensor.requires_grad:
                        model_params = [p for p in vel_model.parameters() if p.requires_grad]
                        
                        if len(model_params) > 0:
                            grads = torch.autograd.grad(
                                outputs=target_tensor,
                                inputs=model_params,
                                grad_outputs=grad_tensor,
                                retain_graph=True,
                                allow_unused=True
                            )
                            
                            for param, grad in zip(model_params, grads):
                                if grad is not None:
                                    if param.grad is None:
                                        param.grad = grad.clone()
                                    else:
                                        param.grad += grad
                    else:
                        print(f"Warning: Frame {t_val} tensor has no grad_fn!")
            
            # grad_xyz_t_step = torch.autograd.grad(
            #     outputs=velocity_stored,
            #     inputs=xyz_t,
            #     grad_outputs=grad_velocity,
            #     retain_graph=True,
            #     create_graph=False,
            #     only_inputs=True
            # )[0]
            # grad_xyz = grad_xyz + grad_xyz_t_step
        
        grad_xyz_t0 = grad_xyz
        
        return grad_xyz_t0, None, None, None, None, None, None, None


class VelocityKernel(nn.Module):
    """
    Kernel-based velocity field represented by TiDFRBF.
    Each time step uses an independent TiDFRBF model.
    """
    def __init__(self, n_kernels=1000, num_time_steps=30, xyz_min=None, xyz_max=None, 
                 init_centers=None, h=None, grid_resolution=[100, 150, 100], device='cuda'):
        super(VelocityKernel, self).__init__()
        if not TIDFRBF_AVAILABLE:
            raise ImportError("TiDFRBF not available. Please install required dependencies.")
        
        self.n_kernels = n_kernels
        self.num_time_steps = num_time_steps
        self.grid_resolution = grid_resolution  # [d, h, w]
        self.device = device
        
        if xyz_min is not None and xyz_max is not None:
            self.set_aabb(xyz_max, xyz_min)
        else:
            self.register_buffer('xyz_min', None)
            self.register_buffer('xyz_max', None)
        
        if init_centers is None:
            if xyz_min is None or xyz_max is None:
                raise ValueError("Either init_centers or (xyz_min, xyz_max) must be provided")
            init_centers = generate_kernel_centers(xyz_min, xyz_max, n_kernels, device)
        
        if h is None:
            raise ValueError("h (radius) must be provided")
        
        radial_func = WendlandC4()
        self.vel_models = nn.ModuleList([
            TiDFRBF(radial_func, init_centers.cpu().numpy(), h, device=device)
            for _ in range(num_time_steps)
        ]).to(device)
        
        print(f'[*] VelocityKernel complete: {n_kernels} kernel, {num_time_steps} time steps')
    
    def set_aabb(self, xyz_max, xyz_min):
        """Set the world-space bounding box."""
        if isinstance(xyz_max, np.ndarray):
            xyz_max = torch.tensor(xyz_max, dtype=torch.float32, device=self.device)
        elif isinstance(xyz_max, torch.Tensor):
            xyz_max = xyz_max.to(self.device)
        if isinstance(xyz_min, np.ndarray):
            xyz_min = torch.tensor(xyz_min, dtype=torch.float32, device=self.device)
        elif isinstance(xyz_min, torch.Tensor):
            xyz_min = xyz_min.to(self.device)
        self.register_buffer('xyz_min', xyz_min)
        self.register_buffer('xyz_max', xyz_max)
    
    def sample_velocity(self, xyz, time):
        """
        Sample velocity at the given positions and times.
        useTiDFRBF，interpolationquery points
        
        Args:
            xyz: (N, 3) world coordinatesposition
            time: (N, 1) ，[0, 1]
        
        Returns:
            velocity: (N, 3) velocity vector
        """
        if self.xyz_min is None or self.xyz_max is None:
            raise ValueError("AABB not set. Call set_aabb() first.")
        
        N = xyz.shape[0]
        device = xyz.device
        d_res, h_res, w_res = self.grid_resolution[0], self.grid_resolution[1], self.grid_resolution[2]
        
        time_normalized = time.clamp(0.0, 1.0)  # (N, 1)
        
        xyz_norm = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)  # (N, 3) in [0, 1]
        xyz_norm = xyz_norm * 2.0 - 1.0  # (N, 3) in [-1, 1]
        
        time_norm = time_normalized * (self.num_time_steps - 1)  # (N, 1) in [0, num_time_steps-1]
        
        time_floor = torch.floor(time_norm).long().clamp(0, self.num_time_steps - 1)  # (N, 1)
        time_ceil = torch.ceil(time_norm).long().clamp(0, self.num_time_steps - 1)  # (N, 1)
        time_alpha = time_norm - time_floor.float()  # (N, 1)
        
        d_coords = torch.linspace(0, 1, d_res, device=device)  # (d_res,)
        h_coords = torch.linspace(0, 1, h_res, device=device)  # (h_res,)
        w_coords = torch.linspace(0, 1, w_res, device=device)  # (w_res,)
        
        d_coords_world = d_coords * (self.xyz_max[0] - self.xyz_min[0]) + self.xyz_min[0]
        h_coords_world = h_coords * (self.xyz_max[1] - self.xyz_min[1]) + self.xyz_min[1]
        w_coords_world = w_coords * (self.xyz_max[2] - self.xyz_min[2]) + self.xyz_min[2]
        
        grid_d, grid_h, grid_w = torch.meshgrid(d_coords_world, h_coords_world, w_coords_world, indexing='ij')
        grid_points = torch.stack([grid_d, grid_h, grid_w], dim=-1)  # (d_res, h_res, w_res, 3)
        grid_points_flat = grid_points.reshape(-1, 3)  # (d_res * h_res * w_res, 3)
        
        xyz_norm_grid = xyz_norm.reshape(1, 1, 1, N, 3)  # (1, 1, 1, N, 3)
        xyz_norm_grid = xyz_norm_grid.flip(-1)
        
        unique_floor_times = torch.unique(time_floor.squeeze())
        unique_ceil_times = torch.unique(time_ceil.squeeze())
        unique_times = torch.unique(torch.cat([unique_floor_times, unique_ceil_times]))
        
        vel_dict_floor = {}
        vel_dict_ceil = {}
        
        for t_val in unique_times:
            t_val = t_val.item()
            vel_model = self.vel_models[t_val]
            
            grid_velocities = vel_model(grid_points_flat)  # (d_res * h_res * w_res, 3)
            
            grid_velocities = grid_velocities.reshape(d_res, h_res, w_res, 3).permute(3, 0, 1, 2).unsqueeze(0)
            
            vel_sample = F.grid_sample(
                grid_velocities,
                xyz_norm_grid,
                mode='bilinear',
                align_corners=True,
                padding_mode='border'
            )  # [1, 3, 1, 1, N]
            
            vel_sample = vel_sample.squeeze().permute(1, 0)  # (N, 3)
            
            if t_val in unique_floor_times:
                vel_dict_floor[t_val] = vel_sample
            if t_val in unique_ceil_times:
                vel_dict_ceil[t_val] = vel_sample
        
        vel_floor = torch.zeros((N, 3), device=device)
        vel_ceil = torch.zeros((N, 3), device=device)
        
        for i in range(N):
            t_floor = time_floor[i, 0].item()
            t_ceil = time_ceil[i, 0].item()
            vel_floor[i] = vel_dict_floor[t_floor][i]
            vel_ceil[i] = vel_dict_ceil[t_ceil][i]
        
        velocity = vel_floor * (1 - time_alpha) + vel_ceil * time_alpha  # (N, 3)
        
        return velocity
    
    def advect_positions(self, xyz_t0, time, dt=1e-3):
        """
        Advect frame-0 positions to time t with the velocity field.
        Use a custom Function for analytic-gradient backpropagation.TiDFRBF
        
        Args:
            xyz_t0: (N, 3) t=0position（world coordinates）
            time: (N, 1) target time，[0, 1]
            dt: time steps
        
        Returns:
            xyz_t: (N, 3) tposition
        """
        if self.xyz_min is None or self.xyz_max is None:
            raise ValueError("AABB not set. Call set_aabb() first.")
        
        self.vel_models.train()
        for p in self.vel_models.parameters():
            p.requires_grad = True
        
        return VelocityKernelAdvectionFunction.apply(
            xyz_t0,
            self.vel_models,
            self.xyz_min,
            self.xyz_max,
            time,
            dt,
            self.num_time_steps,
            self.grid_resolution
        )


class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        # breakpoint()
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        self.ratio=0
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):

        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:

            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            # breakpoint()
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature,self.grid_pe)
            hidden = torch.cat([grid_feature],-1) 
        
        
        hidden = self.feature_out(hidden)   
 

        return hidden
    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)
        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)
        # breakpoint()
        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            dx = self.pos_deform(hidden)
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            pts = rays_pts_emb[:,:3]*mask + dx
        if self.args.no_ds :
            
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)

            scales = torch.zeros_like(scales_emb[:,:3])
            scales = scales_emb[:,:3]*mask + ds
            
        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)

            rotations = torch.zeros_like(rotations_emb[:,:4])
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr

        if self.args.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
          
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do
        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])

            shs = torch.zeros_like(shs_emb)
            # breakpoint()
            shs = shs_emb*mask.unsqueeze(-1) + dshs

        return pts, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        
        self.use_velocity_advection = getattr(args, 'use_velocity_advection', False)
        
        if self.use_velocity_advection:
            num_time_steps = getattr(args, 'velocity_num_time_steps', 30)
            grid_resolution = getattr(args, 'velocity_grid_resolution', [100, 150, 100])
            self.velocity_grid = VelocityGrid(
                num_time_steps=num_time_steps,
                grid_resolution=grid_resolution
            )
        else:
            self.velocity_grid = None
        
        self.use_velocity_kernel = getattr(args, 'use_velocity_kernel', False)
        
        if self.use_velocity_kernel:
            if not TIDFRBF_AVAILABLE:
                raise ImportError("TiDFRBF not available. Cannot use velocity_kernel mode.")
            self.velocity_kernel = None
            self._kernel_num = getattr(args, 'kernel_num', 1000)
            self._velocity_kernel_num_time_steps = getattr(args, 'velocity_kernel_num_time_steps', 30)
            self._velocity_kernel_grid_resolution = getattr(args, 'velocity_kernel_grid_resolution', [100, 150, 100])
        else:
            self.velocity_kernel = None
        
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        if self.use_velocity_advection:
            
            means3D = self.velocity_grid.advect_positions(point, times_sel, dt=1.0/30)
            
            return means3D, scales, rotations, opacity, shs
        elif self.use_velocity_kernel:
            
            if self.velocity_kernel is None:
                raise RuntimeError("VelocityKernel not initialized. Call initialize_velocity_kernel() first.")
            
            means3D = self.velocity_kernel.advect_positions(point, times_sel, dt=1.0/30)
            
            return means3D, scales, rotations, opacity, shs
        else:
            # times_emb = poc_fre(times_sel, self.time_poc)
            point_emb = poc_fre(point,self.pos_poc)
            scales_emb = poc_fre(scales,self.rotation_scaling_poc)
            rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
            # time_emb = poc_fre(times_sel, self.time_poc)
            # times_feature = self.timenet(time_emb)
            means3D, scales, rotations, opacity, shs = self.deformation_net( point_emb,
                                                      scales_emb,
                                                    rotations_emb,
                                                    opacity,
                                                    shs,
                                                    None,
                                                    times_sel)
            return means3D, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        mlp_params = self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
        return mlp_params
    
    def get_grid_parameters(self):
        grid_params = self.deformation_net.get_grid_parameters()
        if self.use_velocity_advection and self.velocity_grid is not None:
            grid_params.append(self.velocity_grid.velocity_grid)
        return grid_params
    
    def get_velocity_kernel_parameters(self):
        """
        Return kernel velocity-field parameters（weights, centers, radii）
        used to assign separate learning rates
        """
        kernel_params = []
        if self.use_velocity_kernel and self.velocity_kernel is not None:
            for vel_model in self.velocity_kernel.vel_models:
                kernel_params.extend([vel_model.weights, vel_model.centers, vel_model.radii])
        return kernel_params
    
    def set_velocity_aabb(self, xyz_max, xyz_min):
        """Set the velocity-grid AABB."""
        if self.use_velocity_advection and self.velocity_grid is not None:
            self.velocity_grid.set_aabb(xyz_max, xyz_min)
    
    def set_velocity_kernel_aabb(self, xyz_max, xyz_min):
        """Set the kernel velocity-field AABB."""
        if self.use_velocity_kernel and self.velocity_kernel is not None:
            self.velocity_kernel.set_aabb(xyz_max, xyz_min)
    
    def initialize_velocity_kernel(self, gaussians, xyz_max, xyz_min, n_kernels=None, device='cuda'):
        """
        Initialize the kernel velocity field（lazy initialization）
        
        Args:
            gaussians: GaussianModelobject（，use）
            xyz_max: bboxlatest
            xyz_min: bbox
            n_kernels: kernelcount（None，useargsin）
            device: device
        """
        if not self.use_velocity_kernel:
            return
        
        if n_kernels is None:
            n_kernels = getattr(self, '_kernel_num', 1000)
        
        num_time_steps = getattr(self, '_velocity_kernel_num_time_steps', 30)
        grid_resolution = getattr(self, '_velocity_kernel_grid_resolution', [100, 150, 100])
        
        init_centers = generate_kernel_centers(xyz_min, xyz_max, n_kernels, device)
        
        h = compute_h_from_kernels(init_centers, device)
        
        self.velocity_kernel = VelocityKernel(
            n_kernels=n_kernels,
            num_time_steps=num_time_steps,
            xyz_min=xyz_min,
            xyz_max=xyz_max,
            init_centers=init_centers,
            h=h,
            grid_resolution=grid_resolution,
            device=device
        )
        
        print(f'[*] VelocityKernel complete: {n_kernels} kernel, h={h:.6f}, grid_resolution={grid_resolution}')

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb