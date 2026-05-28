import numpy as np
import math
import time
import taichi as ti

ti.init(arch=ti.cuda, default_fp=ti.f32)

@ti.data_oriented
class PoissonDisk3DFast:
    def __init__(self, max_points=500000):
        self.max_points = max_points
        self.samples = ti.Vector.field(3, dtype=ti.f32, shape=max_points)
        self.active_indices = ti.field(dtype=ti.i32, shape=max_points)
        
        self.samples_count = ti.field(dtype=ti.i32, shape=())
        self.active_count = ti.field(dtype=ti.i32, shape=())
        
        self.grid = None
        self.grid_size = ti.Vector.field(3, dtype=ti.i32, shape=())
        self.grid_origin = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.inv_cell_size = ti.field(dtype=ti.f32, shape=())
        
        self.bounds_min = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.bounds_max = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.r_min = ti.field(dtype=ti.f32, shape=())
        self.r_min_sq = ti.field(dtype=ti.f32, shape=())

    def init_grid(self, min_corner, max_corner, spacing):
        """ Python  Taichi """
        cell_size = spacing / math.sqrt(3)
        grid_dim = np.ceil((max_corner - min_corner) / cell_size).astype(np.int32)
        
        self.grid_origin[None] = min_corner
        self.grid_size[None] = grid_dim
        self.inv_cell_size[None] = 1.0 / cell_size
        self.bounds_min[None] = min_corner
        self.bounds_max[None] = max_corner
        self.r_min[None] = spacing
        self.r_min_sq[None] = spacing * spacing
        
        self.grid = ti.field(dtype=ti.i32, shape=(grid_dim[0], grid_dim[1], grid_dim[2]))
        self.grid.fill(-1)
        
        self.samples_count[None] = 0
        self.active_count[None] = 0

    @ti.func
    def get_grid_index(self, p):
        """incoordinate"""
        return ti.floor((p - self.grid_origin[None]) * self.inv_cell_size[None]).cast(ti.i32)

    @ti.func
    def is_valid(self, p):
        """checkvalid (check + check)"""
        valid = True
        
        if (p.x < self.bounds_min[None].x or p.x >= self.bounds_max[None].x or
            p.y < self.bounds_min[None].y or p.y >= self.bounds_max[None].y or
            p.z < self.bounds_min[None].z or p.z >= self.bounds_max[None].z):
            valid = False
        
        if valid:
            base_idx = self.get_grid_index(p)
            grid_dims = self.grid_size[None]
            
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        if not valid: break 
                        
                        neighbor_idx = base_idx + ti.Vector([i, j, k])
                        
                        if (neighbor_idx.x >= 0 and neighbor_idx.x < grid_dims.x and
                            neighbor_idx.y >= 0 and neighbor_idx.y < grid_dims.y and
                            neighbor_idx.z >= 0 and neighbor_idx.z < grid_dims.z):
                            
                            pid = self.grid[neighbor_idx]
                            if pid != -1:
                                dist_sq = (self.samples[pid] - p).norm_sqr()
                                if dist_sq < self.r_min_sq[None] - 1e-6:
                                    valid = False
        return valid

    @ti.kernel
    def generate_kernel(self, k_limit: int):
        """
        logic： Kernel complete
        """
        
        start_p = ti.Vector([
            ti.random() * (self.bounds_max[None].x - self.bounds_min[None].x) + self.bounds_min[None].x,
            ti.random() * (self.bounds_max[None].y - self.bounds_min[None].y) + self.bounds_min[None].y,
            ti.random() * (self.bounds_max[None].z - self.bounds_min[None].z) + self.bounds_min[None].z
        ])
        
        self.samples[0] = start_p
        self.samples_count[None] = 1
        
        idx = self.get_grid_index(start_p)
        self.grid[idx] = 0
        
        self.active_indices[0] = 0
        self.active_count[None] = 1
        
        while self.active_count[None] > 0:
            rand_i = int(ti.random() * self.active_count[None])
            p_idx = self.active_indices[rand_i]
            center = self.samples[p_idx]
            
            found = False
            for _ in range(k_limit):
                theta = ti.acos(1 - 2 * ti.random())
                phi = 2 * math.pi * ti.random()
                
                r_min = self.r_min[None]
                r_rand = ti.random() * (7 * r_min**3) + r_min**3 # (2r)^3 - r^3 = 8r^3 - r^3 = 7r^3
                radius = ti.pow(r_rand, 1.0/3.0) 
                
                offset = radius * ti.Vector([
                    ti.sin(theta) * ti.cos(phi),
                    ti.sin(theta) * ti.sin(phi),
                    ti.cos(theta)
                ])
                
                candidate = center + offset
                
                if self.is_valid(candidate):
                    curr_count = self.samples_count[None]
                    if curr_count < self.max_points:
                        self.samples[curr_count] = candidate
                        self.samples_count[None] += 1
                        
                        c_idx = self.get_grid_index(candidate)
                        self.grid[c_idx] = curr_count
                        
                        act_cnt = self.active_count[None]
                        self.active_indices[act_cnt] = curr_count
                        self.active_count[None] += 1
                        
                        found = True
                        break 
            
            if not found:
                last_idx = self.active_count[None] - 1
                self.active_indices[rand_i] = self.active_indices[last_idx]
                self.active_count[None] -= 1

    def run(self, min_corner, max_corner, spacing, k=30, seed=0):
        self.init_grid(min_corner, max_corner, spacing)
        
        try:
            ti.reset_random_seed(seed)
        except:
            pass 
        
        t0 = time.time()
        self.generate_kernel(k)
        ti.sync()
        t1 = time.time()
        
        count = self.samples_count[None]
        print(f"Taichi Kernel Time: {t1-t0:.4f}s | Generated {count} points")
        
        return self.samples.to_numpy()[:count]

def fast_sample(N: int, l_bounds: np.ndarray, u_bounds: np.ndarray, k: int = 30, seed: int = 0):
    lengths = u_bounds - l_bounds
    V = np.prod(lengths)
    # rad = np.cbrt(V / N * 0.7) 
    rad = np.cbrt(math.sqrt(2) * V / N) 
    
    print(f"Target N: {N}, Calculated Spacing: {rad:.4f}")
    
    sampler = PoissonDisk3DFast(max_points=int(N * 1.5)) 
    points = sampler.run(l_bounds, u_bounds, rad, k, seed)
    return points
