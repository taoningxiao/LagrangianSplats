from configs.base import *

expname = "code_render_kernel/recons_sphere"
datadir = "data/smoke/sphere"
half_res = True
background_color = [0.0, 0.0, 0.0]

Nx, Ny, Nz = 128, 128, 128
n_keyframes = 5
sliding_window_size = 10
sliding_window_subsequent_epochs = 100
sim_steps = 2

inflow_ratio = 0.1
insert_ratio = 0.1
visualize_inflow_region = True
inflow_region_min = [0.35, 0.08, 0.35]
inflow_region_max = [0.65, 0.12, 0.65]

gt_prefix = "gt/sphere/density_"
gt_prefix_vel = "gt/sphere/velocityGrid_"

ModelParams.update(dict(
    source_path=datadir,
    half_res=half_res,
))
OptimizationParams.update(dict(
    use_test_in_training=True,
))
