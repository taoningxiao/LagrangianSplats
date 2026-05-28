from configs.base import *

expname = "code_render_kernel/recons_scalarsyn"
datadir = "data/smoke/scalarsyn"
half_res = False

Nx, Ny, Nz = 100, 150, 100
lambda_regular = 0.01
sliding_window_size = 10
sliding_window_subsequent_epochs = 100
sim_steps = 2

inflow_ratio = 0.1
insert_ratio = 0.1
inflow_region_min = [0.3, 0.05, 0.3]
inflow_region_max = [0.7, 0.15, 0.7]

gt_prefix = "gt/scalarsyn/density_"
gt_prefix_vel = "gt/scalarsyn/velocity_"

ModelParams.update(dict(
    source_path=datadir,
    half_res=half_res,
))
OptimizationParams.update(dict(
    use_test_in_training=True,
))
