from configs.base import *

expname = "code_render_kernel/recons_scalarreal"
datadir = "data/smoke/scalarreal"
half_res = "quarter"

Nx, Ny, Nz = 128, 192, 128
lambda_regular = 0.01
n_keyframes = 10
sliding_window_size = 10
sliding_window_subsequent_epochs = 100
sim_steps = 2

inflow_ratio = 0.1
insert_ratio = 0.1
visualize_inflow_region = False
inflow_region_min = [0.3, 0.03, 0.3]
inflow_region_max = [0.7, 0.1, 0.7]

# Scalarreal does not ship with default GT in this release.
gt_prefix = None
gt_prefix_vel = None
gt_ext = None
gt_ext_vel = None

ModelParams.update(dict(
    source_path=datadir,
    half_res=half_res,
))
OptimizationParams.update(dict(
    use_test_in_training=True,
))
