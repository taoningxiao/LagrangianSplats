from configs.base import *

expname = "code_render_kernel/recons_biplume"
datadir = "data/smoke/biplume"
half_res = True
background_color = [0.7686, 0.7686, 0.7686]

Nx, Ny, Nz = 160, 320, 160
n_keyframes = 5
num_epochs = 600
i_draw = 200
i_save = 200
sliding_window_size = 5
sliding_window_subsequent_epochs = 200
sim_steps = 1

inflow_ratio = 0.0
insert_ratio = 0.1
visualize_inflow_region = False
inflow_region_min = [0.4, 0.05, 0.4]
inflow_region_max = [0.6, 0.15, 0.6]

gt_prefix = "gt/biplume/density_"
gt_prefix_vel = "gt/biplume/velocityGrid_"

ModelParams.update(dict(
    source_path=datadir,
    half_res=half_res,
))
OptimizationParams.update(dict(
    use_test_in_training=True,
))
