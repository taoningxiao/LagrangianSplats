from configs.base import *

expname = "code_render_kernel/recons_suzanne"
datadir = "data/smoke/suzanne"
half_res = True
background_color = [0.0, 0.0, 0.0]

Nx, Ny, Nz = 128, 128, 128
n_keyframes = 10
sliding_window_size = 5
sliding_window_subsequent_epochs = 100
sim_steps = 1

inflow_ratio = 0.0
insert_ratio = 0.01
visualize_inflow_region = True
inflow_region_min = [0.425, 0.048, 0.425]
inflow_region_max = [0.575, 0.085, 0.575]

gt_prefix = "gt/suzanne/density_"
gt_prefix_vel = "gt/suzanne/velocityGrid_"

ModelParams.update(dict(
    source_path=datadir,
    half_res=half_res,
))
