basedir = "./log"
dataset_type = "pinf_data"
white_bkgd = True

# Simulation grid and renderer scale.
scene_scale = 1.0
scale = 2.0
batchsize = 1780000

# Velocity model and sliding-window optimization.
kernel_num = 100000
lrate_vel = 0.1
num_epochs = 500
sliding_window_size = 10
sliding_window_subsequent_epochs = 100
i_draw = 100
i_save = 100

lambda_regular = 0.005
lambda_nse = 0.1

# Inflow Gaussians.
inflow_ratio = 0.1
insert_ratio = 0.1
visualize_inflow_region = False

# Evaluation and visualization.
gt_ext = ".npz"
gt_ext_vel = ".npz"
save_velocity_vtk = False
vel_color = 300
vor_color = 1500

sim_steps = 2

frame_zero_iterations = 3000

ModelParams = dict(
    white_background=False,
    sh_degree=3,
    images="images",
    resolution=-1,
    data_device="cuda",
    extension=".png",
    llffhold=8,
    num_init_points=2000,
)

OptimizationParams = dict(
    iterations=30000,
    coarse_iterations=3000,
    position_lr_init=0.00016,
    position_lr_final=0.0000016,
    position_lr_delay_mult=0.01,
    position_lr_max_steps=20000,
    feature_lr=0.0025,
    opacity_lr=0.05,
    scaling_lr=0.005,
    rotation_lr=0.001,
    deformation_lr_init=0.00016,
    deformation_lr_final=0.000016,
    deformation_lr_delay_mult=0.01,
    grid_lr_init=0.0016,
    grid_lr_final=0.00016,
    percent_dense=0.01,
    lambda_dssim=0.0,
    lambda_spherical=10.0,
    densification_interval=100,
    densify_from_iter=500,
    densify_until_iter=15000,
    densify_grad_threshold_coarse=0.0002,
    pruning_from_iter=500,
    pruning_interval=100,
    opacity_threshold_coarse=0.005,
    opacity_reset_interval=3000,
    batch_size=1,
    add_point=False,
    visualize_interval=3000,
    visualize_pointcloud_interval=3000,
    use_test_in_training=False,
    velocity_kernel_lr_init=0.0016,
    velocity_kernel_lr_final=0.00016,
)

ModelHiddenParams = dict(
    net_width=64,
    timebase_pe=4,
    defor_depth=1,
    posebase_pe=10,
    scale_rotation_pe=2,
    opacity_pe=2,
    timenet_width=64,
    timenet_output=32,
    bounds=1.6,
    plane_tv_weight=0.0001,
    time_smoothness_weight=0.01,
    l1_time_planes=0.0001,
    kplanes_config=dict(
        grid_dimensions=2,
        input_coordinate_dim=4,
        output_coordinate_dim=32,
        resolution=[64, 64, 64, 25],
    ),
    multires=[1, 2, 4, 8],
    no_dx=False,
    no_grid=False,
    no_ds=False,
    no_dr=False,
    no_do=True,
    no_dshs=True,
    empty_voxel=False,
    grid_pe=0,
    static_mlp=False,
    apply_rotation=False,
    use_velocity_advection=False,
    velocity_num_time_steps=30,
    velocity_grid_resolution=[75, 150, 75],
    use_velocity_kernel=False,
    velocity_kernel_num_time_steps=30,
    velocity_kernel_grid_resolution=[100, 150, 100],
)

PipelineParams = dict(
    convert_SHs_python=False,
    compute_cov3D_python=False,
    debug=False,
)
