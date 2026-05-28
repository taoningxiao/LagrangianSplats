import os
import json
import torch
import numpy as np
import shutil
from datetime import datetime
import configargparse


def set_device(args):
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is not available")

	if args.deviceID >= torch.cuda.device_count():
		raise ValueError(f"Device ID {args.deviceID} is not available. Only {torch.cuda.device_count()} devices found.")

	device = torch.device(f'cuda:{args.deviceID}')
	torch.cuda.set_device(args.deviceID)

	print(f"Using device: {torch.cuda.get_device_name(args.deviceID)}")
	return device


def get_background_color(dataset, device="cuda"):
	"""
	getbackground color，prefer a custom color，otherwise use white_background logic
	
	Args:
		dataset: ModelParams object，contains white_background  background_color attributes
		device: torch device， "cuda"
	
	Returns:
		background: torch.Tensor，shape (3,)，RGB  [0, 1]
	"""
	if hasattr(dataset, 'background_color') and dataset.background_color is not None:
		bg_color = dataset.background_color
		if isinstance(bg_color, str):
			import ast
			try:
				bg_color = ast.literal_eval(bg_color)
			except:
				bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
		if not isinstance(bg_color, (list, tuple, np.ndarray)):
			bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
		if len(bg_color) != 3:
			bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
		print(f"[Background] Using custom background color: {bg_color}")
	else:
		bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
	return torch.tensor(bg_color, dtype=torch.float32, device=device)


def set_rand_seed(seed):
	import random
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic= True
		torch.backends.cudnn.benchmark = False


def init(args):
	current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	savedir = os.path.join("log", f"{args.expname}_{current_time}")
	os.makedirs(savedir, exist_ok=True)
	os.mkdir(os.path.join(savedir, "images"))
	os.mkdir(os.path.join(savedir, "images/vel"))
	os.mkdir(os.path.join(savedir, "images/vor"))
	os.mkdir(os.path.join(savedir, "images/trans"))
	os.mkdir(os.path.join(savedir, "video"))
	os.mkdir(os.path.join(savedir, "video/vel"))
	os.mkdir(os.path.join(savedir, "video/vor"))
	os.mkdir(os.path.join(savedir, "video/trans"))
	os.mkdir(os.path.join(savedir, "ckpt"))
	os.mkdir(os.path.join(savedir, "backup"))
	if args.config:
		shutil.copy(args.config, os.path.join(savedir, "backup"))
	if os.path.exists('recons_vel_kernel.py'):
		shutil.copy('recons_vel_kernel.py', os.path.join(savedir, "backup"))
	return savedir


def config_parser():
	parser = configargparse.ArgumentParser()
	parser.add_argument('--config', is_config_file=True, 
						help='config file path (text format)')
	parser.add_argument('--configs', type=str, default='',
						help='config file path (Python format, similar to train.py)')
	parser.add_argument("--expname", type=str, 
						help='experiment name')
	parser.add_argument("--basedir", type=str, default='./logs/', 
						help='where to store ckpts and logs')
	parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
						help='input data directory')

	# training options
	parser.add_argument("--lrate", type=float, default=5e-4, 
						help='learning rate')
	parser.add_argument("--fix_seed", type=int, default=42,
						help='the random seed.')
	parser.add_argument("--model_path", type=str, default=None, 
						help='specific weights npy file to reload for coarse network')
	parser.add_argument("--train_vel_grid_size", type=int, default=32,
						help='the random seed.')

	## Stage 1
	parser.add_argument("--stage1_finish_recon", type=int, default=50000, help="stage 1 total training steps" )
	parser.add_argument("--stage2_finish_recon", type=int, default=600000, help="stage 2 total training steps" )
	parser.add_argument("--uniform_sample_step", type=int, default = 20000, help="stage 1 first uniform sample steps" )
	parser.add_argument("--smoke_recon_delay_start", type=int, default=0,
						help='for hybrid models, the step to start learning the temporal dynamic component.')
	parser.add_argument("--smoke_recon_delay_last", type=int, default=10000,
						help='for hybrid models, the step to start learning the temporal dynamic component.')
	parser.add_argument("--sdf_loss_delay", type=int, default=2000,
						help='for hybrid models, the step to start learning the temporal dynamic component.')
	parser.add_argument("--fading_layers", type=int, default=-1,
						help='for siren and hybrid models, the step to finish fading model layers one by one during training.')
	parser.add_argument("--density_distillation_delay", type=int, default=2000, help="stage 2 total training steps" )
	parser.add_argument("--feature_regulization_weight", type=float, default=1e-1)
	parser.add_argument("--density_distillation_weight", type=float, default=1.0)
	parser.add_argument("--optimize_color_mlp", action='store_true', default=False)
 
	# Stage 2
	parser.add_argument("--mapping_frame_range_fading_start", type=int, default=20000, help="frame_range" )
	parser.add_argument("--mapping_frame_range_fading_last", type=int, default=50000, help="frame_range" )
	parser.add_argument("--max_mapping_frame_range", type=int, default=50, help="frame_range" )
	
	parser.add_argument("--stage2_train_vel_interval", type=int, default=1, help="stage 2 total training steps" )
	parser.add_argument('--neus_early_terminated', action = 'store_true')
	parser.add_argument('--neus_larger_lr_decay', action = 'store_true')
	parser.add_argument("--mapping_loss_fading", type=int, default=10000, help="frame_range" )


	# network model
	## lagrangian network
	parser.add_argument('--use_two_level_density', action = 'store_true')
	parser.add_argument("--lagrangian_feature_dim", type=int, default=16, 
						help='Lagrangian feature dimension')   
	
	parser.add_argument("--feature_map_first_omega", type=int, default=30, 
						help='Lagrangian feature dimension')   
	parser.add_argument("--position_map_first_omega", type=int, default=30, 
						help='Lagrangian feature dimension')   
	parser.add_argument("--density_map_first_omega", type=int, default=30, 
						help='Lagrangian feature dimension')   
	parser.add_argument("--density_activation", type=str,
						default='identity', help='activation function for density')
	parser.add_argument("--lagrangian_density_activation", type=str,
						default='softplus', help='activation function for density')
	
	## siren nerf    
	parser.add_argument("--siren_nerf_netdepth", type=int, default=8, 
						help='layers in network')
	parser.add_argument("--siren_nerf_first_omega", type=int, default=30, 
						help='layers in network')
	
	## neus
	parser.add_argument('--use_scene_scale_before_pe', action = 'store_true')
	parser.add_argument('--neus_progressive_pe', action = 'store_true')
	parser.add_argument('--neus_progressive_pe_min_mask', type=float, default=0.5)
	parser.add_argument('--neus_progressive_pe_start', type=int, default=20000)
	parser.add_argument('--neus_progressive_pe_duration', type=int, default=10000)

	# loss hyper params, negative values means to disable the loss terms
	parser.add_argument("--vgg_strides", type=int, default=4,
						help='vgg stride, should >= 2')
	parser.add_argument("--ghostW", type=float,
						default=-0.0, help='weight for the ghost density regularization')
	parser.add_argument("--vggW", type=float,
						default=-0.0, help='weight for the VGG loss')
	parser.add_argument("--ColorDivergenceW", type=float,
						default=0.0, help='weight for the VGG loss')
	parser.add_argument("--smokeMaskW", type=float,
						default=0.0, help='weight for the VGG loss')
	parser.add_argument("--smokeOverlayW", type=float,
						default=0.0, help='weight for the VGG loss')
	parser.add_argument("--overlayW", type=float,
						default=-0.0, help='weight for the overlay regularization')
	parser.add_argument("--d2vW", type=float,
						default=-0.0, help='weight for the d2v loss')
	parser.add_argument("--nseW", type=float,
						default=0.01, help='velocity model, training weight for the physical equations')
	parser.add_argument("--ekW", type=float,
						default=0.0, help='weight for the Ekinoal loss')
	parser.add_argument("--boundaryW", type=float,
						default=0.5, help='weight for the Boardary constrain loss')
	parser.add_argument("--hardW", type=float,
						default=0.0, help='weight for the Boardary constrain loss')
	parser.add_argument("--MinusDensityW", type=float,
						default=0.0, help='weight for the Boardary constrain loss')
	parser.add_argument("--SmokeInsideSDFW", type=float,
						default=0.5, help='weight for the Boardary constrain loss')
	parser.add_argument("--SmokeAlphaReguW", type=float,
						default=0.05, help='weight for the Boardary constrain loss')
	parser.add_argument("--SmokeAlphaReguW_warmup", type=float,
						default=0.05, help='weight for the Boardary constrain loss')
	parser.add_argument("--CurvatureW", type=float,
						default=0.00, help='weight for the Boardary constrain loss')
	parser.add_argument("--train_vel_within_rendering", action='store_true')
	parser.add_argument("--train_vel_uniform_sample", type=int, default = 2)
	parser.add_argument("--inside_sdf", type=float, default = 0.0)
	parser.add_argument("--vel_regulization_weight", type=float,
						default=0.1, help='weight for the Boardary constrain loss')
	parser.add_argument("--coarse_transport_weight", type=float,
						default=10, help='weight for the Boardary constrain loss')
	parser.add_argument("--fine_transport_weight", type=float,
						default=0.1, help='weight for the Boardary constrain loss')
	parser.add_argument("--feature_transport_weight", type=float,
						default=1, help='weight for the Boardary constrain loss')
	## Lagrangian Feature loss
	parser.add_argument("--self_cycle_loss_weight", type=float, default = 1.0)
	parser.add_argument("--cross_cycle_loss_weight", type=float, default = 1.0)
	
	## Lagrangian mapping loss
	parser.add_argument("--density_mapping_loss_weight", type=float, default = 0.01)
	parser.add_argument("--velocity_mapping_loss_weight", type=float, default = 0.01)
	parser.add_argument("--color_mapping_loss_weight", type=float, default = 0.001)


	parser.add_argument("--net_model", type=str, default='nerf',
						help='which model to use, nerf, siren, hybrid..')
	parser.add_argument("--netdepth", type=int, default=8, 
						help='layers in network')
	parser.add_argument("--netwidth", type=int, default=256, 
						help='channels per layer')
	parser.add_argument("--netdepth_fine", type=int, default=8, 
						help='layers in fine network')
	parser.add_argument("--netwidth_fine", type=int, default=256, 
						help='channels per layer in fine network')
	parser.add_argument("--N_rand", type=int, default=32*32*4, 
						help='batch size (number of random rays per gradient step)')
	parser.add_argument("--lrate_decay", type=int, default=250, 
						help='exponential learning rate decay (in 1000 steps)')
	parser.add_argument("--chunk", type=int, default=1024*32, 
						help='number of rays processed in parallel, decrease if running out of memory')
	parser.add_argument("--training_ray_chunk", type=int, default=1024*32, 
						help='number of rays processed in parallel, decrease if running out of memory')
	parser.add_argument("--test_chunk", type=int, default=1024*4, 
						help='number of rays processed in parallel, decrease if running out of memory')
	parser.add_argument("--netchunk", type=int, default=1024*64, 
						help='number of pts sent through network in parallel, decrease if running out of memory')
	parser.add_argument("--no_batching", action='store_true', 
						help='only take random rays from 1 image at a time')
	parser.add_argument("--no_reload", action='store_true', 
						help='do not reload weights from saved ckpt')
						
	parser.add_argument("--tempo_delay", type=int, default=0,
						help='for hybrid models, the step to start learning the temporal dynamic component.')
	parser.add_argument("--vgg_delay", type=int, default=0,
						help='for hybrid models, the step to start learning the temporal dynamic component.')
	parser.add_argument("--vel_delay", type=int, default=10000,
						help='for siren and hybrid models, the step to start learning the velocity.')
	parser.add_argument("--boundary_delay", type=int, default=10000,
						help='for siren and hybrid models, the step to start learning the velocity.')
	parser.add_argument("--N_iter", type=int, default=200000,
						help='for siren and hybrid models, the step to start learning the velocity.')  
	parser.add_argument("--adaptive_num_rays", action='store_true')
	parser.add_argument("--target_batch_size", type=int, default=2**17)

	# scene options
	parser.add_argument("--scene_scale", type=float, default = 1.0)
	parser.add_argument("--bbox_min", type=str,
						default='0.0,0.0,0.0', help='use a boundingbox, the minXYZ')
	parser.add_argument("--bbox_max", type=str,
						default='1.0,1.0,1.0', help='use a boundingbox, the maxXYZ')
	parser.add_argument("--occ_grid_bound_static", type=float, default = 1.0)
	parser.add_argument("--occ_grid_bound_dynamic", type=float, default = 1.0)
	


	# task params
	parser.add_argument("--test_mode", action='store_true', 
						help='test mode')
	parser.add_argument("--output_voxel", action='store_true', 
						help='do not optimize, reload weights and output volumetric density and velocity')
	parser.add_argument("--voxel_video", action='store_true', 
						help='do not optimize, reload weights and output volumetric density and velocity')
	parser.add_argument("--visualize_mapping", action='store_true', 
						help='do not optimize, reload weights and output volumetric density and velocity')
	parser.add_argument("--visualize_feature", action='store_true', 
						help='do not optimize, reload weights and output volumetric density and velocity')
	parser.add_argument("--vol_output_W", type=int, default=256, 
						help='In output mode: the output resolution along x; In training mode: the sampling resolution for training')
	parser.add_argument("--full_vol_output", action='store_true', 
						help='do not optimize, reload weights and output volumetric density and velocity')
	parser.add_argument("--save_jacobian_den", action='store_true') 
	
	
	
	parser.add_argument("--vol_output_only", action='store_true', 
						help='do not optimize, reload weights and output volumetric density and velocity')
	parser.add_argument("--masked_vol_otuput", action='store_true', 
						help='do not optimize, reload weights and output volumetric density and velocity')
	parser.add_argument("--render_only", action='store_true', 
						help='do not optimize, reload weights and render out render_poses path')
	parser.add_argument("--mesh_only", action='store_true', 
						help='do not optimize, reload weights and render out render_poses path')
	parser.add_argument("--render_test", action='store_true', 
						help='render the test set instead of render_poses path')
	parser.add_argument("--render_eval", action='store_true', 
						help='render the test set instead of render_poses path')
	parser.add_argument("--render_train", action='store_true', 
						help='render the training set instead of render_poses path')
	parser.add_argument("--render_vis", action='store_true', 
						help='render the training set instead of render_poses path')
	parser.add_argument("--render_2d_trajectory", action='store_true') 
	parser.add_argument("--render_trajectory_only", action='store_true', 
						help='render the training set instead of render_poses path')
	parser.add_argument("--render_no_vorticity", action='store_true', 
						help='render the training set instead of render_poses path')
	parser.add_argument("--vis_view", type=int, default=0)
	
	parser.add_argument("--preload_gt_den_vol", action='store_true', 
						help='render the training set instead of render_poses path')
	parser.add_argument("--vis_feature", action='store_true', 
						help='render the training set instead of render_poses path')
	parser.add_argument("--n_keyframes", type=int, default=0,
						help='number of intermediate keyframes for light training mode (default: 0)')
	parser.add_argument("--keyframe_ckpt_dir", type=str, default=None,
						help='pre-trained keyframe checkpoint root directory (optional). If provided, will load keyframe models from this directory first. Directory structure should be: keyframe_ckpt_dir/gaussian_keyframe_{k_idx}/... (default: None)')
   
	# rendering options
	parser.add_argument("--N_samples", type=int, default=64, 
						help='number of coarse samples per ray')
	parser.add_argument("--N_importance", type=int, default=0,
						help='number of additional fine samples per ray')
	parser.add_argument("--perturb", type=float, default=1.,
						help='set to 0. for no jitter, 1. for jitter')
	parser.add_argument("--use_viewdirs", action='store_true', 
						help='use full 5D input instead of 3D')
	parser.add_argument("--i_embed", type=int, default=0, 
						help='set 0 for default positional encoding, -1 for none')
	parser.add_argument("--multires", type=int, default=10, 
						help='log2 of max freq for positional encoding (3D location)')
	parser.add_argument("--multires_views", type=int, default=4, 
						help='log2 of max freq for positional encoding (2D direction)')
	parser.add_argument("--i_embed_neus", type=int, default=0, 
						help='set 0 for default positional encoding, -1 for none')
	parser.add_argument("--multires_neus", type=int, default=10, 
						help='log2 of max freq for positional encoding (3D location)')
	parser.add_argument("--multires_smoke", type=int, default=6, 
						help='log2 of max freq for positional encoding (3D location)')
	parser.add_argument("--multires_views_neus", type=int, default=4, 
						help='log2 of max freq for positional encoding (2D direction)')
	parser.add_argument("--raw_noise_std", type=float, default=0., 
						help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
	parser.add_argument("--render_factor", type=int, default=0, 
						help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
	
	# cuda_ray
	parser.add_argument("--cuda_ray", action='store_true', 
						help='sampling linearly in disparity rather than depth')
	parser.add_argument("--time_size", type=int, default=150, 
						help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
	parser.add_argument("--density_thresh", type=float, default=1.0, 
						help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
	parser.add_argument("--density_thresh_static", type=float, default=30.0, 
						help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
	parser.add_argument("--use_triplane_occ_grid", action='store_true', 
						help='sampling linearly in disparity rather than depth')

	# NeuS rendering options
	parser.add_argument("--up_sample_steps", type=int, default=4, 
						help='number of up samples per ray')
	parser.add_argument("--anneal_end", type=int, default=50000, 
						help='number of up samples per ray')
	
	# Network options
	parser.add_argument('--use_neus2_network', action = 'store_true')
	parser.add_argument('--swish_network', action = 'store_true')
	parser.add_argument('--disentangled_density_color', action = 'store_true')
	parser.add_argument('--density_init_zero', action = 'store_true')


	# training options
	parser.add_argument("--precrop_iters", type=int, default=0,
						help='number of steps to train on central crops')
	parser.add_argument("--precrop_frac", type=float,
						default=.5, help='fraction of img taken for central crops') 
	parser.add_argument("--neus_early_termination", type=int, default=-1,
						help='number of steps to train on central crops')
	parser.add_argument("--lagrangian_warmup", type=int, default=10000,
						help='number of steps to train on central crops')

	# dataset options
	parser.add_argument("--dataset_type", type=str, default='llff', 
						help='options: llff / blender / deepvoxels')
	parser.add_argument("--testskip", type=int, default=8, 
						help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
	parser.add_argument("--trainskip", type=int, default=1, 
						help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

	## deepvoxels flags
	parser.add_argument("--shape", type=str, default='greek', 
						help='options : armchair / cube / greek / vase')

	## blender flags
	parser.add_argument("--white_bkgd", action='store_true', 
						help='set to render synthetic data on a given bkgd (always use for dvoxels)')
	parser.add_argument("--half_res", type=str, default='normal', 
						help='load blender synthetic data at 400x400 instead of 800x800')

	## llff flags
	parser.add_argument("--factor", type=int, default=8, 
						help='downsample factor for LLFF images')
	parser.add_argument("--no_ndc", action='store_true', 
						help='do not use normalized device coordinates (set for non-forward facing scenes)')
	parser.add_argument("--lindisp", action='store_true', 
						help='sampling linearly in disparity rather than depth')
	parser.add_argument("--spherify", action='store_true', 
						help='set for spherical 360 scenes')
	parser.add_argument("--llffhold", type=int, default=8, 
						help='will take every 1/N images as LLFF test set, paper uses 8')

	# logging/saving options
	parser.add_argument("--i_print",   type=int, default=400, 
						help='frequency of console printout and metric loggin')
	parser.add_argument("--i_img",     type=int, default=2000, 
						help='frequency of tensorboard image logging')
	parser.add_argument("--i_weights", type=int, default=25000, 
						help='frequency of weight ckpt saving')
	parser.add_argument("--i_testset", type=int, default=50000, 
						help='frequency of testset saving')
	parser.add_argument("--i_video",   type=int, default=200000, 
						help='frequency of render_poses video saving')
	parser.add_argument("--i_visualize",   type=int, default=10000, 
						help='frequency of render_poses video saving')
	
	# my options
	parser.add_argument("--occ_dynamic_path", type=str, default=None)
	parser.add_argument("--occ_static_path", type=str, default=None)
	
	parser.add_argument("--lambda_vel", type=float, default=0.01)
	parser.add_argument("--long_term_len", type=int, default=5)
	parser.add_argument("--long_term_beta", type=float, default=0.95)
	parser.add_argument("--use_ours", action='store_true')
	parser.add_argument("--long_term", action="store_true")
	parser.add_argument("--lambda_regular", type=float, default=1.0)
	parser.add_argument("--lambda_nse", type=float, default=1e-6)
	parser.add_argument("--lambda_vort", type=float, default=0.1)
	parser.add_argument("--lambda_radius", type=float, default=0.0, help="Regularization weight for kernel radius (L2 penalty)")
	parser.add_argument("--lambda_ingp", type=float, default=1)
	
	parser.add_argument("--Nx", type=int, default=64)
	parser.add_argument("--Ny", type=int, default=64)
	parser.add_argument("--Nz", type=int, default=64)
	parser.add_argument("--load_path", type=str, default=None)
	parser.add_argument("--ckpt_load_path", type=str, default=None)
	parser.add_argument("--gaussian_ckpt_path", type=str, default=None)
	parser.add_argument("--skip_train", action="store_true", default=False)
	parser.add_argument("--skip_visualize", action="store_true", default=False)
	parser.add_argument("--skip_render", action="store_true", default=False)
	parser.add_argument("--quick_test", action="store_true", default=False)
	parser.add_argument("--max_windows", type=int, default=None)
	parser.add_argument("--eval_frame_limit", type=int, default=None)
	parser.add_argument("--siren_model_path", type=str, default=None)
	
	parser.add_argument("--in_dim", type=int, default=4)
	parser.add_argument("--out_dim", type=int, default=3)
	parser.add_argument("--hidden_layers", type=int, default=6)
	parser.add_argument("--hidden_dim", type=int, default=128)
	
	parser.add_argument("--lrate_vel", type=float, default=1e-3)
	parser.add_argument("--lrate_vort", type=float, default=1e-3)
	parser.add_argument("--lrate_ingp", type=float, default=1e-3)
	
	parser.add_argument("--n_particles", type=int, default=100)
	parser.add_argument("--vort_path", type=str, default="None")
	parser.add_argument("--vort_intensity", type=float, default=20)
	
	parser.add_argument("--batchsize", type=int, default=100000)
	parser.add_argument("--num_epochs", type=int, default=20)
	
	parser.add_argument("--finest_resolution_v", type=int, default=128)
	parser.add_argument("--finest_resolution_v_t", type=int, default=128)
	parser.add_argument("--base_resolution_v", type=int, default=16)
	parser.add_argument("--base_resolution_v_t", type=int, default=16)
	parser.add_argument("--finest_resolution", type=int, default=128)
	parser.add_argument("--finest_resolution_t", type=int, default=128)
	parser.add_argument("--base_resolution", type=int, default=16)
	parser.add_argument("--base_resolution_t", type=int, default=16)
	parser.add_argument("--num_levels", type=int, default=16)
	parser.add_argument("--log2_hashmap_size", type=int, default=19)
	parser.add_argument("--vel_num_layers", type=int, default=2)
	parser.add_argument("--use_f", action='store_true', default=False)
	
	parser.add_argument("--gt_prefix", type=str, default=None)
	parser.add_argument("--gt_ext", type=str, default=None)
	parser.add_argument("--gt_prefix_vel", type=str, default=None)
	parser.add_argument("--gt_ext_vel", type=str, default=None)
	
	parser.add_argument("--vel_color", type=int, default=1000)
	parser.add_argument("--vor_color", type=int, default=1500)
	
	parser.add_argument("--vis_vel", action="store_true")
	parser.add_argument("--resim", action="store_true")
	parser.add_argument("--pred", action="store_true")
	
	parser.add_argument("--lambda_proj", type=float, default=1.0)
	
	parser.add_argument("--proj_interval", type=int, default=5)
	parser.add_argument("--proj_size", type=int, default=128)

	parser.add_argument("--y_start", type=int, default=48)
	parser.add_argument("--y_proj", type=int, default=128)
	
	parser.add_argument("--frame_start", type=int, default=89)
	parser.add_argument("--frame_duration", type=int, default=30)
	
	parser.add_argument('--mask', action = 'store_true')
	parser.add_argument('--dens_thresh', type=float, default=0)
	
	parser.add_argument('--inflow_dir', type=str, default=None)
	
	parser.add_argument('--render_pred', action = 'store_true')
	parser.add_argument('--pred_path', type=str, default=None)
	
	parser.add_argument('--render_interval', type=float, default=1.0)
	
	parser.add_argument('--sim_steps', type=float, default=1)

	parser.add_argument('--deviceID', type=int, default=0)
	parser.add_argument('--kernel_num', type=int, default=1000000)
	parser.add_argument('--fit_mode', type=str, default='temporal', choices=['temporal', 'per_frame', 'train_per_frame', 'train_vel_kernel', 'load_and_visualize'], 
						help='Fitting mode: temporal (temporal model), per_frame (per-frame fitting), train_per_frame (per-frame continuity-loss training), train_vel_kernel (long-range kernel training), load_and_visualize (load and visualize)')
	parser.add_argument('--i_draw', type=int, default=5)
	parser.add_argument('--i_save', type=int, default=10)
	parser.add_argument('--ckpt_num', type=int, default=500, 
						help='Checkpoint number to load (e.g., 500 for ckpt_000500)')
	parser.add_argument('--model_load_template', type=str, default=None,
						help='Template for model path. If None, uses load_path/ckpt/velrbf_frame_{frame_idx:03d}_ckpt_{ckpt_num:06d}.pth')

	parser.add_argument('--lambda_far_zero', type=float, default=1.0,
						help='weight for far-plane density penalty')
	parser.add_argument('--far_penalty_ratio', type=float, default=0.1,
						help='ratio of samples near far plane to penalize (default: 0.1 for 10%%)')
	
	return parser
