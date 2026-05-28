import torch
import numpy as np
from velocity_common.utils import config_parser, init, set_rand_seed
from velocity_training.train import train_velocity_model_sliding_window
from velocity_evaluation.visualize import load_and_visualize_vel
from velocity_evaluation.evaluate import evaluate_full_reconstruction


if __name__ == "__main__":
	torch.set_default_dtype(torch.float32)
	if hasattr(torch, 'set_default_device'):
		torch.set_default_device('cuda')
	else:
		torch.set_default_tensor_type('torch.cuda.FloatTensor')

	parser = config_parser()
	args = parser.parse_args()
	
	if args.configs:
		if args.config is None:
			args.config = args.configs

		try:
			import mmcv
			config = mmcv.Config.fromfile(args.configs)
			
			cfg_dict = {}
			if isinstance(config, dict):
				cfg_dict = config
			else:
				for key in config:
					cfg_dict[key] = config[key]

			gs_param_groups = ['ModelParams', 'OptimizationParams', 'PipelineParams', 'ModelHiddenParams']

			for key, value in cfg_dict.items():
				setattr(args, key, value)
				
				if key in gs_param_groups and isinstance(value, dict):
					print(f"[Config] Flattening parameter group: {key}")
					for sub_key, sub_value in value.items():
						setattr(args, sub_key, sub_value)
						
			print(f"Loaded and flattened configuration from {args.configs}")
			
		except ImportError:
			print("Warning: mmcv not found. Cannot load Python config file.")
		except Exception as e:
			print(f"Warning: Failed to load config from {args.configs}: {e}")
			import traceback
			traceback.print_exc()
			exit(1)

	if getattr(args, 'quick_test', False):
		args.num_epochs = min(getattr(args, 'num_epochs', 1), 1)
		args.sliding_window_subsequent_epochs = 1
		args.sliding_window_size = min(getattr(args, 'sliding_window_size', 2), 2)
		args.frame_zero_iterations = min(getattr(args, 'frame_zero_iterations', 20), 20)
		args.coarse_iterations = min(getattr(args, 'coarse_iterations', 20), 20)
		args.i_draw = max(getattr(args, 'i_draw', 100), 100)
		args.i_save = max(getattr(args, 'i_save', 100), 100)
		args.max_windows = 1 if getattr(args, 'max_windows', None) is None else args.max_windows
		args.eval_frame_limit = 2 if getattr(args, 'eval_frame_limit', None) is None else args.eval_frame_limit

	set_rand_seed(args.fix_seed)
	savedir = init(args)

	bkg_flag = args.white_bkgd
	args.white_bkgd = np.ones([3], dtype=np.float32) if bkg_flag else None
	args.test_mode = True

	train_scale = getattr(args, 'scale', 2.0)
	ckpt_load_path = getattr(args, 'ckpt_load_path', None) or getattr(args, 'load_path', None) or savedir
	gaussian_ckpt_path = getattr(args, 'gaussian_ckpt_path', None)
	
	inflow_ratio = getattr(args, 'inflow_ratio', 0.15)
	insert_ratio = getattr(args, 'insert_ratio', 0.01)
	visualize_inflow_region = getattr(args, 'visualize_inflow_region', False)
	start_frame = getattr(args, 'start_frame', None)
	end_frame = getattr(args, 'end_frame', None)
	sliding_window_size = getattr(args, 'sliding_window_size', 10)
	
	if not getattr(args, 'skip_train', False):
		train_velocity_model_sliding_window(
			args, savedir=savedir, scale=train_scale,
			gaussian_ckpt_path=gaussian_ckpt_path,
			inflow_ratio=inflow_ratio, insert_ratio=insert_ratio,
			w=sliding_window_size,
		)
	
	if not getattr(args, 'skip_visualize', False):
		load_and_visualize_vel(
			args, savedir=savedir, load_path=ckpt_load_path,
			save_velocity_vtk=getattr(args, 'save_velocity_vtk', False),
		)
	
	if not getattr(args, 'skip_render', False):
		evaluate_full_reconstruction(args, savedir=savedir, scale=train_scale, ckpt_load_path=ckpt_load_path)
