from argparse import ArgumentParser, Namespace
from typing import Optional

from arguments import ModelHiddenParams, ModelParams
from scene import GaussianModel, Scene


def load_gaussian_model(
	model_path: str,
	data_dir: str,
	load_iteration: Optional[int] = None,
	load_only_xyz: bool = False,
	base_args=None,
):
	"""Load a trained GaussianModel and its Scene metadata."""
	parser = ArgumentParser()
	model_params_obj = ModelParams(parser, sentinel=True)
	hyperparam_obj = ModelHiddenParams(parser)

	args = Namespace()
	if base_args is not None:
		for key, value in vars(base_args).items():
			setattr(args, key, value)

	args.model_path = model_path
	args.source_path = data_dir

	defaults = {
		"sh_degree": 3,
		"images": "images",
		"resolution": -1,
		"white_background": True,
		"data_device": "cuda",
		"eval": True,
		"render_process": False,
		"add_points": False,
		"extension": ".png",
		"llffhold": 8,
		"net_width": 64,
		"timebase_pe": 4,
		"defor_depth": 1,
		"posebase_pe": 10,
		"scale_rotation_pe": 2,
		"opacity_pe": 2,
		"timenet_width": 64,
		"timenet_output": 32,
		"bounds": 1.6,
		"plane_tv_weight": 0.0001,
		"time_smoothness_weight": 0.01,
		"l1_time_planes": 0.0001,
		"kplanes_config": {
			"grid_dimensions": 2,
			"input_coordinate_dim": 4,
			"output_coordinate_dim": 32,
			"resolution": [64, 64, 64, 25],
		},
		"multires": [1, 2, 4, 8],
		"no_dx": False,
		"no_grid": False,
		"no_ds": False,
		"no_dr": False,
		"no_do": True,
		"no_dshs": True,
		"empty_voxel": False,
		"grid_pe": 0,
		"static_mlp": False,
		"apply_rotation": False,
		"use_velocity_advection": False,
		"velocity_num_time_steps": 30,
		"velocity_grid_resolution": [75, 150, 75],
		"use_velocity_kernel": False,
		"kernel_num": 1000,
		"velocity_kernel_num_time_steps": 30,
		"velocity_kernel_grid_resolution": [100, 150, 100],
	}
	for key, value in defaults.items():
		if not hasattr(args, key):
			setattr(args, key, value)

	try:
		model_params = model_params_obj.extract(args)
		hyperparam = hyperparam_obj.extract(args)

		print(f"Creating GaussianModel with sh_degree={model_params.sh_degree}...")
		gaussians = GaussianModel(model_params.sh_degree, hyperparam)
		print("GaussianModel created successfully")

		print(
			f"Creating Scene with model_path={model_path}, "
			f"data_dir={data_dir}, load_iteration={load_iteration}..."
		)
		scene = Scene(
			model_params,
			gaussians,
			load_iteration=load_iteration,
			shuffle=False,
			load_only_xyz=load_only_xyz,
		)
		print("Scene created successfully")

		print(f"Loaded Gaussian model from {model_path} at iteration {scene.loaded_iter}")
		if hasattr(gaussians, "get_xyz") and gaussians.get_xyz is not None:
			print(f"Total number of Gaussians: {gaussians.get_xyz.shape[0]}")
		else:
			print("Warning: gaussians.get_xyz is None or not accessible")

		return gaussians, scene
	except Exception as exc:
		print(f"Error in load_gaussian_model: {exc}")
		import traceback
		traceback.print_exc()
		raise
