import sys
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.timer import Timer
from utils.scene_utils import render_training_image
from utils.pointcloud_vis import visualize_pointcloud_distribution
from random import randint
import copy
import torch
import os
import numpy as np
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def get_background_color(dataset, device="cuda"):
    """Return the configured RGB background color as a CUDA tensor."""
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


def combine_train_test_datasets(train_cams, test_cams):
    """Convert train/test camera containers to lists and concatenate them."""
    if isinstance(train_cams, list):
        train_list = train_cams
    else:
        train_list = [train_cams[i] for i in range(len(train_cams))]
    
    if isinstance(test_cams, list):
        test_list = test_cams
    else:
        test_list = [test_cams[i] for i in range(len(test_cams))]
    
    combined_list = train_list + test_list
    return combined_list


def train_specific_frame_gaussian(args, target_frame_idx, model_path=None, source_path=None, iterations=None, saving_iterations=None):
    """Train the coarse Gaussian model for one target frame."""
    from argparse import ArgumentParser
    
    if model_path is None:
        model_path = getattr(args, 'model_path', None)
        if model_path is None:
            basedir = getattr(args, 'basedir', './log')
            expname = getattr(args, 'expname', f'gaussian_frame_{target_frame_idx}')
            model_path = os.path.join(basedir, expname)
            
    if source_path is None:
        source_path = getattr(args, 'datadir', getattr(args, 'source_path', None))
    source_path = os.path.abspath(source_path)
    
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    hyper = hp.extract(args)
    
    # Fill defaults for parameters that are not present in Python scene configs.
    for target, source in [(dataset, lp), (opt, op), (pipe, pp), (hyper, hp)]:
        for key, value in vars(source).items():
            if isinstance(value, ArgumentParser): continue
            
            attr_name = key[1:] if key.startswith("_") else key
            
            if not hasattr(target, attr_name):
                setattr(target, attr_name, value)
    
    dataset.model_path = model_path
    dataset.source_path = source_path
    
    if iterations is None:
        iterations = opt.coarse_iterations
        
    if saving_iterations is None:
        saving_iterations = [iterations]

    print(f"\n[Frame {target_frame_idx} Gaussian Training]")
    os.makedirs(model_path, exist_ok=True)
    
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, gaussians, load_iteration=None, load_coarse=None, load_only_xyz=False)
    gaussians.training_setup(opt)
    timer = Timer()
    timer.start()
    
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    
    viewpoint_stack = []
    
    frame_num = getattr(args, 'frame_num', 300)
    
    target_time_approx = target_frame_idx / max(1, frame_num - 1)
    time_epsilon = 0.5 / max(1, frame_num - 1)
    
    all_cams = train_cams + (test_cams if opt.use_test_in_training else [])
    
    for cam in all_cams:
        if abs(cam.time - target_time_approx) < time_epsilon:
            viewpoint_stack.append(cam)
            
    if len(viewpoint_stack) == 0:
        print(f"Warning: No cameras found by time matching for frame {target_frame_idx}. Using index matching if applicable.")
        raise ValueError(f"No cameras found for frame {target_frame_idx}!")

    print(f"  Found {len(viewpoint_stack)} cameras for frame {target_frame_idx}")
    temp_list = copy.deepcopy(viewpoint_stack)
    
    test_cams_original = scene.getTestCameras()
    test_viewpoint_stack = []
    for cam in test_cams_original:
        if abs(cam.time - target_time_approx) < time_epsilon:
            test_viewpoint_stack.append(cam)
    test_temp_list = copy.deepcopy(test_viewpoint_stack)
    
    print(f"  Training cameras (Frame {target_frame_idx}): {len(viewpoint_stack)}")
    print(f"  Test cameras (Frame {target_frame_idx}): {len(test_viewpoint_stack)}")
    if len(viewpoint_stack) == 0:
        raise ValueError(f"No cameras found for frame {target_frame_idx}!")

    background = get_background_color(dataset, device="cuda")
    
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    
    first_iter = 1
    final_iter = iterations
    
    progress_bar = tqdm(range(first_iter, final_iter + 1), desc="Training progress")
    
    # ==========================
    # Visualization setup
    # ==========================
    
    # TensorBoard logging
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(model_path)
    else:
        print("Tensorboard not available: not logging progress")
    
    # ==========================
    # Training loop
    # ==========================
    
    for iteration in range(first_iter, final_iter + 1):
        iter_start.record()
        
        gaussians.update_learning_rate(iteration)
        
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        viewpoint_cams = []
        idx = 0
        while idx < opt.batch_size:
            if len(viewpoint_stack) == 0:
                viewpoint_stack = temp_list.copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            viewpoint_cams.append(viewpoint_cam)
            idx += 1
            
        if (iteration - 1) == 0: # debug_from
            pipe.debug = True
            
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage="coarse", cam_type=scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            
            if scene.dataset_type != "PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        
        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images, 0)
        gt_image_tensor = torch.cat(gt_images, 0)
        
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
        loss = Ll1
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor, gt_image_tensor)
            loss += opt.lambda_dssim * (1.0 - ssim_loss)
        
        # Spherical regularization loss: encourage Gaussians to be spherical (not elongated)
        # This penalizes the variance of scaling across the three dimensions
        if hasattr(opt, 'lambda_spherical') and opt.lambda_spherical > 0:
            # Get activated scaling (after activation function)
            scaling_activated = gaussians.scaling_activation(gaussians._scaling)  # [N, 3]
            # Compute mean scaling for each Gaussian (across 3 dimensions)
            scaling_mean = scaling_activated.mean(dim=1, keepdim=True)  # [N, 1]
            # Compute variance of scaling for each Gaussian
            scaling_variance = ((scaling_activated - scaling_mean) ** 2).mean(dim=1)  # [N]
            # Average variance across all Gaussians
            spherical_loss = scaling_variance.mean()
            loss += opt.lambda_spherical * spherical_loss
        
        loss.backward()
        
        if torch.isnan(loss).any():
            print("loss is nan, end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
            
        with torch.no_grad():
            viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
            for idx in range(0, len(viewspace_point_tensor_list)):
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad

        iter_end.record()
        
        with torch.no_grad():
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point": f"{total_point}"})
                progress_bar.update(10)
            
            if iteration == final_iter:
                progress_bar.close()
            
            # ==========================
            # Visualization
            # ==========================
            
            # Periodically render train and test camera previews
            if opt.visualize_interval > 0 and iteration % opt.visualize_interval == 0:
                # Training-camera previews
                print(f"\n[ITER {iteration}] Visualizing all training cameras...")
                try:
                    # Use the full camera list because viewpoint_stack is mutated during training.
                    cameras_for_vis = temp_list.copy()
                    render_training_image(scene, gaussians, cameras_for_vis, render, pipe, background,
                                        "coarse_all_train", iteration, timer.get_elapsed_time(), scene.dataset_type)
                    print(f"[ITER {iteration}] Visualization saved to {os.path.join(scene.model_path, 'coarse_all_train_render', 'images')}")
                    print(f"[ITER {iteration}] Visualized {len(cameras_for_vis)} training cameras")
                except Exception as e:
                    print(f"[ITER {iteration}] Warning: Failed to visualize training cameras: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Test-camera previews
                if len(test_temp_list) > 0:
                    print(f"\n[ITER {iteration}] Visualizing all test cameras...")
                    try:
                        # Use the full test-camera list.
                        test_cameras_for_vis = test_temp_list.copy()
                        render_training_image(scene, gaussians, test_cameras_for_vis, render, pipe, background,
                                            "coarse_all_test", iteration, timer.get_elapsed_time(), scene.dataset_type)
                        print(f"[ITER {iteration}] Visualization saved to {os.path.join(scene.model_path, 'coarse_all_test_render', 'images')}")
                        print(f"[ITER {iteration}] Visualized {len(test_cameras_for_vis)} test cameras")
                    except Exception as e:
                        print(f"[ITER {iteration}] Warning: Failed to visualize test cameras: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[ITER {iteration}] No test cameras found for frame {target_frame_idx}, skipping test visualization")
            
            # Periodically visualize the point distribution.
            # Visualize early iterations densely, then follow the configured interval.
            should_visualize = (opt.visualize_pointcloud_interval > 0 and iteration % opt.visualize_pointcloud_interval == 0)
            
            if should_visualize:
                try:
                    pointcloud_vis_dir = os.path.join(scene.model_path, "coarse_pointcloud_distribution")
                    os.makedirs(pointcloud_vis_dir, exist_ok=True)
                    pointcloud_vis_path = os.path.join(pointcloud_vis_dir, f"pointcloud_iter_{iteration}.png")
                    
                    # Include both train and test cameras for visualization.
                    cameras_for_vis = temp_list.copy()
                    if len(test_temp_list) > 0:
                        cameras_for_vis = cameras_for_vis + test_temp_list.copy()
                    
                    visualize_pointcloud_distribution(gaussians, pointcloud_vis_path, 
                                                      cameras=cameras_for_vis, iteration=iteration)
                except Exception as e:
                    print(f"[ITER {iteration}] Warning: Failed to visualize point cloud distribution: {e}")
                    import traceback
                    traceback.print_exc()
            
            # TensorBoard logging
            if tb_writer is not None:
                tb_writer.add_scalar('train/loss', loss.item(), iteration)
                tb_writer.add_scalar('train/psnr', psnr_.item(), iteration)
                tb_writer.add_scalar('train/point_count', total_point, iteration)
                tb_writer.add_scalar('train/l1_loss', Ll1.item(), iteration)
                if hasattr(opt, 'lambda_spherical') and opt.lambda_spherical > 0:
                    tb_writer.add_scalar('train/spherical_loss', spherical_loss.item(), iteration)
            
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
                
                opacity_threshold = opt.opacity_threshold_coarse
                densify_threshold = opt.densify_grad_threshold_coarse
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < 360000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, "coarse")
                
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0] > 200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < 360000 and opt.add_point:
                    gaussians.grow(5, 5, scene.model_path, iteration, "coarse")
                
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
            
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, "coarse")
                checkpoint_path = os.path.join(scene.model_path, f"chkpnt_coarse_{iteration}.pth")
                torch.save((gaussians.capture(), iteration), checkpoint_path)
    
    if final_iter not in saving_iterations:
        print(f"\n[Final Save] Saving final model at iteration {final_iter}")
        scene.save(final_iter, "coarse")
        checkpoint_path = os.path.join(scene.model_path, f"chkpnt_coarse_{final_iter}.pth")
        torch.save((gaussians.capture(), final_iter), checkpoint_path)
    
    timer.pause()
    print(f"\n[Training Complete] Total time: {timer.get_elapsed_time():.2f}s")
    
    # Close the TensorBoard writer.
    if tb_writer is not None:
        tb_writer.close()
    
    # Return the trained model.
    return gaussians, scene
