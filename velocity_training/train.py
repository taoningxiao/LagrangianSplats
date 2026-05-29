import os
import json
import gc
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
import imageio
from datetime import datetime
import shutil
import cv2 as cv
from scipy.interpolate import RegularGridInterpolator
from velocity_common.rbf import WendlandC4
import utils.grid_utils as utils_grid
import torch.nn as nn
from velocity_common.dfrbf import TiDFRBF
from velocity_training.semilag import semi_lagrangian_forward, semi_lagrangian_backward
from gaussian_renderer import render
from gaussian_renderer.training import train_specific_frame_gaussian, combine_train_test_datasets
from utils.loss_utils import l1_loss, ssim
from scene.gaussian_loader import load_gaussian_model
from utils.sh_utils import RGB2SH
from utils.general_utils import inverse_sigmoid
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from velocity_common.coordinate_transform import CoordinateTransform
from velocity_common.kernels import generate_kernels
from velocity_common.taichi_utils import scatter_grad_to_grid_taichi
from velocity_common.utils import set_device, get_background_color
from velocity_training.models import InflowGaussians
from velocity_training.wrappers import ExtendedGaussianWrapper, GaussianOverrideWrapper
from velocity_training.utils import precompute_camera_masks, pretrain_inflow_gaussians, visualize_inflow_region_all_cameras
from velocity_common.ray_utils import get_rays_from_camera, ray_inflow_region_intersection


def _release_unused_memory():
	"""Return freed tensors/arrays to the system between long sliding windows."""
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	try:
		import ctypes
		ctypes.CDLL("libc.so.6").malloc_trim(0)
	except Exception:
		pass


# Visualization helper functions
def velLegendHSV(hsvin, is3D, lw=-1, constV=255):
	# hsvin: (b), h, w, 3
	# always overwrite hsvin borders [lw], please pad hsvin before hand
	# or fill whole hsvin (lw < 0)
	ih, iw = hsvin.shape[-3:-1]
	if lw<=0: # fill whole
		a_list, b_list = [range(ih)], [range(iw)]
	else: # fill border
		a_list = [range(ih),  range(lw), range(ih), range(ih-lw, ih)]
		b_list = [range(lw),  range(iw), range(iw-lw, iw), range(iw)]
	for a,b in zip(a_list, b_list):
		for _fty in a:
			for _ftx in b:
				fty = _fty - ih//2
				ftx = _ftx - iw//2
				ftang = np.arctan2(fty, ftx) + np.pi
				ftang = ftang*(180/np.pi/2)
				hsvin[...,_fty,_ftx,0] = np.expand_dims(ftang, axis=-1) # 0-360 
				hsvin[...,_fty,_ftx,2] = constV
				if (not is3D) or (lw == 1):
					hsvin[...,_fty,_ftx,1] = 255
				else:
					thetaY1 = 1.0 - ((ih//2) - abs(fty)) / float( lw if (lw > 1) else (ih//2) )
					thetaY2 = 1.0 - ((iw//2) - abs(ftx)) / float( lw if (lw > 1) else (iw//2) )
					fthetaY = max(thetaY1, thetaY2) * (0.5*np.pi)
					ftxY, ftyY = np.cos(fthetaY), np.sin(fthetaY)
					fangY = np.arctan2(ftyY, ftxY)
					fangY = fangY*(240/np.pi*2) # 240 - 0
					hsvin[...,_fty,_ftx,1] = 255 - fangY


def cubecenter(cube, axis, half = 0):
	# cube: (b,)h,h,h,c
	# axis: 1 (z), 2 (y), 3 (x)
	reduce_axis = [a for a in [1,2,3] if a != axis]
	pack = np.mean(cube, axis=tuple(reduce_axis)) # (b,)h,c
	pack = np.sqrt(np.sum( np.square(pack), axis=-1 ) + 1e-6) # (b,)h

	length = cube.shape[axis-5] # h
	weights = np.arange(0.5/length,1.0,1.0/length)
	if half == 1: # first half
		weights = np.where( weights < 0.5, weights, np.zeros_like(weights))
		pack = np.where( weights < 0.5, pack, np.zeros_like(pack))
	elif half == 2: # second half
		weights = np.where( weights > 0.5, weights, np.zeros_like(weights))
		pack = np.where( weights > 0.5, pack, np.zeros_like(pack))

	weighted = pack * weights # (b,)h
	weiAxis = np.sum(weighted, axis=-1) / np.sum(pack, axis=-1) * length # (b,)
	
	return weiAxis.astype(np.int32) # a ceiling is included


def vel2hsv(velin, is3D, logv, scale=None): # 2D
	fx, fy = velin[...,0], velin[...,1]
	ori_shape = list(velin.shape[:-1]) + [3]
	if is3D: 
		fz = velin[...,2]
		ang = np.arctan2(fz, fx) + np.pi # angXZ
		zxlen2 = fx*fx+fz*fz
		angY = np.arctan2(np.abs(fy), np.sqrt(zxlen2))
		v = np.sqrt(zxlen2+fy*fy)
	else:
		v = np.sqrt(fx*fx+fy*fy)
		ang = np.arctan2(fy, fx) + np.pi
	
	if logv:
		v = np.log10(v+1)
	
	hsv = np.zeros(ori_shape, np.uint8)
	hsv[...,0] = ang *(180/np.pi/2)
	if is3D:
		hsv[...,1] = 255 - angY*(240/np.pi*2)  
	else:
		hsv[...,1] = 255
	if scale is not None:
		hsv[...,2] = np.minimum(v*scale, 255)
	else:
		hsv[...,2] = v/max(v.max(),1e-6) * 255.0
	return hsv


def vel_uv2hsv(vel, scale = 160, is3D=False, logv=False, mix=False):
	# vel: a np.float32 array, in shape of (?=b,) d,h,w,3 for 3D and (?=b,)h,w, 2 or 3 for 2D
	# scale: scale content to 0~255, something between 100-255 is usually good. 
	#        content will be normalized if scale is None
	# logv: visualize value with log
	# mix: use more slices to get a volumetric visualization if True, which is slow

	ori_shape = list(vel.shape[:-1]) + [3] # (?=b,) d,h,w,3
	if is3D: 
		new_range = list( range( len(ori_shape) ) )
		z_new_range = new_range[:]
		z_new_range[-4] = new_range[-3]
		z_new_range[-3] = new_range[-4]
		# print(z_new_range)
		YZXvel = np.transpose(vel, z_new_range)
		
		_xm,_ym,_zm = (ori_shape[-2]-1)//2, (ori_shape[-3]-1)//2, (ori_shape[-4]-1)//2
		
		if mix:
			_xlist = [cubecenter(vel, 3, 1),_xm,cubecenter(vel, 3, 2)]
			_ylist = [cubecenter(vel, 2, 1),_ym,cubecenter(vel, 2, 2)]
			_zlist = [cubecenter(vel, 1, 1),_zm,cubecenter(vel, 1, 2)]
		else:
			_xlist, _ylist, _zlist = [_xm], [_ym], [_zm]

		hsv = []
		for _x, _y, _z in zip (_xlist, _ylist, _zlist):
			# print(_x, _y, _z)
			_x, _y, _z = np.clip([_x, _y, _z], 0, ori_shape[-2:-5:-1])
			_yz = YZXvel[...,_x,:]
			_yz = np.stack( [_yz[...,2],_yz[...,0],_yz[...,1]], axis=-1)
			_yx = YZXvel[...,_z,:,:]
			_yx = np.stack( [_yx[...,0],_yx[...,2],_yx[...,1]], axis=-1)
			_zx = YZXvel[...,_y,:,:,:]
			_zx = np.stack( [_zx[...,0],_zx[...,1],_zx[...,2]], axis=-1)
			# print(_yx.shape, _yz.shape, _zx.shape)

			# in case resolution is not a cube, (res,res,res)
			_yxz = np.concatenate( [ #yz, yx, zx
				_yx, _yz ], axis = -2) # (?=b,),h,w+zdim,3
			
			if ori_shape[-3] < ori_shape[-4]:
				pad_shape = list(_yxz.shape) #(?=b,),h,w+zdim,3
				pad_shape[-3] = ori_shape[-4] - ori_shape[-3]
				_pad = np.zeros(pad_shape, dtype=np.float32)
				_yxz = np.concatenate( [_yxz,_pad], axis = -3)
			elif ori_shape[-3] > ori_shape[-4]:
				pad_shape = list(_zx.shape) #(?=b,),h,w+zdim,3
				pad_shape[-3] = ori_shape[-3] - ori_shape[-4]

				_zx = np.concatenate( 
					[_zx,np.zeros(pad_shape, dtype=np.float32)], axis = -3)
			
			midVel = np.concatenate( [ #yz, yx, zx
				_yxz, _zx
			], axis = -2) # (?=b,),h,w*3,3
			hsv += [vel2hsv(midVel, True, logv, scale)]
		# remove depth dim, increase with zyx slices
		ori_shape[-3] = 3 * ori_shape[-2]
		ori_shape[-2] = ori_shape[-1]
		ori_shape = ori_shape[:-1]
	else:
		hsv = [vel2hsv(vel, False, logv, scale)]

	bgr = []
	for _hsv in hsv:
		if len(ori_shape) > 3:
			_hsv = _hsv.reshape([-1]+ori_shape[-2:])
		if is3D:
			velLegendHSV(_hsv, is3D, lw=max(1,min(6,int(0.025*ori_shape[-2]))), constV=255)
		_hsv = cv.cvtColor(_hsv, cv.COLOR_HSV2BGR)
		if len(ori_shape) > 3:
			_hsv = _hsv.reshape(ori_shape)
		bgr += [_hsv]
	if len(bgr) == 1:
		bgr = bgr[0]
	else:
		bgr = bgr[0] * 0.2 + bgr[1] * 0.6 + bgr[2] * 0.2
	return bgr.astype(np.uint8)[::-1] # flip Y


def den_scalar2rgb(den, scale=160, is3D=False, logv=False, mix=True):
	# den: a np.float32 array, in shape of (?=b,) d,h,w,1 for 3D and (?=b,)h,w,1 for 2D
	# scale: scale content to 0~255, something between 100-255 is usually good. 
	#        content will be normalized if scale is None
	# logv: visualize value with log
	# mix: use averaged value as a volumetric visualization if True, else show middle slice

	ori_shape = list(den.shape)
	if ori_shape[-1] != 1:
		ori_shape.append(1)
		den = np.reshape(den, ori_shape)

	if is3D: 
		new_range = list( range( len(ori_shape) ) )
		z_new_range = new_range[:]
		z_new_range[-4] = new_range[-3]
		z_new_range[-3] = new_range[-4]
		# print(z_new_range)
		YZXden = np.transpose(den, z_new_range)
				
		if not mix:
			_yz = YZXden[...,(ori_shape[-2]-1)//2,:]
			_yx = YZXden[...,(ori_shape[-4]-1)//2,:,:]
			_zx = YZXden[...,(ori_shape[-3]-1)//2,:,:,:]
		else:
			_yz = np.average(YZXden, axis=-2)
			_yx = np.average(YZXden, axis=-3)
			_zx = np.average(YZXden, axis=-4)
			# print(_yx.shape, _yz.shape, _zx.shape)

		# in case resolution is not a cube, (res,res,res)
		_yxz = np.concatenate( [ #yz, yx, zx
			_yx, _yz ], axis = -2) # (?=b,),h,w+zdim,1
		
		if ori_shape[-3] < ori_shape[-4]:
			pad_shape = list(_yxz.shape) #(?=b,),h,w+zdim,1
			pad_shape[-3] = ori_shape[-4] - ori_shape[-3]
			_pad = np.zeros(pad_shape, dtype=np.float32)
			_yxz = np.concatenate( [_yxz,_pad], axis = -3)
		elif ori_shape[-3] > ori_shape[-4]:
			pad_shape = list(_zx.shape) #(?=b,),h,w+zdim,1
			pad_shape[-3] = ori_shape[-3] - ori_shape[-4]

			_zx = np.concatenate( 
				[_zx,np.zeros(pad_shape, dtype=np.float32)], axis = -3)
		
		midDen = np.concatenate( [ #yz, yx, zx
			_yxz, _zx
		], axis = -2) # (?=b,),h,w*3,1
	else:
		midDen = den

	if logv:
		midDen = np.log10(midDen+1)
	if scale is None:
		midDen = midDen / max(midDen.max(),1e-6) * 255.0
	else:
		midDen = midDen * scale
	grey = np.clip(midDen, 0, 255)

	grey = grey.astype(np.uint8)[::-1] # flip y

	if grey.shape[-1] == 1:
		grey = np.repeat(grey, 3, -1)

	return grey


def save_vtk_image_data(data, origin, spacing, filename, scalar_name="opacity"):
	"""
	Save 3D grid data as VTK Image Data (.vti) format for ParaView.
	
	Args:
		data: 3D numpy array of shape (nx, ny, nz) or (nz, ny, nx)
		origin: Origin point [x0, y0, z0]
		spacing: Spacing between grid points [dx, dy, dz]
		filename: Output filename (.vti)
		scalar_name: Name of the scalar field in VTK file (default: "opacity")
	"""
	# Ensure data is contiguous and float32
	data = np.ascontiguousarray(data, dtype=np.float32)
	
	# Get dimensions
	if len(data.shape) != 3:
		raise ValueError(f"Data must be 3D array, got shape {data.shape}")
	
	# VTK uses (nx, ny, nz) convention where nx is fastest changing
	# Our data is (nx, ny, nz) from grid_shape
	nx, ny, nz = data.shape
	
	# Write VTK XML Image Data format
	with open(filename, 'w') as f:
		f.write('<?xml version="1.0"?>\n')
		f.write('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
		f.write(f'  <ImageData WholeExtent="0 {nx-1} 0 {ny-1} 0 {nz-1}" ')
		f.write(f'Origin="{origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}" ')
		f.write(f'Spacing="{spacing[0]:.6f} {spacing[1]:.6f} {spacing[2]:.6f}">\n')
		f.write(f'    <Piece Extent="0 {nx-1} 0 {ny-1} 0 {nz-1}">\n')
		f.write(f'      <PointData Scalars="{scalar_name}">\n')
		f.write(f'        <DataArray type="Float32" Name="{scalar_name}" format="ascii" NumberOfComponents="1">\n')
		
		# Write data in VTK order: x varies fastest, then y, then z
		# VTK Image Data uses (x, y, z) indexing where x is fastest changing
		# Our data is (nx, ny, nz) = (x, y, z), so we iterate z, y, x (outer to inner)
		# and write data[i, j, k] where i=x, j=y, k=z
		for k in range(nz):  # z (outermost)
			for j in range(ny):  # y (middle)
				for i in range(nx):  # x (innermost, fastest changing)
					f.write(f'          {data[i, j, k]:.6e}\n')
		
		f.write('        </DataArray>\n')
		f.write('      </PointData>\n')
		f.write('    </Piece>\n')
		f.write('  </ImageData>\n')
		f.write('</VTKFile>\n')
	
	print(f"Saved VTK Image Data to {filename}")


def save_gaussians_to_vtk_polydata(xyz, opacity, scaling, rotation, features_dc=None, features_rest=None, filename=None, gaussian_type="original"):
	"""
	 Gaussian Splatting point cloudsave as VTK PolyData ，ParaView 
	
	Args:
		xyz: position，shape (n_gaussians, 3)， World Space  Sim Space
		opacity: opacity（），shape (n_gaussians, 1)  (n_gaussians,)
		scaling: scaling（），shape (n_gaussians, 3)
		rotation: rotation，shape (n_gaussians, 4)
		features_dc: optional DC ，shape (n_gaussians, 1, 3)
		features_rest: optional，shape (n_gaussians, sh_degree^2-1, 3)
		filename: output filename (.vtp)
		gaussian_type: type tag（"original"  "inflow"），used for ParaView in
	"""
	if isinstance(xyz, torch.Tensor):
		xyz = xyz.detach().cpu().numpy()
	if isinstance(opacity, torch.Tensor):
		opacity = opacity.detach().cpu().numpy()
	if isinstance(scaling, torch.Tensor):
		scaling = scaling.detach().cpu().numpy()
	if isinstance(rotation, torch.Tensor):
		rotation = rotation.detach().cpu().numpy()
	if features_dc is not None and isinstance(features_dc, torch.Tensor):
		features_dc = features_dc.detach().cpu().numpy()
	if features_rest is not None and isinstance(features_rest, torch.Tensor):
		features_rest = features_rest.detach().cpu().numpy()
	
	n_gaussians = len(xyz)
	
	if opacity.ndim > 1:
		opacity = opacity.squeeze()
	
	scaling_mean = np.mean(scaling, axis=1) if scaling.ndim > 1 else scaling
	scaling_max = np.max(scaling, axis=1) if scaling.ndim > 1 else scaling
	scaling_min = np.min(scaling, axis=1) if scaling.ndim > 1 else scaling
	
	rotation_norm = np.linalg.norm(rotation, axis=1) if rotation.ndim > 1 else np.abs(rotation)
	
	rgb_color = None
	if features_dc is not None:
		if features_dc.ndim == 3:
			features_dc = features_dc.squeeze(1)  # (n_gaussians, 3)
		rgb_color = features_dc
	
	with open(filename, 'w') as f:
		f.write('<?xml version="1.0"?>\n')
		f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
		f.write('  <PolyData>\n')
		f.write(f'    <Piece NumberOfPoints="{n_gaussians}" NumberOfVerts="{n_gaussians}">\n')
		
		# Points
		f.write('      <Points>\n')
		f.write('        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii">\n')
		for i in range(n_gaussians):
			f.write(f'          {xyz[i, 0]:.6e} {xyz[i, 1]:.6e} {xyz[i, 2]:.6e}\n')
		f.write('        </DataArray>\n')
		f.write('      </Points>\n')
		
		f.write('      <Verts>\n')
		f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
		for i in range(n_gaussians):
			f.write(f'          {i}\n')
		f.write('        </DataArray>\n')
		f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
		for i in range(n_gaussians):
			f.write(f'          {i + 1}\n')
		f.write('        </DataArray>\n')
		f.write('      </Verts>\n')
		
		f.write('      <PointData>\n')
		
		f.write(f'        <DataArray type="Int32" Name="gaussian_type" NumberOfComponents="1" format="ascii">\n')
		type_value = 0 if gaussian_type == "original" else 1
		for i in range(n_gaussians):
			f.write(f'          {type_value}\n')
		f.write('        </DataArray>\n')
		
		f.write('        <DataArray type="Float32" Name="opacity" NumberOfComponents="1" format="ascii">\n')
		for i in range(n_gaussians):
			f.write(f'          {opacity[i]:.6e}\n')
		f.write('        </DataArray>\n')
		
		f.write('        <DataArray type="Float32" Name="scaling_mean" NumberOfComponents="1" format="ascii">\n')
		for i in range(n_gaussians):
			f.write(f'          {scaling_mean[i]:.6e}\n')
		f.write('        </DataArray>\n')
		
		f.write('        <DataArray type="Float32" Name="scaling_max" NumberOfComponents="1" format="ascii">\n')
		for i in range(n_gaussians):
			f.write(f'          {scaling_max[i]:.6e}\n')
		f.write('        </DataArray>\n')
		
		f.write('        <DataArray type="Float32" Name="scaling_min" NumberOfComponents="1" format="ascii">\n')
		for i in range(n_gaussians):
			f.write(f'          {scaling_min[i]:.6e}\n')
		f.write('        </DataArray>\n')
		
		if scaling.ndim > 1 and scaling.shape[1] == 3:
			f.write('        <DataArray type="Float32" Name="scaling" NumberOfComponents="3" format="ascii">\n')
			for i in range(n_gaussians):
				f.write(f'          {scaling[i, 0]:.6e} {scaling[i, 1]:.6e} {scaling[i, 2]:.6e}\n')
			f.write('        </DataArray>\n')
		
		if rotation.ndim > 1 and rotation.shape[1] == 4:
			f.write('        <DataArray type="Float32" Name="rotation" NumberOfComponents="4" format="ascii">\n')
			for i in range(n_gaussians):
				f.write(f'          {rotation[i, 0]:.6e} {rotation[i, 1]:.6e} {rotation[i, 2]:.6e} {rotation[i, 3]:.6e}\n')
			f.write('        </DataArray>\n')
		
		f.write('        <DataArray type="Float32" Name="rotation_norm" NumberOfComponents="1" format="ascii">\n')
		for i in range(n_gaussians):
			f.write(f'          {rotation_norm[i]:.6e}\n')
		f.write('        </DataArray>\n')
		
		if rgb_color is not None and rgb_color.shape[1] == 3:
			f.write('        <DataArray type="Float32" Name="rgb_color" NumberOfComponents="3" format="ascii">\n')
			for i in range(n_gaussians):
				r = max(0, min(1, rgb_color[i, 0]))
				g = max(0, min(1, rgb_color[i, 1]))
				b = max(0, min(1, rgb_color[i, 2]))
				f.write(f'          {r:.6e} {g:.6e} {b:.6e}\n')
			f.write('        </DataArray>\n')
		
		f.write('      </PointData>\n')
		f.write('    </Piece>\n')
		f.write('  </PolyData>\n')
		f.write('</VTKFile>\n')
	
	print(f"Saved {gaussian_type} Gaussian data to {filename} ({n_gaussians} Gaussians)")


def visualize_pointcloud_with_opacity(positions, opacity_values, save_path, title="Gaussian Point Cloud"):
	"""
	visualizationpoint cloudthree-view plot，based on opacity coloring
	
	Args:
		positions: point cloudposition，shape (N, 3)，sim space
		opacity_values: opacity ，shape (N,)
		save_path: save path
		title: 
	"""
	if isinstance(positions, torch.Tensor):
		positions = positions.detach().cpu().numpy()
	if isinstance(opacity_values, torch.Tensor):
		opacity_values = opacity_values.detach().cpu().numpy()
	
	if positions.ndim == 1:
		positions = positions.reshape(1, -1)
	if opacity_values.ndim > 1:
		opacity_values = opacity_values.squeeze()
	
	fig = plt.figure(figsize=(18, 6))
	
	views = [
		{'title': 'Front View (X-Y plane)', 'x': 0, 'y': 1, 'xlabel': 'X', 'ylabel': 'Y'},
		{'title': 'Side View (Y-Z plane)', 'x': 1, 'y': 2, 'xlabel': 'Y', 'ylabel': 'Z'},
		{'title': 'Top View (X-Z plane)', 'x': 0, 'y': 2, 'xlabel': 'X', 'ylabel': 'Z'}
	]
	
	min_coords = positions.min(axis=0)
	max_coords = positions.max(axis=0)
	
	margin = (max_coords - min_coords).max() * 0.1
	min_coords -= margin
	max_coords += margin
	
	opacity_min = opacity_values.min()
	opacity_max = opacity_values.max()
	norm = Normalize(vmin=opacity_min, vmax=opacity_max)
	
	cmap = plt.cm.viridis
	
	for view_idx, view in enumerate(views):
		ax = fig.add_subplot(1, 3, view_idx + 1)
		
		scatter = ax.scatter(
			positions[:, view['x']], 
			positions[:, view['y']], 
			c=opacity_values,
			cmap=cmap,
			norm=norm,
			s=0.5,
			alpha=0.8,
			edgecolors='none'
		)
		
		ax.set_xlabel(view['xlabel'], fontsize=12)
		ax.set_ylabel(view['ylabel'], fontsize=12)
		ax.set_title(view['title'], fontsize=12, fontweight='bold')
		ax.grid(True, alpha=0.3)
		ax.set_aspect('equal', adjustable='box')
		
		ax.set_xlim(min_coords[view['x']], max_coords[view['x']])
		ax.set_ylim(min_coords[view['y']], max_coords[view['y']])
		
		if view_idx == 0:
			cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02)
			cbar.set_label('Opacity', rotation=270, labelpad=15, fontsize=10)
	
	plt.suptitle(title, fontsize=14, fontweight='bold')
	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f"Saved point cloud visualization with opacity coloring to {save_path}")


def trilinear_interpolation_torch(field, x, y, z, Nx, Ny, Nz):
	"""
	PyTorchinterpolation
	
	Args:
		field: 3D [Nx, Ny, Nz] (torch.Tensor)
		x, y, z: interpolationcoordinate (torch.Tensor)
		Nx, Ny, Nz: field size
	
	Returns:
		interpolation (torch.Tensor)
	"""
	x = torch.clamp(x, 0, Nx - 1)
	y = torch.clamp(y, 0, Ny - 1)
	z = torch.clamp(z, 0, Nz - 1)
	
	x0 = torch.floor(x).long()
	y0 = torch.floor(y).long()
	z0 = torch.floor(z).long()
	x1 = torch.clamp(x0 + 1, 0, Nx - 1)
	y1 = torch.clamp(y0 + 1, 0, Ny - 1)
	z1 = torch.clamp(z0 + 1, 0, Nz - 1)
	
	dx = x - x0.float()
	dy = y - y0.float()
	dz = z - z0.float()
	
	c000 = field[x0, y0, z0]
	c001 = field[x0, y0, z1]
	c010 = field[x0, y1, z0]
	c011 = field[x0, y1, z1]
	c100 = field[x1, y0, z0]
	c101 = field[x1, y0, z1]
	c110 = field[x1, y1, z0]
	c111 = field[x1, y1, z1]
	
	c00 = c000 * (1 - dx) + c100 * dx
	c01 = c001 * (1 - dx) + c101 * dx
	c10 = c010 * (1 - dx) + c110 * dx
	c11 = c011 * (1 - dx) + c111 * dx
	
	c0 = c00 * (1 - dy) + c10 * dy
	c1 = c01 * (1 - dy) + c11 * dy
	
	c = c0 * (1 - dz) + c1 * dz
	
	return c


def advect_vorticity_field_torch(initial_vorticity, velocity_field, dt, inflow_height_ratio=0.25):
	"""
	usevelocity fieldvorticity（PyTorch，）

	Args:
		initial_vorticity: vorticity [Nx, Ny, Nz, 3] (torch.Tensor)
		velocity_field: velocity field [Nx, Ny, Nz, 3] (torch.Tensor)
		dt: time steps (float)
		inflow_height_ratio: inflow-height ratio（y < inflow_height_ratio * Ny）

	Returns:
		advected_vorticity: vorticity [Nx, Ny, Nz, 3] (torch.Tensor)
	"""
	Nx, Ny, Nz, _ = initial_vorticity.shape
	device = initial_vorticity.device
	dtype = initial_vorticity.dtype

	x = torch.linspace(0, Nx-1, Nx, device=device, dtype=dtype)
	y = torch.linspace(0, Ny-1, Ny, device=device, dtype=dtype)
	z = torch.linspace(0, Nz-1, Nz, device=device, dtype=dtype)
	X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

	inflow_mask = Y < (inflow_height_ratio * Ny)

	advected_vorticity = initial_vorticity.clone()

	non_inflow_mask = ~inflow_mask

	if torch.any(non_inflow_mask):
		vx = velocity_field[:, :, :, 0]
		vy = velocity_field[:, :, :, 1]
		vz = velocity_field[:, :, :, 2]

		x_back = X - vx * dt
		y_back = Y - vy * dt
		z_back = Z - vz * dt

		x_norm = x_back / (Nx - 1)
		y_norm = y_back / (Ny - 1)
		z_norm = z_back / (Nz - 1)

		valid_mask = (x_norm >= 0) & (x_norm <= 1) & (y_norm >= 0) & (y_norm <= 1) & (z_norm >= 0) & (z_norm <= 1)

		advection_mask = non_inflow_mask & valid_mask

		if torch.any(advection_mask):
			i_back = x_norm * (Nx - 1)
			j_back = y_norm * (Ny - 1)
			k_back = z_norm * (Nz - 1)

			for c in range(3):
				advected_comp = trilinear_interpolation_torch(
					initial_vorticity[:, :, :, c],
					i_back[advection_mask],
					j_back[advection_mask],
					k_back[advection_mask],
					Nx, Ny, Nz
				)
				advected_vorticity[:, :, :, c][advection_mask] = advected_comp

	return advected_vorticity


def compute_velocity_gradient_torch(velocity_field: torch.Tensor) -> torch.Tensor:
	"""
	 ∇u（unit grid spacing），outputshape [Nx, Ny, Nz, 3, 3]
	Args:
		velocity_field: [Nx, Ny, Nz, 3]
	Returns:
		grad_v: [Nx, Ny, Nz, 3, 3]，framevelocity component(u,v,w)，framecoordinate(x,y,z)
	"""
	assert velocity_field.dim() == 4 and velocity_field.shape[-1] == 3, "velocity_field must be [Nx,Ny,Nz,3]"
	vx = velocity_field[:, :, :, 0]
	vy = velocity_field[:, :, :, 1]
	vz = velocity_field[:, :, :, 2]
	dx_vx = torch.gradient(vx, dim=0)[0]
	dx_vy = torch.gradient(vy, dim=0)[0]
	dx_vz = torch.gradient(vz, dim=0)[0]
	dy_vx = torch.gradient(vx, dim=1)[0]
	dy_vy = torch.gradient(vy, dim=1)[0]
	dy_vz = torch.gradient(vz, dim=1)[0]
	dz_vx = torch.gradient(vx, dim=2)[0]
	dz_vy = torch.gradient(vy, dim=2)[0]
	dz_vz = torch.gradient(vz, dim=2)[0]
	grad_v = torch.stack([
		torch.stack([dx_vx, dy_vx, dz_vx], dim=-1),
		torch.stack([dx_vy, dy_vy, dz_vy], dim=-1),
		torch.stack([dx_vz, dy_vz, dz_vz], dim=-1)
	], dim=-2)
	return grad_v


# ============================================================================
# ============================================================================

def _load_or_train_frame_gaussian(args, savedir, gaussian_ckpt_path, start_frame=0):
	"""
	Load or train the Gaussian model for a specific frame.
	
	Args:
		args: training arguments
		savedir: output directory
		gaussian_ckpt_path: checkpointpath（optional）
		start_frame: start frame（0）
	
	Returns:
		gaussians: Gaussian model
		scene: Scene object
	"""
	total_iter = getattr(args, 'coarse_iterations', 3000)
	current_densify_until = getattr(args, 'densify_until_iter', 15000)
	
	if current_densify_until >= total_iter:
		new_densify_stop = total_iter - 100
		print(f"\n[Config Warning] densify_until_iter ({current_densify_until}) >= total_iter ({total_iter}).")
		print(f"[Config Fix] Clamping densify_until_iter to {new_densify_stop} to prevent opacity reset at final step.")
		args.densify_until_iter = new_densify_stop
		
	if gaussian_ckpt_path is not None and os.path.exists(gaussian_ckpt_path):
		print(f"\n[Step 1] Loading pre-trained Frame {start_frame} Gaussian from checkpoint...")
		print(f"Loading from: {gaussian_ckpt_path}")
		try:
			gaussians, scene = load_gaussian_model(
				model_path=gaussian_ckpt_path,
				data_dir=args.datadir,
				load_iteration=-1,
				load_only_xyz=False,
				base_args=args
			)
			if gaussians is None:
				raise ValueError("gaussians is None after loading")
			if scene is None:
				raise ValueError("scene is None after loading")
			print(f"Successfully loaded Gaussian model from {gaussian_ckpt_path}")
			print(f"Gaussians type: {type(gaussians)}, Scene type: {type(scene)}")
			
			from argparse import ArgumentParser
			from arguments import OptimizationParams
			op = OptimizationParams(ArgumentParser())
			opt = op.extract(args)
			print("Setting up optimizer for loaded Gaussians...")
			gaussians.training_setup(opt)
			print("Optimizer setup completed")
		except Exception as e:
			print(f"Error loading Gaussian model: {e}")
			import traceback
			traceback.print_exc()
			raise
	else:
		print(f"\n[Step 1] Training Frame {start_frame} Gaussian...")
		gaussian_path = os.path.join(savedir, f"gaussian_frame{start_frame}")
		gaussians, scene = train_specific_frame_gaussian(args, target_frame_idx=start_frame, model_path=gaussian_path)
	
	return gaussians, scene


def calculate_velocity_optimization_windows(w, total_frames):
	"""
	eachvelocity field
	
	Args:
		w: window size（frame）
		total_frames: total frame count
	
	Returns:
		optimization_counts: ，{frame_idx: num_windows} eachvelocity fieldin
	"""
	optimization_counts = {}
	
	for i in range(total_frames - 1):
		window_start = max(0, i - w + 1)
		window_end = min(i + 1, total_frames - w + 1)
		num_windows = max(0, window_end - window_start)
		
		if i < w - 1:
			num_windows = i + 1
		elif w - 1 <= i < total_frames - w + 1:
			num_windows = w
		else:
			num_windows = total_frames - i
		
		optimization_counts[i] = num_windows
	
	return optimization_counts


def _setup_velocity_models_with_individual_optimizers(args, device, scale, start_frame, end_frame, dt,
													   velocity_optimization_counts=None,
													   existing_velocity_models=None,
													   existing_optimizers=None,
													   existing_schedulers=None):
	"""
	velocity field，eachvelocity fieldoptimizer
	
	Args:
		args: training arguments
		device: device
		scale: scaling
		start_frame: start frame
		end_frame: end frame（contains）
		dt: time steps
		velocity_optimization_counts: ，{frame_idx: num_windows} eachvelocity fieldin
		existing_velocity_models: optional velocity  {global_frame_idx: model}
		existing_optimizers: optionaloptimizer {global_frame_idx: optimizer}
		existing_schedulers: optional {global_frame_idx: scheduler}
	
	Returns:
		vel_models: velocity fieldlist（ModuleList）
		vel_optimizers: eachvelocity fieldoptimizer {global_frame_idx: optimizer}
		vel_schedulers: eachvelocity field {global_frame_idx: scheduler}
		grid_points: grid points
		grid_shape: shape
		lengths_tensor: length tensor
		coord_trans: coordinate transform
		background: background color
		pipe: rendering pipeline parameters
		dataset: dataset parameters
	"""
	# Load Meta Info
	with open(os.path.join(args.datadir, 'info.json'), 'r') as fp:
		meta = json.load(fp)
		voxel_tran = np.float32(meta['voxel_matrix'])
		voxel_tran = np.stack([voxel_tran[:,2],voxel_tran[:,1],voxel_tran[:,0],voxel_tran[:,3]],axis=1)
		voxel_scale = np.broadcast_to(meta['voxel_scale'],[3])
		voxel_scale = voxel_scale * args.scene_scale
		voxel_tran[:3,3] *= args.scene_scale
		
	coord_trans = CoordinateTransform(voxel_tran, voxel_scale, device)
	
	# Grid Setup
	s = float(scale)
	nx, ny, nz = int(args.Nx/s), int(args.Ny/s), int(args.Nz/s)
	grid_shape = (nx, ny, nz)
 
	if dt is None:
		dt = (args.sim_steps / s)
	
	# Create RBF Models
	num_vel_models = end_frame - start_frame - 1
	if num_vel_models <= 0:
		raise ValueError(f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}. Need at least 2 frames.")
	
	lengths = np.array([args.Nx, args.Ny, args.Nz])
	lengths_tensor = torch.from_numpy(lengths).float().to(device)
	n_kernels_full = int(args.kernel_num)
	n_kernels = max(1, int(round(n_kernels_full / (s ** 3))))
	init_centers, h = generate_kernels(lengths, n_kernels)
	
	vel_models = nn.ModuleList()
	vel_optimizers = {}
	vel_schedulers = {}
	
	for i in range(num_vel_models):
		global_frame_idx = start_frame + i
		
		if existing_velocity_models and global_frame_idx in existing_velocity_models:
			vel_models.append(existing_velocity_models[global_frame_idx])
		else:
			vel_models.append(TiDFRBF(WendlandC4(), init_centers, h, device=device))
		
		if existing_optimizers and global_frame_idx in existing_optimizers:
			vel_optimizers[global_frame_idx] = existing_optimizers[global_frame_idx]
		else:
			optimizer = torch.optim.Adam(vel_models[i].parameters(), lr=args.lrate_vel)
			vel_optimizers[global_frame_idx] = optimizer
		
		if velocity_optimization_counts is not None:
			num_windows = velocity_optimization_counts.get(global_frame_idx, 1)
		else:
			num_windows = 1
		
		total_epochs_for_this_velocity = num_windows * args.num_epochs
		gamma = 0.01 ** (1.0 / total_epochs_for_this_velocity)
		
		if existing_schedulers and global_frame_idx in existing_schedulers:
			vel_schedulers[global_frame_idx] = existing_schedulers[global_frame_idx]
		else:
			scheduler = torch.optim.lr_scheduler.ExponentialLR(
				vel_optimizers[global_frame_idx], gamma, verbose=True
			)
			vel_schedulers[global_frame_idx] = scheduler
	
	vel_models = vel_models.to(device)
	
	# Pre-generate grid points
	min_corner = np.zeros(3)
	max_corner = lengths
	grid_points_np = utils_grid.generate_gridpoints(grid_shape, min_corner, max_corner)
	grid_points = torch.from_numpy(grid_points_np).float().to(device)
	
	# Render Setup
	from argparse import ArgumentParser
	from arguments import PipelineParams, ModelParams
	lp = ModelParams(ArgumentParser(), sentinel=True)
	dataset = lp.extract(args)
	pp = PipelineParams(ArgumentParser())
	pipe = pp.extract(args)
	
	for target, source in [(dataset, lp), (pipe, pp)]:
		for key, value in vars(source).items():
			if isinstance(value, ArgumentParser): continue
			attr_name = key[1:] if key.startswith("_") else key
			if not hasattr(target, attr_name):
				setattr(target, attr_name, value)
	
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
	background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
	
	return vel_models, vel_optimizers, vel_schedulers, grid_points, grid_shape, lengths_tensor, coord_trans, background, pipe, dataset, dt


def _setup_velocity_models(args, device, scale, start_frame, end_frame, dt):
	"""
	velocity field
	
	Args:
		args: training arguments
		device: device
		scale: scaling
		start_frame: start frame
		end_frame: end frame（contains）
		dt: time steps
	
	Returns:
		vel_models: velocity fieldlist
		vel_optimizer: optimizer
		vel_scheduler: learning-rate scheduler
		grid_points: grid points
		grid_shape: shape
		lengths_tensor: length tensor
		coord_trans: coordinate transform
		background: background color
		pipe: rendering pipeline parameters
		dataset: dataset parameters
	"""
	# Load Meta Info
	with open(os.path.join(args.datadir, 'info.json'), 'r') as fp:
		meta = json.load(fp)
		voxel_tran = np.float32(meta['voxel_matrix'])
		voxel_tran = np.stack([voxel_tran[:,2],voxel_tran[:,1],voxel_tran[:,0],voxel_tran[:,3]],axis=1)
		voxel_scale = np.broadcast_to(meta['voxel_scale'],[3])
		voxel_scale = voxel_scale * args.scene_scale
		voxel_tran[:3,3] *= args.scene_scale
		
	coord_trans = CoordinateTransform(voxel_tran, voxel_scale, device)
	
	# Grid Setup
	s = float(scale)
	nx, ny, nz = int(args.Nx/s), int(args.Ny/s), int(args.Nz/s)
	grid_shape = (nx, ny, nz)
 
	if dt is None:
		dt = (args.sim_steps / s)
	
	# Create RBF Models
	num_vel_models = end_frame - start_frame - 1
	if num_vel_models <= 0:
		raise ValueError(f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}. Need at least 2 frames.")
	
	lengths = np.array([args.Nx, args.Ny, args.Nz])
	lengths_tensor = torch.from_numpy(lengths).float().to(device)
	n_kernels_full = int(args.kernel_num)
	n_kernels = max(1, int(round(n_kernels_full / (s ** 3))))
	init_centers, h = generate_kernels(lengths, n_kernels)
	
	vel_models = nn.ModuleList([
		TiDFRBF(WendlandC4(), init_centers, h, device=device) 
		for _ in range(num_vel_models)
	]).to(device)
	
	vel_optimizer = torch.optim.Adam(vel_models.parameters(), lr=args.lrate_vel)
	gamma = 1e-2 ** (1.0 / args.num_epochs)
	vel_scheduler = torch.optim.lr_scheduler.ExponentialLR(vel_optimizer, gamma, verbose=True)
	
	# Pre-generate grid points
	min_corner = np.zeros(3)
	max_corner = lengths
	grid_points_np = utils_grid.generate_gridpoints(grid_shape, min_corner, max_corner)
	grid_points = torch.from_numpy(grid_points_np).float().to(device)

	# Render Setup
	from argparse import ArgumentParser
	from arguments import PipelineParams, ModelParams
	lp = ModelParams(ArgumentParser(), sentinel=True)
	dataset = lp.extract(args)
	pp = PipelineParams(ArgumentParser())
	pipe = pp.extract(args)
	
	for target, source in [(dataset, lp), (pipe, pp)]:
		for key, value in vars(source).items():
			if isinstance(value, ArgumentParser): continue
			attr_name = key[1:] if key.startswith("_") else key
			if not hasattr(target, attr_name):
				setattr(target, attr_name, value)
	
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
	background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

	return vel_models, vel_optimizer, vel_scheduler, grid_points, grid_shape, lengths_tensor, coord_trans, background, pipe, dataset, dt


def _precompute_camera_mapping(scene, frame_num, args, dataset):
	"""
	Precompute the camera mapping for each frame.
	
	Returns:
		frame_to_cameras: frame indexcameralist
		train_cameras: training cameraslist
	"""
	if scene is None:
		print("Warning: scene is None, creating a new Scene object...")
		from scene import Scene
		from scene.gaussian_model import GaussianModel
		from arguments import ModelHiddenParams
		from argparse import ArgumentParser
		temp_gaussians = GaussianModel(dataset.sh_degree, ModelHiddenParams(ArgumentParser()).extract(args))
		scene = Scene(dataset, temp_gaussians, load_iteration=None, load_coarse=None, load_only_xyz=False)
	
	train_cameras = scene.getTrainCameras()
	test_cameras = scene.getTestCameras()
	
	use_test_in_training = getattr(args, 'use_test_in_training', False)
	if use_test_in_training:
		print(f"[Velocity Training] Using test dataset in training (train: {len(train_cameras)}, test: {len(test_cameras)})")
		train_cameras = combine_train_test_datasets(train_cameras, test_cameras)
		print(f"[Velocity Training] Combined dataset size: {len(train_cameras)}")
	
	frame_to_cameras = {}
	for t in range(frame_num):
		frame_to_cameras[t] = []
	
	for cam_idx, cam in enumerate(train_cameras):
		if hasattr(cam, 'time') and cam.time is not None:
			if frame_num > 1:
				frame_idx = round(cam.time * (frame_num - 1))
			else:
				frame_idx = 0
			frame_idx = max(0, min(frame_idx, frame_num - 1))
			frame_to_cameras[frame_idx].append((cam_idx, cam))
		else:
			print(f"  Warning: Camera {cam_idx} has no time attribute, assigning to frame 0")
			frame_to_cameras[0].append((cam_idx, cam))
	
	for t in range(frame_num):
		num_cams = len(frame_to_cameras[t])
		if num_cams > 1:
			print(f"  Frame {t}: Found {num_cams} matching cameras")
		elif num_cams == 0:
			print(f"  Warning: Frame {t}: No cameras found")
	
	print(f"[Pre-computing] Camera mapping complete for {frame_num} frames")
	return frame_to_cameras, train_cameras


def _merge_inflow_to_gaussians(gaussians, inflow_gaussians, group_indices, coord_trans, lengths_tensor, device):
	"""
	inflow groupsgaussiansin
	
	Args:
		gaussians: GaussianModelobject
		inflow_gaussians: InflowGaussiansobject
		group_indices: inflow grouplist
		coord_trans: CoordinateTransformobject
		lengths_tensor: Sim Spacetensor
		device: device
		
	Returns:
		gaussians: mergedGaussianModelobject（）
		inflow_point_indices: markinflowlist，Trueinflow，Falseoriginal
	"""
	if len(group_indices) == 0:
		inflow_point_indices = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device=device)
		return gaussians, inflow_point_indices
	
	inflow_xyz_list = []
	inflow_opacity_list = []
	inflow_scaling_list = []
	inflow_rotation_list = []
	inflow_features_dc_list = []
	inflow_features_rest_list = []
	inflow_deformation_table_list = []
	
	for group_idx in group_indices:
		inflow_xyz = inflow_gaussians.get_group_xyz(group_idx)  # [M, 3]
		inflow_xyz_list.append(inflow_xyz)
		
		inflow_opacity = inflow_gaussians._opacity_groups[group_idx]  # [M, 1]
		inflow_scaling = inflow_gaussians._scaling_groups[group_idx]  # [M, 3]
		inflow_rotation = inflow_gaussians._rotation_groups[group_idx]  # [M, 4]
		inflow_features_dc = inflow_gaussians._features_dc_groups[group_idx]  # [M, 1, C]
		inflow_features_rest = inflow_gaussians._features_rest_groups[group_idx]  # [M, K, C]
		
		inflow_opacity_list.append(inflow_opacity)
		inflow_scaling_list.append(inflow_scaling)
		inflow_rotation_list.append(inflow_rotation)
		inflow_features_dc_list.append(inflow_features_dc)
		inflow_features_rest_list.append(inflow_features_rest)
		
		num_points = inflow_xyz.shape[0]
		inflow_deformation_table = torch.zeros(num_points, dtype=torch.bool, device=device)
		inflow_deformation_table_list.append(inflow_deformation_table)
	
	merged_inflow_xyz = torch.cat(inflow_xyz_list, dim=0)  # [N_inflow, 3]
	merged_inflow_opacity = torch.cat(inflow_opacity_list, dim=0)  # [N_inflow, 1]
	merged_inflow_scaling = torch.cat(inflow_scaling_list, dim=0)  # [N_inflow, 3]
	merged_inflow_rotation = torch.cat(inflow_rotation_list, dim=0)  # [N_inflow, 4]
	merged_inflow_features_dc = torch.cat(inflow_features_dc_list, dim=0)  # [N_inflow, 1, C]
	merged_inflow_features_rest = torch.cat(inflow_features_rest_list, dim=0)  # [N_inflow, K, C]
	merged_inflow_deformation_table = torch.cat(inflow_deformation_table_list, dim=0)  # [N_inflow]
	
	orig_xyz = gaussians._xyz  # [N_orig, 3]
	orig_opacity = gaussians._opacity  # [N_orig, 1]
	orig_scaling = gaussians._scaling  # [N_orig, 3]
	orig_rotation = gaussians._rotation  # [N_orig, 4]
	orig_features_dc = gaussians._features_dc  # [N_orig, 1, C]
	orig_features_rest = gaussians._features_rest  # [N_orig, K, C]
	orig_deformation_table = gaussians._deformation_table  # [N_orig]
	
	merged_xyz = torch.cat([orig_xyz, merged_inflow_xyz], dim=0)  # [N_orig + N_inflow, 3]
	merged_opacity = torch.cat([orig_opacity, merged_inflow_opacity], dim=0)  # [N_orig + N_inflow, 1]
	merged_scaling = torch.cat([orig_scaling, merged_inflow_scaling], dim=0)  # [N_orig + N_inflow, 3]
	merged_rotation = torch.cat([orig_rotation, merged_inflow_rotation], dim=0)  # [N_orig + N_inflow, 4]
	merged_features_dc = torch.cat([orig_features_dc, merged_inflow_features_dc], dim=0)  # [N_orig + N_inflow, 1, C]
	merged_features_rest = torch.cat([orig_features_rest, merged_inflow_features_rest], dim=0)  # [N_orig + N_inflow, K, C]
	merged_deformation_table = torch.cat([orig_deformation_table, merged_inflow_deformation_table], dim=0)  # [N_orig + N_inflow]
	
	gaussians._xyz = nn.Parameter(merged_xyz.requires_grad_(True))
	gaussians._opacity = nn.Parameter(merged_opacity.requires_grad_(True))
	gaussians._scaling = nn.Parameter(merged_scaling.requires_grad_(True))
	gaussians._rotation = nn.Parameter(merged_rotation.requires_grad_(True))
	gaussians._features_dc = nn.Parameter(merged_features_dc.requires_grad_(True))
	gaussians._features_rest = nn.Parameter(merged_features_rest.requires_grad_(True))
	gaussians._deformation_table = merged_deformation_table
	
	orig_max_radii2D = gaussians.max_radii2D  # [N_orig]
	new_max_radii2D = torch.zeros(merged_xyz.shape[0], device=device)
	new_max_radii2D[:orig_max_radii2D.shape[0]] = orig_max_radii2D
	gaussians.max_radii2D = new_max_radii2D
	
	orig_num_points = orig_xyz.shape[0]
	inflow_num_points = merged_inflow_xyz.shape[0]
	inflow_point_indices = torch.zeros(orig_num_points + inflow_num_points, dtype=torch.bool, device=device)
	inflow_point_indices[orig_num_points:] = True
	
	return gaussians, inflow_point_indices


def _setup_inflow_gaussians(args, gaussians, coord_trans, lengths_tensor, start_frame, end_frame,
							inflow_ratio, insert_ratio, device, savedir, scene, frame_to_cameras,
							pipe, background, visualize_inflow_region, existing_inflow_gaussians=None):
	"""
	Inflow Gaussians
	
	Args:
		start_frame: start frame
		end_frame: end frame（contains）
		existing_inflow_gaussians: InflowGaussiansobject（inflow groups）
	
	Returns:
		inflow_gaussians: InflowGaussiansobjectNone
	"""
	inflow_region_min = getattr(args, 'inflow_region_min', [0.0, 0.1, 0.0])
	inflow_region_max = getattr(args, 'inflow_region_max', [1.0, 0.3, 1.0])
	
	inflow_gaussians = None
	if inflow_ratio > 0:
		print(f"\n[Step 2.5] Initializing Inflow Gaussians...")
		print(f"  Frame range: start_frame={start_frame}, end_frame={end_frame}")
		print(f"  Inflow region bbox: min={inflow_region_min}, max={inflow_region_max}")

		base_inflow_num_points = getattr(args, 'inflow_num_points_base', None)
		if base_inflow_num_points is None:
			initial_num_points = gaussians.get_xyz.shape[0]
			base_inflow_num_points = max(1, int(initial_num_points * insert_ratio))
			args.inflow_num_points_base = base_inflow_num_points
			print(f"  [Inflow] Set base_inflow_num_points={base_inflow_num_points} (initial_num_points={initial_num_points}, insert_ratio={insert_ratio})")
		else:
			print(f"  [Inflow] Using cached base_inflow_num_points={base_inflow_num_points}")

		inflow_num_points = base_inflow_num_points

		if existing_inflow_gaussians is not None:
			prev_points_per_group = existing_inflow_gaussians.num_points_per_group
			if prev_points_per_group != inflow_num_points:
				print(f"  [Warning] existing_inflow_gaussians.num_points_per_group={prev_points_per_group} "
					  f"!= base_inflow_num_points={inflow_num_points}, using base value for new InflowGaussians.")
		
		num_continue_groups = 0
		if existing_inflow_gaussians is not None:
			num_continue_groups = existing_inflow_gaussians.num_groups
			print(f"  Found {num_continue_groups} existing inflow groups to continue training")
		
		if existing_inflow_gaussians is None:
			num_new_groups = max(0, end_frame - start_frame - 1)
		else:
			num_new_groups = 1
		
		total_num_groups = num_continue_groups + num_new_groups
		
		if total_num_groups <= 0:
			print(f"  Warning: total_num_groups={total_num_groups} <= 0, skipping inflow initialization")
			return None
		
		print(f"  Creating {total_num_groups} groups of inflow points:")
		if num_continue_groups > 0:
			print(f"    - {num_continue_groups} groups from previous window (to continue training)")
		print(f"    - {num_new_groups} new group for frame {end_frame-1}")
		
		inflow_gaussians = InflowGaussians(
			num_groups=total_num_groups,
			num_points_per_group=inflow_num_points,
			lengths_tensor=lengths_tensor,
			inflow_ratio=inflow_ratio,
			device=device,
			gaussians_template=gaussians,
			coord_trans=coord_trans,
			inflow_region_min=inflow_region_min,
			inflow_region_max=inflow_region_max
		).to(device)
		
		if existing_inflow_gaussians is not None:
			with torch.no_grad():
				for i in range(num_continue_groups):
					inflow_gaussians._xyz_groups[i].data.copy_(existing_inflow_gaussians._xyz_groups[i].data)
					inflow_gaussians._opacity_groups[i].data.copy_(existing_inflow_gaussians._opacity_groups[i].data)
					inflow_gaussians._scaling_groups[i].data.copy_(existing_inflow_gaussians._scaling_groups[i].data)
					inflow_gaussians._rotation_groups[i].data.copy_(existing_inflow_gaussians._rotation_groups[i].data)
					inflow_gaussians._features_dc_groups[i].data.copy_(existing_inflow_gaussians._features_dc_groups[i].data)
					inflow_gaussians._features_rest_groups[i].data.copy_(existing_inflow_gaussians._features_rest_groups[i].data)
					
					inflow_gaussians._initialized_groups[i] = existing_inflow_gaussians._initialized_groups[i]
					inflow_gaussians._group_origin_frames[i] = existing_inflow_gaussians._group_origin_frames[i]
			
			print(f"  Copied {num_continue_groups} groups from existing inflow_gaussians")
		
		if visualize_inflow_region:
			print("\n[Step 2.6] Visualizing Inflow Region in all camera views...")
			try:
				from gaussian_renderer import render as render_func
				visualize_inflow_region_all_cameras(
					gaussians, scene, coord_trans, lengths_tensor, 
					savedir, render_func, pipe, background,
					inflow_region_min=inflow_region_min,
					inflow_region_max=inflow_region_max
				)
			except Exception as e:
				print(f"  Warning: Inflow region visualization: {e}")
				import traceback
				traceback.print_exc()
			return None
		
		print("\n[Step 2.7] Inflow Gaussians will be initialized during training from T-1 frame GS...")
		
		lr_map = {}
		for param_group in gaussians.optimizer.param_groups:
			param_name = param_group.get('name', '')
			if param_name:
				lr_map[param_name] = param_group['lr']
		
		from argparse import ArgumentParser
		from arguments import OptimizationParams
		op = OptimizationParams(ArgumentParser())
		opt = op.extract(args)
		
		for group_idx in range(total_num_groups):
			xyz_lr = lr_map.get('xyz', opt.position_lr_init * gaussians.spatial_lr_scale)
			gaussians.optimizer.add_param_group({
				'params': [inflow_gaussians._xyz_groups[group_idx]],
				'lr': xyz_lr,
				'name': f'inflow_xyz_{group_idx}'
			})
			
			features_dc_lr = lr_map.get('f_dc', opt.feature_lr)
			gaussians.optimizer.add_param_group({
				'params': [inflow_gaussians._features_dc_groups[group_idx]],
				'lr': features_dc_lr,
				'name': f'inflow_f_dc_{group_idx}'
			})
			
			features_rest_lr = lr_map.get('f_rest', opt.feature_lr / 20.0)
			gaussians.optimizer.add_param_group({
				'params': [inflow_gaussians._features_rest_groups[group_idx]],
				'lr': features_rest_lr,
				'name': f'inflow_f_rest_{group_idx}'
			})
			
			opacity_lr = lr_map.get('opacity', opt.opacity_lr)
			gaussians.optimizer.add_param_group({
				'params': [inflow_gaussians._opacity_groups[group_idx]],
				'lr': opacity_lr,
				'name': f'inflow_opacity_{group_idx}'
			})
			
			scaling_lr = lr_map.get('scaling', opt.scaling_lr)
			gaussians.optimizer.add_param_group({
				'params': [inflow_gaussians._scaling_groups[group_idx]],
				'lr': scaling_lr,
				'name': f'inflow_scaling_{group_idx}'
			})
			
			rotation_lr = lr_map.get('rotation', opt.rotation_lr)
			gaussians.optimizer.add_param_group({
				'params': [inflow_gaussians._rotation_groups[group_idx]],
				'lr': rotation_lr,
				'name': f'inflow_rotation_{group_idx}'
			})
		
		print(f"  Added {6 * total_num_groups} parameter groups for inflow Gaussians to optimizer")
		print(f"    Learning rates: xyz={xyz_lr:.6f}, f_dc={features_dc_lr:.6f}, f_rest={features_rest_lr:.6f}, opacity={opacity_lr:.6f}, scaling={scaling_lr:.6f}, rotation={rotation_lr:.6f}")
		
		# Save initial inflow Gaussians to VTK (after initialization and pretraining)
		print(f"\n[Step 2.8] Saving initial Inflow Gaussians to VTK...")
		vtk_inflow_init_dir = os.path.join(savedir, "vis_train", "inflow_gaussians_init_vtk")
		os.makedirs(vtk_inflow_init_dir, exist_ok=True)
	
	return inflow_gaussians


def _forward_simulation(gaussians, coord_trans, lengths_tensor, vel_models, grid_points,
						grid_shape, dt, start_frame, end_frame, advect_frame_num, 
						inflow_gaussians, inflow_ratio, epoch):
	"""
	forward pass：Gaussianpoint cloud
	
	Args:
		start_frame: start frame（used forvel_models）
		end_frame: end frame（contains）
		advect_frame_num: epochadvectframe（start_frame）
	
	Returns:
		traj_sim: list（Sim Space）
		vel_grids: list
	"""
	with torch.no_grad():
		xyz_world_0 = gaussians.get_xyz.detach().clone()
		xyz_smoke_0 = coord_trans.world2smoke(xyz_world_0)
		xyz_sim_0 = xyz_smoke_0 * lengths_tensor
		
		traj_sim = [xyz_sim_0]
		vel_grids = []
		current_sim_pos = xyz_sim_0
		
		nx, ny, nz = grid_shape
		
		for t in range(advect_frame_num - 1):
			if inflow_ratio > 0 and inflow_gaussians is not None:
				group_idx = t
				if group_idx < inflow_gaussians.num_groups:
					if not inflow_gaussians.is_group_initialized(group_idx):
						inflow_region_min = inflow_gaussians.inflow_region_min
						inflow_region_max = inflow_gaussians.inflow_region_max
						
						aabb_min_sim = inflow_region_min * lengths_tensor
						aabb_max_sim = inflow_region_max * lengths_tensor
						
						mask_min = (current_sim_pos >= aabb_min_sim.unsqueeze(0)).all(dim=1)
						mask_max = (current_sim_pos <= aabb_max_sim.unsqueeze(0)).all(dim=1)
						inflow_mask = mask_min & mask_max
						inflow_indices = torch.where(inflow_mask)[0]
						
						if len(inflow_indices) > 0:
							num_points_needed = inflow_gaussians.num_points_per_group
							
							if len(inflow_indices) >= num_points_needed:
								selected_indices = torch.randperm(len(inflow_indices), device=inflow_gaussians.device)[:num_points_needed]
								selected_inflow_indices = inflow_indices[selected_indices]
							else:
								selected_inflow_indices = inflow_indices[torch.randint(len(inflow_indices), (num_points_needed,), device=inflow_gaussians.device)]
							
							selected_pos_sim = current_sim_pos[selected_inflow_indices]
							
							selected_pos_smoke = selected_pos_sim / lengths_tensor
							selected_pos_world = coord_trans.smoke2world(selected_pos_smoke)
							
							with torch.no_grad():
								inflow_gaussians._xyz_groups[group_idx].data.copy_(selected_pos_world)
							
							orig_num_points = gaussians.get_xyz.shape[0]
							
							with torch.no_grad():
								for i, global_idx in enumerate(selected_inflow_indices):
									if global_idx < orig_num_points:
										inflow_gaussians._opacity_groups[group_idx].data[i] = gaussians._opacity[global_idx].clone()
										inflow_gaussians._scaling_groups[group_idx].data[i] = gaussians._scaling[global_idx].clone()
										inflow_gaussians._rotation_groups[group_idx].data[i] = gaussians._rotation[global_idx].clone()
										inflow_gaussians._features_dc_groups[group_idx].data[i] = gaussians._features_dc[global_idx].clone()
										inflow_gaussians._features_rest_groups[group_idx].data[i] = gaussians._features_rest[global_idx].clone()
									else:
										inflow_idx = global_idx - orig_num_points
										inflow_points_per_group = inflow_gaussians.num_points_per_group
										source_group_idx = inflow_idx // inflow_points_per_group
										source_point_idx = inflow_idx % inflow_points_per_group
										
										if source_group_idx < group_idx:
											inflow_gaussians._opacity_groups[group_idx].data[i] = inflow_gaussians._opacity_groups[source_group_idx].data[source_point_idx].clone()
											inflow_gaussians._scaling_groups[group_idx].data[i] = inflow_gaussians._scaling_groups[source_group_idx].data[source_point_idx].clone()
											inflow_gaussians._rotation_groups[group_idx].data[i] = inflow_gaussians._rotation_groups[source_group_idx].data[source_point_idx].clone()
											inflow_gaussians._features_dc_groups[group_idx].data[i] = inflow_gaussians._features_dc_groups[source_group_idx].data[source_point_idx].clone()
											inflow_gaussians._features_rest_groups[group_idx].data[i] = inflow_gaussians._features_rest_groups[source_group_idx].data[source_point_idx].clone()
							
							inflow_gaussians.mark_group_initialized(group_idx)
							
							global_frame_idx = start_frame + group_idx + 1
							inflow_gaussians.set_group_origin_frame(group_idx, global_frame_idx)
							
							if epoch == 1 or (epoch % 10 == 0 and group_idx == 0):
								print(f"  [Inflow Init] Initialized group {group_idx} with {len(selected_inflow_indices)} points from frame {t} GS (before advect), origin_frame={global_frame_idx}")
			
			# Evaluate velocity field
			v_flat = vel_models[t](grid_points)
			v_vol = v_flat.view(nx, ny, nz, 3).permute(3, 2, 1, 0).unsqueeze(0)
			vel_grids.append(v_vol)
			
			# Sample velocity and advect
			norm_pos = current_sim_pos.clone()
			norm_pos = 2 * (norm_pos / lengths_tensor) - 1.0
			grid_in = norm_pos.view(1, 1, 1, -1, 3)
			v_part = F.grid_sample(v_vol, grid_in, align_corners=True, mode='bilinear')
			v_part = v_part.view(3, -1).permute(1, 0)
			
			next_sim_pos = current_sim_pos + v_part * dt
			
			if inflow_ratio > 0 and inflow_gaussians is not None:
				group_idx = t
				if group_idx < inflow_gaussians.num_groups:
					inflow_xyz_sim = inflow_gaussians.get_group_xyz_sim(group_idx)
					next_sim_pos = torch.cat([next_sim_pos, inflow_xyz_sim], dim=0)
					
					if epoch == 1 or (epoch % 10 == 0 and t == 0):
						print(f"  [Inflow] Added group {group_idx} ({inflow_xyz_sim.shape[0]} points) after advecting to frame {start_frame + t + 1} (global frame {start_frame + t + 1}, total: {next_sim_pos.shape[0]})")
				else:
					if epoch == 1 or (epoch % 10 == 0 and t == 0):
						print(f"  [Inflow] Warning: group {group_idx} out of range (max: {inflow_gaussians.num_groups - 1}), skipping at frame {start_frame + t + 1}")
			
			traj_sim.append(next_sim_pos)
			current_sim_pos = next_sim_pos
	
	return traj_sim, vel_grids


def train_velocity_model_with_gaussian(args, savedir=None, scale: int = 1, gaussian_ckpt_path: str = None, dt = None, visualize_opacity=False, inflow_ratio=0.05, insert_ratio=0.01, visualize_inflow_region=False, start_frame=None, end_frame=None,
										initial_gaussians_state=None, initial_velocity_models_dict=None, initial_velocity_optimizers_dict=None, initial_velocity_schedulers_dict=None,
										freeze_initial_gaussians_xyz=False, velocity_optimization_counts=None, use_individual_optimizers=False,
										use_progressive_training=True, num_epochs_override=None, existing_inflow_gaussians=None):
	"""
	Train velocity field using Gaussian Splatting rendering with Analytic Gradients.
	
	Args:
		args: training arguments
		savedir: output directory
		scale: scaling
		gaussian_ckpt_path:  Gaussian checkpoint path（optional）。，。
		inflow_ratio: y direction inflow ，used for Gaussian 
		insert_ratio: ，used for Gaussian 
		visualize_inflow_region: visualization inflow region
		start_frame: start frame（None，frame0framestart）
		end_frame: end frame（None，frame，contains）
		initial_gaussians_state: optional GS （ capture() ，used for frame_1, frame_2, ...）
		initial_velocity_models_dict: optional velocity  {frame_idx: model}
		initial_velocity_optimizers_dict: optionaloptimizer {frame_idx: optimizer}
		initial_velocity_schedulers_dict: optional {frame_idx: scheduler}
		freeze_initial_gaussians_xyz:  GS position（frame_0 position）
		velocity_optimization_counts: eachvelocity fieldin
		use_individual_optimizers: useoptimizer（eachvelocity field）
		use_progressive_training: use（ advect frame）
		num_epochs_override:  args.num_epochs  epoch （）
	"""
	device = set_device(args)
	
	# Load Meta Info to get frame_num and determine frame range
	with open(os.path.join(args.datadir, 'info.json'), 'r') as fp:
		meta = json.load(fp)
		train_video = meta['train_videos'][0]
		total_frame_num = train_video['frame_num']
		args.frame_num = total_frame_num
	
	if start_frame is None:
		start_frame = 0
	if end_frame is None:
		end_frame = total_frame_num
	
	if start_frame < 0 or end_frame > total_frame_num or start_frame >= end_frame:
		raise ValueError(f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}, total_frame_num={total_frame_num}")
	
	frame_range = end_frame - start_frame
	print(f"\n[Frame Range] Training from frame {start_frame} to frame {end_frame-1} (total: {frame_range} frames)")
	
	prev_velocity_model_for_advection = None
	advection_grid_points = None
	advection_coord_trans = None
	advection_lengths_tensor = None
	advection_dt = None
	advection_grid_shape = None
	initial_xyz_world_i = None
	
	# ---------------------------------------------------------
	# 1. Train Start Frame Gaussian (Coarse) or Load from Checkpoint
	# ---------------------------------------------------------
	if initial_gaussians_state is not None:
		print(f"\n[Step 1] Initializing Frame {start_frame} Gaussian from Frame {start_frame-1}...")
		from argparse import ArgumentParser
		from arguments import ModelParams, ModelHiddenParams, OptimizationParams
		from scene.gaussian_model import GaussianModel
		
		parser = ArgumentParser()
		model_params_obj = ModelParams(parser, sentinel=True)
		hyperparam_obj = ModelHiddenParams(parser)
		op = OptimizationParams(parser)
		
		model_params = model_params_obj.extract(args)
		hyperparam = hyperparam_obj.extract(args)
		opt = op.extract(args)
		
		gaussians = GaussianModel(model_params.sh_degree, hyperparam)
		
		gaussians.restore(initial_gaussians_state, opt)
		
		with open(os.path.join(args.datadir, 'info.json'), 'r') as fp:
			meta = json.load(fp)
			voxel_tran = np.float32(meta['voxel_matrix'])
			voxel_tran = np.stack([voxel_tran[:,2],voxel_tran[:,1],voxel_tran[:,0],voxel_tran[:,3]],axis=1)
			voxel_scale = np.broadcast_to(meta['voxel_scale'],[3])
			voxel_scale = voxel_scale * args.scene_scale
			voxel_tran[:3,3] *= args.scene_scale
		
		coord_trans_temp = CoordinateTransform(voxel_tran, voxel_scale, device)
		lengths = np.array([args.Nx, args.Ny, args.Nz])
		lengths_tensor_temp = torch.from_numpy(lengths).float().to(device)
		
		if initial_velocity_models_dict and (start_frame - 1) in initial_velocity_models_dict:
			prev_velocity_model_for_advection = initial_velocity_models_dict[start_frame - 1]
			print(f"  Using velocity model for frame {start_frame - 1} from initial_velocity_models_dict")
			print(f"  Will allow gradients to flow back to this velocity model")
			
			s = float(scale)
			nx, ny, nz = int(args.Nx/s), int(args.Ny/s), int(args.Nz/s)
			grid_shape_temp = (nx, ny, nz)
			min_corner = np.zeros(3)
			max_corner = lengths
			grid_points_np = utils_grid.generate_gridpoints(grid_shape_temp, min_corner, max_corner)
			grid_points_temp = torch.from_numpy(grid_points_np).float().to(device)
			
			advection_grid_points = grid_points_temp
			advection_coord_trans = coord_trans_temp
			advection_lengths_tensor = lengths_tensor_temp
			advection_grid_shape = grid_shape_temp
			if dt is None:
				advection_dt = (args.sim_steps / s)
			else:
				advection_dt = dt
			
			initial_xyz_world_i = gaussians.get_xyz.detach().clone()
			xyz_smoke_i = coord_trans_temp.world2smoke(initial_xyz_world_i)
			xyz_sim_i = xyz_smoke_i * lengths_tensor_temp
			
			with torch.no_grad():
				v_flat = prev_velocity_model_for_advection(grid_points_temp)
			v_vol = v_flat.view(nx, ny, nz, 3).permute(3, 2, 1, 0).unsqueeze(0)
			
			norm_pos = xyz_sim_i.clone()
			norm_pos = 2 * (norm_pos / lengths_tensor_temp) - 1.0
			grid_in = norm_pos.view(1, 1, 1, -1, 3)
			v_part = F.grid_sample(v_vol, grid_in, align_corners=True, mode='bilinear')
			v_part = v_part.view(3, -1).permute(1, 0)
			
			xyz_sim_i_plus_1 = xyz_sim_i + v_part * advection_dt
			
			xyz_smoke_i_plus_1 = xyz_sim_i_plus_1 / lengths_tensor_temp
			xyz_world_i_plus_1 = coord_trans_temp.smoke2world(xyz_smoke_i_plus_1)
			
			gaussians._xyz.data = xyz_world_i_plus_1.detach().requires_grad_(True)
			
			print(f"  Advected {initial_xyz_world_i.shape[0]} points from frame {start_frame-1} to frame {start_frame}")
		else:
			print(f"  Warning: Velocity model for frame {start_frame - 1} not found in initial_velocity_models_dict")
			print(f"    initial_velocity_models_dict keys: {list(initial_velocity_models_dict.keys()) if initial_velocity_models_dict else 'None'}")
			print(f"    Will use frame {start_frame-1} GS position directly (no advection)")
		
		if freeze_initial_gaussians_xyz:
			pass
		
		from scene import Scene
		if not hasattr(model_params, 'eval'):
			model_params.eval = getattr(args, 'eval', True)
		if not hasattr(model_params, 'white_background'):
			model_params.white_background = getattr(args, 'white_background', True)
		if not hasattr(model_params, 'extension'):
			model_params.extension = getattr(args, 'extension', '.png')
		if not hasattr(model_params, 'images'):
			model_params.images = getattr(args, 'images', 'images')
		if not hasattr(model_params, 'llffhold'):
			model_params.llffhold = getattr(args, 'llffhold', 8)
		if not hasattr(model_params, 'add_points'):
			model_params.add_points = getattr(args, 'add_points', False)
		if not hasattr(model_params, 'num_init_points'):
			model_params.num_init_points = getattr(args, 'num_init_points', 2000)
		if not hasattr(model_params, 'half_res'):
			model_params.half_res = getattr(args, 'half_res', False)
		scene = Scene(model_params, gaussians, load_iteration=None, shuffle=False)
	else:
		gaussians, scene = _load_or_train_frame_gaussian(args, savedir, gaussian_ckpt_path, start_frame)
	
	# ---------------------------------------------------------
	# 2. Setup Velocity Model
	# ---------------------------------------------------------
	print("\n[Step 2] Initializing Velocity Field...")
	
	# Setup velocity models and related components
	prev_frame_velocity_model = None
	if start_frame > 0 and initial_velocity_models_dict is not None:
		prev_frame_idx = start_frame - 1
		if prev_frame_idx in initial_velocity_models_dict:
			prev_frame_velocity_model = initial_velocity_models_dict[prev_frame_idx]
			print(f"  Found previous frame velocity model (frame {prev_frame_idx}) for NS loss calculation")
	
	if use_individual_optimizers:
		vel_models, vel_optimizers, vel_schedulers, grid_points, grid_shape, lengths_tensor, \
			coord_trans, background, pipe, dataset, dt = _setup_velocity_models_with_individual_optimizers(
				args, device, scale, start_frame, end_frame, dt,
				velocity_optimization_counts=velocity_optimization_counts,
				existing_velocity_models=initial_velocity_models_dict,
				existing_optimizers=initial_velocity_optimizers_dict,
				existing_schedulers=initial_velocity_schedulers_dict
			)
		vel_optimizer = None
		vel_scheduler = None
	else:
		vel_models, vel_optimizer, vel_scheduler, grid_points, grid_shape, lengths_tensor, \
			coord_trans, background, pipe, dataset, dt = _setup_velocity_models(
				args, device, scale, start_frame, end_frame, dt
			)
		vel_optimizers = None
		vel_schedulers = None
	
	# Tensorboard & Checkpointing
	writer = SummaryWriter(log_dir=savedir)
	os.makedirs(os.path.join(savedir, "ckpt"), exist_ok=True)
	os.makedirs(os.path.join(savedir, "vis_train"), exist_ok=True)
	
	# Gradient Decay Ratio for BPTT
	grad_decay = getattr(args, 'grad_decay', 0.9)
	
	print("\n[Pre-computing] Camera mapping for each frame...")
	frame_to_cameras, train_cameras = _precompute_camera_mapping(scene, total_frame_num, args, dataset)
	
	filtered_frame_to_cameras = {}
	for t in range(start_frame, end_frame):
		if t in frame_to_cameras:
			filtered_frame_to_cameras[t - start_frame] = frame_to_cameras[t]
		else:
			filtered_frame_to_cameras[t - start_frame] = []
	frame_to_cameras = filtered_frame_to_cameras
	
	# ---------------------------------------------------------
	# 2.5. Initialize Inflow Gaussians (if enabled)
	# ---------------------------------------------------------
	inflow_gaussians = _setup_inflow_gaussians(
		args, gaussians, coord_trans, lengths_tensor, start_frame, end_frame,
		inflow_ratio, insert_ratio, device, savedir, scene, frame_to_cameras,
		pipe, background, visualize_inflow_region, existing_inflow_gaussians=existing_inflow_gaussians
	)
	
	if inflow_gaussians is None and visualize_inflow_region:
		return
	
	# ---------------------------------------------------------
	# 3. Training Loop
	# ---------------------------------------------------------
	num_epochs = num_epochs_override if num_epochs_override is not None else args.num_epochs
	print(f"\n[Step 3] Starting Velocity Training ({num_epochs} Epochs)...")
	if not use_progressive_training:
		print("  Using full frame range from the start (no progressive training)")
	
	# Extract grid dimensions for use in training loop
	nx, ny, nz = grid_shape
	s = float(scale)
	delta_t = dt
	
	frame_range = end_frame - start_frame
	
	total_train_time = 0.0
	total_checkpoint_time = 0.0
	total_visualization_time = 0.0
	
	for epoch in tqdm(range(1, num_epochs + 1)):
		epoch_train_start = datetime.now()
		vel_models.train()
		if use_individual_optimizers and vel_optimizers is not None:
			for global_frame_idx in vel_optimizers:
				vel_optimizers[global_frame_idx].zero_grad()
		else:
			vel_optimizer.zero_grad()
		gaussians.optimizer.zero_grad() # If we want to refine Gaussians too
		
		# Progressive training: gradually increase the number of frames to advect
		# In the first half of epochs, advect_frame_num linearly increases from 2 to frame_range
		# In the second half, use all frames in the range
		if use_progressive_training:
			if epoch < num_epochs / 2:
				advect_frame_num = int(2 + (frame_range - 2) * epoch / (num_epochs / 2))
			else:
				advect_frame_num = frame_range
		else:
			advect_frame_num = frame_range
		
		# Select frames for NS loss calculation (random sampling each epoch)
		import random
		# Available frame range: [1, advect_frame_num-1), length = max(0, advect_frame_num - 2)
		available_frames = list(range(1, advect_frame_num - 2)) if advect_frame_num > 3 else []
		if len(available_frames) > 0:
			n = min(10, len(available_frames))
			selected_frame_idx = random.sample(available_frames, n)
		else:
			selected_frame_idx = []
		
		# === A. Forward Simulation (Cache Trajectory) ===
		traj_sim, vel_grids = _forward_simulation(
			gaussians, coord_trans, lengths_tensor, vel_models, grid_points,
			grid_shape, dt, start_frame, end_frame, advect_frame_num, 
			inflow_gaussians, inflow_ratio, epoch
		)

		# === C. Backward Pass (BPTT with Analytic Gradients) ===
		
		# Accumulator for gradients from t+1 (upstream)
		# Note: traj_sim[-1] may have different size due to inflow points, so we need to handle this dynamically
		grad_upstream_sim = torch.zeros_like(traj_sim[-1])
		total_epoch_loss = 0.0
		total_l1_loss = 0.0
		total_dssim_loss = 0.0
		total_reg_loss = 0.0
		total_ns_loss = 0.0
		frame_losses = []  # Store loss for each frame
		
		# Check and set default values for loss weights
		lambda_regular = getattr(args, 'lambda_regular', 1.0)
		lambda_nse = getattr(args, 'lambda_nse', 1e-6)
		
		# Iterate backwards: advect_frame_num-1 -> 0 (only process frames that were advected)
		for t in range(advect_frame_num - 1, -1, -1):
			
			# --- 1. Render Loss at Frame t ---
			# Get current position in Sim Space
			curr_pos_sim = traj_sim[t]
			
			# Determine which inflow groups should be included at frame t
			has_inflow = (t > 0 and inflow_ratio > 0 and inflow_gaussians is not None)
			if has_inflow:
				max_group_idx = min(t, inflow_gaussians.num_groups)
				inflow_group_indices = list(range(max_group_idx))
				
				newly_added_group_idx = t - 1 if t > 0 and (t - 1) < inflow_gaussians.num_groups else None
			else:
				inflow_group_indices = []
				newly_added_group_idx = None
   
			if has_inflow:
				# Expected points: Original + len(inflow_group_indices) * points_per_group
				num_orig = gaussians.get_xyz.shape[0]
				num_inflow = inflow_gaussians.num_points_per_group
				expected_count = num_orig + len(inflow_group_indices) * num_inflow
				
				if curr_pos_sim.shape[0] < expected_count:
					print(f"  [Inflow Warning] Missing inflow points in trajectory! Add them now for rendering consistency. {curr_pos_sim.shape[0]} < {expected_count} at frame {t}")
					# Missing inflow points in trajectory! Add them now for rendering consistency.
					# We need to add all missing groups from the ones that should be present.
					# 
					
					# NOTE: This creates a new tensor, make sure it's valid for gradient flow if needed.
					# (Here we are just using it for `curr_pos_world` calculation which requires grad).
					
					missing_groups_start = (curr_pos_sim.shape[0] - num_orig) // num_inflow
					
					max_group_to_add = min(t, inflow_gaussians.num_groups)
					if missing_groups_start < max_group_to_add:
						pts_to_add = []
						for g_idx in range(missing_groups_start, max_group_to_add):
							if g_idx < inflow_gaussians.num_groups:
								pts_to_add.append(inflow_gaussians.get_group_xyz_sim(g_idx))
						
						if pts_to_add:
							curr_pos_sim = torch.cat([curr_pos_sim] + pts_to_add, dim=0)
			
			# Convert Sim Space -> Smoke Space -> World Space for rendering
			curr_pos_smoke = curr_pos_sim / lengths_tensor  # Sim [0, lengths] -> Smoke [0, 1]
			
			# Get pre-computed matching cameras for this frame
			# Note: In progressive training, advect_frame_num may exceed frame_range
			# So we need to handle frames beyond the original frame_range
			relative_frame_idx = t
			if relative_frame_idx < len(frame_to_cameras):
				matching_cameras = frame_to_cameras[relative_frame_idx]
			else:
				# Frame beyond frame_range (progressive training)
				# Map to the last frame's cameras or use empty list
				if len(frame_to_cameras) > 0:
					matching_cameras = frame_to_cameras[len(frame_to_cameras) - 1]
				else:
					matching_cameras = []
			
			# Accumulate gradients from all matching cameras
			grad_render_world_list = []
			total_loss_for_frame = 0.0
			total_l1_for_frame = 0.0
			total_dssim_for_frame = 0.0
			
			for cam_idx, viewpoint_cam in matching_cameras:
				# Convert smoke -> world (need to recreate for each camera to get fresh gradients)
				# Important: We need this tensor to be a leaf for autograd to catch gradients
				# Each iteration creates a NEW tensor, so gradients won't accumulate across iterations
				curr_pos_world = coord_trans.smoke2world(curr_pos_smoke)
				curr_pos_world.requires_grad_(True) # We want gradient w.r.t this world position
				
				orig_num_points = gaussians.get_xyz.shape[0]
				gt_image = viewpoint_cam.original_image.cuda()
				
				# For inflow case, render separately:
				# 1. Original GS + 1->T-1 inflow GS (advected together): outside mask match GT, inside mask match background
				# 2. T-th inflow GS (newly added): inside mask match GT, outside mask match background
				wrapped_advected = None
				wrapped_newly_added = None
				wrapped_gaussians = None
				advected_group_indices = []
				
				if has_inflow:
					# Determine which groups are advected together (all except newly_added_group_idx)
					advected_group_indices = [idx for idx in inflow_group_indices if idx != newly_added_group_idx]
					
					# Split positions
					orig_pos_world = curr_pos_world[:orig_num_points]
					inflow_pos_world = curr_pos_world[orig_num_points:]
					
					# Unified rendering: render all GS together (originGS + all Inflow)
					# Use ExtendedGaussianWrapper to merge all GS
					wrapped_gaussians = ExtendedGaussianWrapper(gaussians, curr_pos_world, inflow_gaussians, inflow_group_indices)
					
					# Render all GS together
					render_pkg = render(viewpoint_cam, wrapped_gaussians, pipe, background, stage="coarse")
					image = render_pkg["render"]  # [3, H, W]
					
					# Standard rendering loss: match GT
					l1 = l1_loss(image, gt_image)
					dssim = 1.0 - ssim(image, gt_image)
					loss = l1 * (1.0 - args.lambda_dssim) + args.lambda_dssim * dssim
					
					# # Additional position loss: encourage originGS and 1->T-1 GS positions to be outside inflow region
					# # Get inflow region boundaries in Sim Space
					# inflow_region_min = getattr(args, 'inflow_region_min', [0.0, 0.1, 0.0])
					# inflow_region_max = getattr(args, 'inflow_region_max', [1.0, 0.3, 1.0])
					
					# # Convert to tensor if needed
					# if isinstance(inflow_region_min, (list, tuple)):
					# 	inflow_region_min = torch.tensor(inflow_region_min, device=device, dtype=torch.float32)
					# if isinstance(inflow_region_max, (list, tuple)):
					# 	inflow_region_max = torch.tensor(inflow_region_max, device=device, dtype=torch.float32)
					
					# # Calculate inflow region boundaries in Sim Space
					# aabb_min_sim = inflow_region_min * lengths_tensor  # [3]
					# aabb_max_sim = inflow_region_max * lengths_tensor  # [3]
					
					# # Get positions in Sim Space for originGS and advected inflow groups (1->T-1)
					# # Split positions: originGS + advected inflow groups
					# inflow_points_per_group = inflow_gaussians.num_points_per_group
					
					# # Get originGS positions in Sim Space
					# orig_pos_sim = curr_pos_sim[:orig_num_points]  # [N_orig, 3]
					
					# # Get advected inflow groups positions in Sim Space
					# if len(advected_group_indices) > 0:
					# 	# Calculate start and end indices for advected groups
					# 	advected_start = orig_num_points
					# 	advected_end = orig_num_points + len(advected_group_indices) * inflow_points_per_group
					# 	advected_pos_sim = curr_pos_sim[advected_start:advected_end]  # [N_advected, 3]
						
					# 	# Combine originGS and advected inflow groups positions
					# 	positions_to_check_sim = torch.cat([orig_pos_sim, advected_pos_sim], dim=0)  # [N_orig + N_advected, 3]
					# else:
					# 	# No advected inflow groups, only originGS
					# 	positions_to_check_sim = orig_pos_sim  # [N_orig, 3]
					
					# # Check which points are inside the inflow region
					# # A point is inside if: aabb_min_sim <= point <= aabb_max_sim (element-wise)
					# inside_min = (positions_to_check_sim >= aabb_min_sim.unsqueeze(0)).all(dim=-1)  # [N]
					# inside_max = (positions_to_check_sim <= aabb_max_sim.unsqueeze(0)).all(dim=-1)  # [N]
					# inside_region = inside_min & inside_max  # [N] bool
					
					# # Penalty: compute distance from points inside region to the nearest boundary
					# # For points inside, compute distance to boundary and penalize
					# if inside_region.any():
					# 	points_inside = positions_to_check_sim[inside_region]  # [N_inside, 3]
						
					# 	# Compute distance to each boundary face
					# 	# Distance to min boundary: point - aabb_min_sim
					# 	dist_to_min = points_inside - aabb_min_sim.unsqueeze(0)  # [N_inside, 3]
					# 	# Distance to max boundary: aabb_max_sim - point
					# 	dist_to_max = aabb_max_sim.unsqueeze(0) - points_inside  # [N_inside, 3]
						
					# 	# For each point, find the minimum distance to any boundary face
					# 	# This gives the "depth" of the point inside the region
					# 	# We want to penalize points that are deep inside the region
					# 	dist_to_boundary_min = torch.minimum(dist_to_min, dist_to_max)  # [N_inside, 3]
					# 	# Take the minimum across all dimensions (closest face)
					# 	min_dist_to_boundary = dist_to_boundary_min.min(dim=-1)[0]  # [N_inside]
						
					# 	# Penalty: use the minimum distance to boundary
					# 	# Points deeper inside the region get higher penalty
					# 	# We can use squared distance for stronger penalty on deep points
					# 	position_penalty = (min_dist_to_boundary ** 2).mean()  # scalar
						
					# 	# Add weighted position loss
					# 	lambda_position = getattr(args, 'lambda_position_inflow', 1.0)  # Weight for position loss
					# 	loss = loss + lambda_position * position_penalty
				else:
					# No inflow points, use regular wrapper
					wrapped_gaussians = GaussianOverrideWrapper(gaussians, curr_pos_world)
					
					# Render
					render_pkg = render(viewpoint_cam, wrapped_gaussians, pipe, background, stage="coarse")
					image = render_pkg["render"]
					
					# Loss
					l1 = l1_loss(image, gt_image)
					dssim = 1.0 - ssim(image, gt_image)
					loss = l1 * (1.0 - args.lambda_dssim) + args.lambda_dssim * dssim
				
				# Spherical regularization loss: encourage Gaussians to be spherical (not elongated)
				# This penalizes the variance of scaling across the three dimensions
				if hasattr(args, 'lambda_spherical') and args.lambda_spherical > 0:
					# Get activated scaling (after activation function) for original Gaussians
					scaling_activated = gaussians.scaling_activation(gaussians._scaling)  # [N, 3]
					# Compute mean scaling for each Gaussian (across 3 dimensions)
					scaling_mean = scaling_activated.mean(dim=1, keepdim=True)  # [N, 1]
					# Compute variance of scaling for each Gaussian
					scaling_variance = ((scaling_activated - scaling_mean) ** 2).mean(dim=1)  # [N]
					# Average variance across all Gaussians
					spherical_loss = scaling_variance.mean()
					
					# Also compute spherical loss for inflow Gaussians if they exist
					if has_inflow and inflow_gaussians is not None:
						inflow_spherical_losses = []
						for group_idx in inflow_group_indices:
							# Get activated scaling for this inflow group
							inflow_scaling_activated = inflow_gaussians.get_group_scaling(group_idx)  # [M, 3]
							# Compute mean scaling for each Gaussian in this group
							inflow_scaling_mean = inflow_scaling_activated.mean(dim=1, keepdim=True)  # [M, 1]
							# Compute variance of scaling for each Gaussian
							inflow_scaling_variance = ((inflow_scaling_activated - inflow_scaling_mean) ** 2).mean(dim=1)  # [M]
							# Average variance across all Gaussians in this group
							inflow_spherical_losses.append(inflow_scaling_variance.mean())
						
						# Average across all inflow groups
						if inflow_spherical_losses:
							inflow_spherical_loss = torch.stack(inflow_spherical_losses).mean()
							spherical_loss = spherical_loss + inflow_spherical_loss
					
					loss += args.lambda_spherical * spherical_loss
				
				total_loss_for_frame += loss.item()
				total_l1_for_frame += l1.item()
				total_dssim_for_frame += dssim.item()
				
				# Backward for Render
				# Note: Since curr_pos_world (and inflow_pos_world) are new tensors each iteration, 
				# backward() will only populate .grad for these specific tensor instances
				# For ExtendedGaussianWrapper, we need to manually handle gradient separation
				# because CUDA kernel backward expects gradients to match original parameter shapes
				loss.backward()
				
				# Get gradient from this camera and immediately detach/clone
				# Separate gradients for original points and inflow points
				if has_inflow:
					# Split gradients: original points + all inflow points (from multiple groups)
					# The order matches traj_sim[t]: [original points, group 0, group 1, ..., group t-1]
					if curr_pos_world.grad is not None:
						grad_render_world_orig = curr_pos_world.grad[:orig_num_points].detach().clone()
						grad_render_world_all_inflow = curr_pos_world.grad[orig_num_points:].detach().clone()
					else:
						grad_render_world_orig = torch.zeros_like(curr_pos_world[:orig_num_points])
						grad_render_world_all_inflow = torch.zeros_like(curr_pos_world[orig_num_points:])
					
					# IMPORTANT: Store combined gradient (original + inflow) for velocity field backpropagation
					grad_render_world_combined = torch.cat([grad_render_world_orig, grad_render_world_all_inflow], dim=0)
					grad_render_world_list.append(grad_render_world_combined)
					
					# Manually backpropagate gradients for merged attributes to original and inflow parameters
					# Now we use unified rendering, so all GS are rendered together via ExtendedGaussianWrapper
					# Get gradients from ExtendedGaussianWrapper (wrapped_gaussians contains all GS)
					if isinstance(wrapped_gaussians, ExtendedGaussianWrapper):
						# Get gradients from ExtendedGaussianWrapper
						if wrapped_gaussians._opacity.grad is not None:
							grad_opacity = wrapped_gaussians._opacity.grad
							grad_opacity_orig = grad_opacity[:orig_num_points]
							grad_opacity_inflow = grad_opacity[orig_num_points:]
							
							# Accumulate to original opacity
							if gaussians._opacity.grad is None:
								gaussians._opacity.grad = grad_opacity_orig.clone()
							else:
								gaussians._opacity.grad += grad_opacity_orig
							
							# Accumulate to inflow opacity (split by group)
							inflow_points_per_group = inflow_gaussians.num_points_per_group
							start_idx = 0
							for group_idx in inflow_group_indices:
								end_idx = start_idx + inflow_points_per_group
								grad_opacity_group = grad_opacity_inflow[start_idx:end_idx]
								if inflow_gaussians._opacity_groups[group_idx].grad is None:
									inflow_gaussians._opacity_groups[group_idx].grad = grad_opacity_group.clone()
								else:
									inflow_gaussians._opacity_groups[group_idx].grad += grad_opacity_group
								start_idx = end_idx
						
						# Similar handling for _scaling, _rotation, _features_dc, _features_rest
						if wrapped_gaussians._scaling.grad is not None:
							grad_scaling = wrapped_gaussians._scaling.grad
							grad_scaling_orig = grad_scaling[:orig_num_points]
							grad_scaling_inflow = grad_scaling[orig_num_points:]
							if gaussians._scaling.grad is None:
								gaussians._scaling.grad = grad_scaling_orig.clone()
							else:
								gaussians._scaling.grad += grad_scaling_orig
							inflow_points_per_group = inflow_gaussians.num_points_per_group
							start_idx = 0
							for group_idx in inflow_group_indices:
								end_idx = start_idx + inflow_points_per_group
								grad_scaling_group = grad_scaling_inflow[start_idx:end_idx]
								if inflow_gaussians._scaling_groups[group_idx].grad is None:
									inflow_gaussians._scaling_groups[group_idx].grad = grad_scaling_group.clone()
								else:
									inflow_gaussians._scaling_groups[group_idx].grad += grad_scaling_group
								start_idx = end_idx
						
						if wrapped_gaussians._rotation.grad is not None:
							grad_rotation = wrapped_gaussians._rotation.grad
							grad_rotation_orig = grad_rotation[:orig_num_points]
							grad_rotation_inflow = grad_rotation[orig_num_points:]
							if gaussians._rotation.grad is None:
								gaussians._rotation.grad = grad_rotation_orig.clone()
							else:
								gaussians._rotation.grad += grad_rotation_orig
							inflow_points_per_group = inflow_gaussians.num_points_per_group
							start_idx = 0
							for group_idx in inflow_group_indices:
								end_idx = start_idx + inflow_points_per_group
								grad_rotation_group = grad_rotation_inflow[start_idx:end_idx]
								if inflow_gaussians._rotation_groups[group_idx].grad is None:
									inflow_gaussians._rotation_groups[group_idx].grad = grad_rotation_group.clone()
								else:
									inflow_gaussians._rotation_groups[group_idx].grad += grad_rotation_group
								start_idx = end_idx
						
						if wrapped_gaussians._features_dc.grad is not None:
							grad_features_dc = wrapped_gaussians._features_dc.grad
							grad_features_dc_orig = grad_features_dc[:orig_num_points]
							grad_features_dc_inflow = grad_features_dc[orig_num_points:]
							if gaussians._features_dc.grad is None:
								gaussians._features_dc.grad = grad_features_dc_orig.clone()
							else:
								gaussians._features_dc.grad += grad_features_dc_orig
							inflow_points_per_group = inflow_gaussians.num_points_per_group
							start_idx = 0
							for group_idx in inflow_group_indices:
								end_idx = start_idx + inflow_points_per_group
								grad_features_dc_group = grad_features_dc_inflow[start_idx:end_idx]
								if inflow_gaussians._features_dc_groups[group_idx].grad is None:
									inflow_gaussians._features_dc_groups[group_idx].grad = grad_features_dc_group.clone()
								else:
									inflow_gaussians._features_dc_groups[group_idx].grad += grad_features_dc_group
								start_idx = end_idx
						
						if wrapped_gaussians._features_rest.grad is not None:
							grad_features_rest = wrapped_gaussians._features_rest.grad
							grad_features_rest_orig = grad_features_rest[:orig_num_points]
							grad_features_rest_inflow = grad_features_rest[orig_num_points:]
							if gaussians._features_rest.grad is None:
								gaussians._features_rest.grad = grad_features_rest_orig.clone()
							else:
								gaussians._features_rest.grad += grad_features_rest_orig
							inflow_points_per_group = inflow_gaussians.num_points_per_group
							start_idx = 0
							for group_idx in inflow_group_indices:
								end_idx = start_idx + inflow_points_per_group
								grad_features_rest_group = grad_features_rest_inflow[start_idx:end_idx]
								if inflow_gaussians._features_rest_groups[group_idx].grad is None:
									inflow_gaussians._features_rest_groups[group_idx].grad = grad_features_rest_group.clone()
								else:
									inflow_gaussians._features_rest_groups[group_idx].grad += grad_features_rest_group
								start_idx = end_idx
					
					# Split inflow gradients by group and accumulate to each group's parameters (for xyz)
					inflow_points_per_group = inflow_gaussians.num_points_per_group
					start_idx = 0
					
					for group_idx in inflow_group_indices:
						end_idx = start_idx + inflow_points_per_group
						grad_render_world_group = grad_render_world_all_inflow[start_idx:end_idx]
						
						# Convert inflow gradients analytically for:
						# world = (sim / lengths * scale) @ R_s2w.T + T_s2w
						grad_sim_inflow = (
							torch.matmul(grad_render_world_group, coord_trans.R_s2w)
							* (coord_trans.scale / lengths_tensor)
						).detach()
						
						# Accumulate gradient to inflow parameter (for optimizing inflow point positions)
						# Now we allow gradients for newly_added_group_idx as well
						if inflow_gaussians._xyz_groups[group_idx].grad is None:
							inflow_gaussians._xyz_groups[group_idx].grad = grad_sim_inflow.clone()
						else:
							inflow_gaussians._xyz_groups[group_idx].grad += grad_sim_inflow
						
						start_idx = end_idx
					
					# Clear gradients
					curr_pos_world.grad = None
				else:
					# No inflow points, just store original gradient
					grad_render_world_single = curr_pos_world.grad.detach().clone()
					grad_render_world_list.append(grad_render_world_single)
					curr_pos_world.grad = None
			
			# Average the loss
			num_cams = len(matching_cameras) if len(matching_cameras) > 0 else 1
			avg_loss = total_loss_for_frame / num_cams
			avg_l1 = total_l1_for_frame / num_cams
			avg_dssim = total_dssim_for_frame / num_cams
			
			total_epoch_loss += avg_loss
			total_l1_loss += avg_l1
			total_dssim_loss += avg_dssim
			# Initialize frame loss dict (reg_loss and ns_loss will be added later if t > 0)
			global_frame_idx = start_frame + t
			frame_loss_dict = {
				'frame': global_frame_idx,
				'loss': avg_loss,
				'l1': avg_l1,
				'dssim': avg_dssim,
				'reg_loss': 0.0,
				'ns_loss': 0.0
			}
			frame_losses.append(frame_loss_dict)
			
			global_frame_idx = start_frame + t
			writer.add_scalar(f"Loss/Frame_{global_frame_idx}_Render_Loss", avg_loss, epoch)
			writer.add_scalar(f"Loss/Frame_{global_frame_idx}_L1_Loss", avg_l1, epoch)
			writer.add_scalar(f"Loss/Frame_{global_frame_idx}_DSSIM_Loss", avg_dssim, epoch)
			
			# Average the gradients from all cameras
			if len(grad_render_world_list) > 0:
				grad_render_world = torch.stack(grad_render_world_list).mean(dim=0)  # [N, 3]
			else:
				# Fallback: should not happen
				print(f"Warning: No gradients found for frame t={t}")
				curr_pos_world = coord_trans.smoke2world(curr_pos_smoke)
				curr_pos_world.requires_grad_(True)
				grad_render_world = torch.zeros_like(curr_pos_world)
			
			# --- 2. Transform Gradient: World -> Smoke ---
			# dL/dP_smoke = dL/dP_world * dP_world/dP_smoke
			# P_world = (P_smoke * s - T) * R_inv^T
			# Jacobian J = s * R_inv^T (assuming column vectors logic in torch matmul)
			# Actually since we used: pos_world = (pos_smoke * s - T) @ R_s2w.t()
			# Gradient backprop is effectively going through the inverse of that transform.
			# We can use autograd for this small part to be safe, or manual.
			# Manual: grad_smoke = grad_world @ R_s2w * s
			
			with torch.enable_grad():
				dummy_smoke = curr_pos_smoke.detach().requires_grad_(True)
				dummy_world = coord_trans.smoke2world(dummy_smoke)
				dummy_world.backward(grad_render_world)
				grad_render_smoke = dummy_smoke.grad.detach()
			
			# Convert gradient from Smoke Space to Sim Space
			# dL/dP_sim = dL/dP_smoke * dP_smoke/dP_sim
			# P_smoke = P_sim / lengths, so dP_smoke/dP_sim = 1 / lengths
			grad_render_sim = grad_render_smoke / lengths_tensor
			
			# --- 3. Compute Velocity Field Direct Losses (reg_loss + ns_loss) ---
			# Compute for t > 0 (using vel_models[t-1]) or t == 0 with prev_frame_velocity_model (sliding window)
			reg_loss = torch.tensor(0.0, device=device)
			ns_loss = torch.tensor(0.0, device=device)
			
			if t > 0:
				# Get velocity field prediction from vel_models[t-1] (velocity at t-1)
				v_flat = vel_models[t-1](grid_points)  # [N_grid, 3]
				
				# Reshape to grid format: (Nx, Ny, Nz, 3) = (x, y, z, 3)
				pred_velocity = v_flat.view(nx, ny, nz, 3)  # (Nx, Ny, Nz, 3)
				
				# Compute reg_loss: L1 regularization on velocity field
				reg_loss = F.l1_loss(pred_velocity, torch.zeros_like(pred_velocity))
				
				# Compute ns_loss (only for selected frames)
				if t in selected_frame_idx:
					# Get vorticity at t-1: this is the vorticity predicted by vel_models[t-1] at t-1
					vorticity_t_minus_1_flat = vel_models[t - 1].vorticity(grid_points)  # [N_grid, 3]
					
					# Reshape to grid format for advection: (Nx, Ny, Nz, 3) = (x, y, z, 3)
					vorticity_t_minus_1_grid = vorticity_t_minus_1_flat.view(grid_shape + (3,))  # (Nx, Ny, Nz, 3)
					
					# Advect vorticity from t-1 to t using velocity at t-1 (predicted by vel_models[t-1])
					ns_vorticity = advect_vorticity_field_torch(
						vorticity_t_minus_1_grid, pred_velocity, dt, inflow_height_ratio=0.0
					)
					
					# Compute stretching term: (omega · ∇u) * dt
					grad_u = compute_velocity_gradient_torch(pred_velocity)  # [Nx, Ny, Nz, 3, 3]
					omega_dot_grad = torch.einsum('...i,...ij->...j', ns_vorticity, grad_u)
					ns_vorticity = ns_vorticity + omega_dot_grad * dt
					
					# Get predicted vorticity at t (predicted by vel_models[t])
					# This is what the model predicts vorticity should be at frame t
					if t < len(vel_models):
						pred_vorticity_flat = vel_models[t].vorticity(grid_points)  # [N_grid, 3] - vorticity at t
						pred_vorticity = pred_vorticity_flat.reshape(grid_shape + (3,))  # (Nx, Ny, Nz, 3)
						
						# Compute L1 loss between NS-evolved vorticity (from t-1) and predicted vorticity at t
						ns_loss = F.l1_loss(ns_vorticity, pred_vorticity)
					else:
						ns_loss = torch.tensor(0.0, device=device)
				else:
					# For non-selected frames, ns_loss is zero
					ns_loss = torch.tensor(0.0, device=device)
				
				# Backward pass for direct velocity losses
				direct_vel_loss = lambda_regular * reg_loss + lambda_nse * ns_loss
				direct_vel_loss.backward(retain_graph=True)
				
				# Accumulate losses for logging
				total_reg_loss += reg_loss.item()
				total_ns_loss += ns_loss.item()
				
				# Update frame_losses with reg_loss and ns_loss
				if len(frame_losses) > 0:
					frame_losses[-1]['reg_loss'] = reg_loss.item()
					frame_losses[-1]['ns_loss'] = ns_loss.item()
				
				global_frame_idx = start_frame + t
				writer.add_scalar(f"Loss/Frame_{global_frame_idx}_Reg_Loss", reg_loss.item(), epoch)
				writer.add_scalar(f"Loss/Frame_{global_frame_idx}_NS_Loss", ns_loss.item(), epoch)
			elif t == 0 and prev_frame_velocity_model is not None:
				# Sliding window case: compute NS loss from previous window's last frame (start_frame-1) 
				# to current window's first frame (start_frame)
				# The previous frame model is already trained, so we detach it to prevent gradients
				# Only gradients to vel_models[0] (current window's first frame) are computed
				
				# Get velocity field prediction from prev_frame_velocity_model (velocity at start_frame-1)
				# Detach to prevent gradients from flowing back to the previous frame model
				with torch.no_grad():
					v_prev_flat = prev_frame_velocity_model(grid_points)  # [N_grid, 3]
					vorticity_prev_flat = prev_frame_velocity_model.vorticity(grid_points)  # [N_grid, 3]
				
				# Detach to ensure no gradients flow to previous frame model
				v_prev_flat = v_prev_flat.detach()
				vorticity_prev_flat = vorticity_prev_flat.detach()
				
				# Reshape to grid format: (Nx, Ny, Nz, 3) = (x, y, z, 3)
				pred_velocity = v_prev_flat.view(nx, ny, nz, 3)  # (Nx, Ny, Nz, 3)
				
				reg_loss = torch.tensor(0.0, device=device)
				
				# Compute ns_loss: advect vorticity from start_frame-1 to start_frame
				# For sliding window, always compute NS loss at t=0 if prev_frame_velocity_model exists
				# This ensures continuity between windows
				if len(vel_models) > 0:
					# Reshape previous frame's vorticity to grid format: (Nx, Ny, Nz, 3) = (x, y, z, 3)
					vorticity_prev_grid = vorticity_prev_flat.view(grid_shape + (3,))  # (Nx, Ny, Nz, 3)
					
					# Advect vorticity from start_frame-1 to start_frame using velocity at start_frame-1
					# pred_velocity is detached, so gradients won't flow to prev_frame_velocity_model
					ns_vorticity = advect_vorticity_field_torch(
						vorticity_prev_grid, pred_velocity, dt, inflow_height_ratio=0.0
					)
					
					# Compute stretching term: (omega · ∇u) * dt
					# pred_velocity is detached, so gradients won't flow to prev_frame_velocity_model
					grad_u = compute_velocity_gradient_torch(pred_velocity)  # [Nx, Ny, Nz, 3, 3]
					omega_dot_grad = torch.einsum('...i,...ij->...j', ns_vorticity, grad_u)
					ns_vorticity = ns_vorticity + omega_dot_grad * dt
					
					# Get predicted vorticity at start_frame (predicted by vel_models[0])
					# This is what the model predicts vorticity should be at frame start_frame
					pred_vorticity_flat = vel_models[0].vorticity(grid_points)  # [N_grid, 3] - vorticity at start_frame
					pred_vorticity = pred_vorticity_flat.reshape(grid_shape + (3,))  # (Nx, Ny, Nz, 3)
					
					# Compute L1 loss between NS-evolved vorticity (from start_frame-1) and predicted vorticity at start_frame
					# Only vel_models[0] will receive gradients from this loss
					ns_loss = F.l1_loss(ns_vorticity, pred_vorticity)
				else:
					# For non-selected frames or when vel_models is empty, ns_loss is zero
					ns_loss = torch.tensor(0.0, device=device)
				
				# Backward pass for direct velocity losses
				# Only vel_models[0] will receive gradients (prev_frame_velocity_model is detached)
				direct_vel_loss = lambda_regular * reg_loss + lambda_nse * ns_loss
				direct_vel_loss.backward(retain_graph=True)
				
				# Accumulate losses for logging
				total_reg_loss += reg_loss.item()
				total_ns_loss += ns_loss.item()
				
				# Update frame_losses with reg_loss and ns_loss
				if len(frame_losses) > 0:
					frame_losses[-1]['reg_loss'] = reg_loss.item()
					frame_losses[-1]['ns_loss'] = ns_loss.item()
				
				global_frame_idx = start_frame + t
				writer.add_scalar(f"Loss/Frame_{global_frame_idx}_Reg_Loss", reg_loss.item(), epoch)
				writer.add_scalar(f"Loss/Frame_{global_frame_idx}_NS_Loss", ns_loss.item(), epoch)
			
			# --- 4. Accumulate Gradients (Current + Upstream) ---
			# Apply Decay to upstream
			# Note: grad_upstream_sim may have different size than grad_render_sim due to inflow points
			# We need to handle this by padding or truncating grad_upstream_sim to match grad_render_sim
			if grad_upstream_sim.shape[0] != grad_render_sim.shape[0]:
				# If sizes don't match, pad or truncate grad_upstream_sim
				if grad_upstream_sim.shape[0] < grad_render_sim.shape[0]:
					# Pad with zeros (new points don't have upstream gradients)
					padding = torch.zeros(grad_render_sim.shape[0] - grad_upstream_sim.shape[0], 3, device=device)
					grad_upstream_sim = torch.cat([grad_upstream_sim, padding], dim=0)
				else:
					# Truncate (shouldn't happen, but handle it)
					grad_upstream_sim = grad_upstream_sim[:grad_render_sim.shape[0]]
			
			grad_total_sim = grad_render_sim + grad_upstream_sim * grad_decay
			
			# If t > 0, Backpropagate to previous step (Advection)
			if t > 0:
				prev_pos_sim = traj_sim[t-1]
				
				# Advection: x_t = x_{t-1} + v(x_{t-1}) * dt (in Sim Space)
				# Gradients:
				# 1. To v(x_{t-1}): dL/dv = dL/dx_t * dt
				grad_v_part = grad_total_sim * delta_t
				
				# 2. To x_{t-1}: dL/dx_{t-1} = dL/dx_t * (1 + dv/dx * dt)
				# Simplify to dL/dx_{t-1} = dL/dx_t (Identity approximation for stability)
				# Note: prev_pos_sim may have different size due to inflow points added at different times
				# We need to handle this by matching sizes
				if prev_pos_sim.shape[0] != grad_total_sim.shape[0]:
					# If sizes don't match, pad or truncate grad_total_sim
					if prev_pos_sim.shape[0] < grad_total_sim.shape[0]:
						# Truncate grad_total_sim (points added at t don't have gradients from t-1)
						grad_total_sim = grad_total_sim[:prev_pos_sim.shape[0]]
					else:
						# Pad with zeros (shouldn't happen, but handle it)
						padding = torch.zeros(prev_pos_sim.shape[0] - grad_total_sim.shape[0], 3, device=device)
						grad_total_sim = torch.cat([grad_total_sim, padding], dim=0)
				
				grad_upstream_sim = grad_total_sim
				
				# --- 4. Scatter Gradient to Grid (Taichi) ---
				# We need to map gradients on particles (grad_v_part) back to the grid nodes (grad_v_grid)
				# Normalize coords to [0, 1] for Taichi kernel (convert Sim Space to normalized)
				norm_pos_taichi = prev_pos_sim.clone() / lengths_tensor  # Sim [0, lengths] -> [0, 1]
				
				# Call Taichi Kernel
				# returns [1, 3, D, H, W] -> (1, 3, nz, ny, nx)
				grad_v_grid = scatter_grad_to_grid_taichi(grad_v_part, norm_pos_taichi, (nz, ny, nx))
				
				# --- 5. Backprop into Velocity Model (RBF) ---
				# RBF output was reshaped. We need to match shape.
				# RBF output: [N_grid, 3] -> view(nx, ny, nz, 3) -> permute(3, 2, 1, 0)
				# grad_v_grid shape: [1, 3, nz, ny, nx]
				
				# Undo reshape for gradient
				# Permute back: (0, 3, 2, 1) -> (1, 3, x, y, z)? No.
				# Forward: (nx, ny, nz, 3) -> (3, nz, ny, nx). 
				# Backward: (3, nz, ny, nx) -> (nx, ny, nz, 3).
				grad_v_grid_flat = grad_v_grid.squeeze(0).permute(3, 2, 1, 0).reshape(-1, 3)
				
				# Re-run forward graph for this timestep to connect params
				# (Standard PyTorch procedure when using custom backward intermediates)
				v_prediction = vel_models[t-1](grid_points)
				v_prediction.backward(grad_v_grid_flat)
				
			else:
				# t = 0. Gradient goes to Frame 0 Gaussian Parameters
				# grad_total_sim needs to be converted back to World gradient for Gaussian params
				# The Gaussian parameters are stored in World Space.
				# So we need dL/dXYZ_World_0.
				
				# Note: At t=0, only initial points exist (no inflow points added yet)
				# So grad_total_sim should match the initial number of points
				# If sizes don't match, truncate to initial points only
				initial_num_points = gaussians.get_xyz.shape[0]
				if grad_total_sim.shape[0] > initial_num_points:
					# Truncate to initial points only (inflow points added later don't have gradients at t=0)
					grad_total_sim = grad_total_sim[:initial_num_points]
				
				# If start_frame == 0, gradient goes to Frame 0 Gaussian Parameters
				# If start_frame > 0, gradient goes to previous frame's velocity model (not GS xyz)
				if start_frame == 0:
					# Convert grad_total_sim -> grad_total_smoke -> grad_total_world_0
					# First convert Sim Space gradient to Smoke Space gradient
					grad_total_smoke_0 = grad_total_sim * lengths_tensor  # dL/dP_smoke = dL/dP_sim * lengths
					
					with torch.enable_grad():
						# Convert Smoke Space gradient to World Space gradient
						# x_smoke = f(x_world). dL/dx_world = dL/dx_smoke * dx_smoke/dx_world.
						dummy_world_in = gaussians.get_xyz.detach().requires_grad_(True)
						dummy_smoke_out = coord_trans.world2smoke(dummy_world_in)
						dummy_smoke_out.backward(grad_total_smoke_0)
						grad_xyz_0 = dummy_world_in.grad.detach()
					
					# Accumulate into Gaussian Optimizer (only for frame 0)
					if gaussians._xyz.grad is None:
						gaussians._xyz.grad = grad_xyz_0
					else:
						gaussians._xyz.grad += grad_xyz_0
				
				# If start_frame > 0, backpropagate gradient to previous frame's velocity model
				# This allows optimizing the velocity model used to advect initialGS
				# Note: We don't update GS xyz grad here because initialGS position is determined by advection
				if start_frame > 0 and prev_velocity_model_for_advection is not None:
					# The gradient flows through the advection: x_{i+1} = x_i + v(x_i) * dt
					# We need to compute dL/dv where v is the velocity from prev_velocity_model_for_advection
					# Similar to t > 0 case, but using the initial position (frame i)
					
					# Get the initial position in sim space (frame i, detached)
					if initial_xyz_world_i is not None:
						xyz_smoke_i = advection_coord_trans.world2smoke(initial_xyz_world_i)
						xyz_sim_i = xyz_smoke_i * advection_lengths_tensor
						
						# Gradient w.r.t. velocity: dL/dv = dL/dx_{i+1} * dt
						# grad_total_sim is dL/dx_{i+1} in sim space
						grad_v_part = grad_total_sim * advection_dt
						
						# Scatter gradient to grid
						norm_pos_taichi = xyz_sim_i.clone() / advection_lengths_tensor  # Sim [0, lengths] -> [0, 1]
						nx_adv, ny_adv, nz_adv = advection_grid_shape
						grad_v_grid = scatter_grad_to_grid_taichi(grad_v_part, norm_pos_taichi, (nz_adv, ny_adv, nx_adv))
						
						# Convert gradient format
						grad_v_grid_flat = grad_v_grid.squeeze(0).permute(3, 2, 1, 0).reshape(-1, 3)
						
						# Re-run forward to connect to velocity model parameters
						v_prediction_prev = prev_velocity_model_for_advection(advection_grid_points)
						v_prediction_prev.backward(grad_v_grid_flat)
						
						if epoch == 1 or epoch % 10 == 0:
							print(f"  [Epoch {epoch}] Backpropagated gradient to previous frame velocity model (frame {start_frame-1})")
					
				# Note: Other gaussian params (opacity, etc) got gradients directly 
				# during the "Render Loss at Frame t" backward pass because 
				# we only swapped get_xyz, but _opacity etc were accessed normally.
				# Note: Inflow points added at t>0 will get gradients through their own render passes
		
		# === C. Optimizer Step ===
		if use_individual_optimizers and vel_optimizers is not None:
			for vel_frame_idx in range(start_frame, end_frame - 1):
				if vel_frame_idx in vel_optimizers:
					vel_optimizers[vel_frame_idx].step()
					if vel_schedulers is not None and vel_frame_idx in vel_schedulers:
						vel_schedulers[vel_frame_idx].step()
			
			if start_frame > 0 and prev_velocity_model_for_advection is not None:
				prev_frame_idx = start_frame - 1
				if initial_velocity_optimizers_dict is not None and prev_frame_idx in initial_velocity_optimizers_dict:
					prev_optimizer = initial_velocity_optimizers_dict[prev_frame_idx]
					prev_optimizer.step()
					if initial_velocity_schedulers_dict is not None and prev_frame_idx in initial_velocity_schedulers_dict:
						prev_scheduler = initial_velocity_schedulers_dict[prev_frame_idx]
						prev_scheduler.step()
		else:
			vel_optimizer.step()
			vel_scheduler.step()
		gaussians.optimizer.step()
		
		# === C.1. Re-advect InitialGS Position (if applicable) ===
		# After optimizer step, update initialGS position using the updated previous frame velocity model
		if start_frame > 0 and prev_velocity_model_for_advection is not None and initial_xyz_world_i is not None:
			with torch.no_grad():
				# Get the initial position in sim space (frame i, detached)
				xyz_smoke_i = advection_coord_trans.world2smoke(initial_xyz_world_i)
				xyz_sim_i = xyz_smoke_i * advection_lengths_tensor
				
				# Evaluate updated velocity field
				v_flat = prev_velocity_model_for_advection(advection_grid_points)
				nx_adv, ny_adv, nz_adv = advection_grid_shape
				v_vol = v_flat.view(nx_adv, ny_adv, nz_adv, 3).permute(3, 2, 1, 0).unsqueeze(0)
				
				# Sample velocity and advect
				norm_pos = xyz_sim_i.clone()
				norm_pos = 2 * (norm_pos / advection_lengths_tensor) - 1.0
				grid_in = norm_pos.view(1, 1, 1, -1, 3)
				v_part = F.grid_sample(v_vol, grid_in, align_corners=True, mode='bilinear')
				v_part = v_part.view(3, -1).permute(1, 0)
				
				xyz_sim_i_plus_1 = xyz_sim_i + v_part * advection_dt
				
				# Convert back to world space
				xyz_smoke_i_plus_1 = xyz_sim_i_plus_1 / advection_lengths_tensor
				xyz_world_i_plus_1 = advection_coord_trans.smoke2world(xyz_smoke_i_plus_1)
				
				# Update GS position (detach to break the computation graph, since we use analytic gradients)
				gaussians._xyz.data = xyz_world_i_plus_1.detach().requires_grad_(True)
				
				if epoch == 1 or epoch % 10 == 0:
					print(f"  [Epoch {epoch}] Re-advected initialGS position using updated velocity model (frame {start_frame-1})")
		
		epoch_train_end = datetime.now()
		epoch_train_duration = (epoch_train_end - epoch_train_start).total_seconds()
		total_train_time += epoch_train_duration
		
		# === D. Visualization & Logging ===
		# Calculate average loss per frame
		avg_loss_per_frame = total_epoch_loss / advect_frame_num if advect_frame_num > 0 else 0.0
		avg_l1_per_frame = total_l1_loss / advect_frame_num if advect_frame_num > 0 else 0.0
		avg_dssim_per_frame = total_dssim_loss / advect_frame_num if advect_frame_num > 0 else 0.0
		avg_reg_loss_per_frame = total_reg_loss / max(1, advect_frame_num - 1) if advect_frame_num > 1 else 0.0
		avg_ns_loss_per_frame = total_ns_loss / max(1, len(selected_frame_idx)) if len(selected_frame_idx) > 0 else 0.0
		
		# Record to Tensorboard (every epoch)
		writer.add_scalar("Train/Total_Loss", total_epoch_loss, epoch)
		writer.add_scalar("Train/Avg_Loss_Per_Frame", avg_loss_per_frame, epoch)
		writer.add_scalar("Train/Total_L1_Loss", total_l1_loss, epoch)
		writer.add_scalar("Train/Avg_L1_Loss_Per_Frame", avg_l1_per_frame, epoch)
		writer.add_scalar("Train/Total_DSSIM_Loss", total_dssim_loss, epoch)
		writer.add_scalar("Train/Avg_DSSIM_Loss_Per_Frame", avg_dssim_per_frame, epoch)
		writer.add_scalar("Train/Total_Reg_Loss", total_reg_loss, epoch)
		writer.add_scalar("Train/Avg_Reg_Loss_Per_Frame", avg_reg_loss_per_frame, epoch)
		writer.add_scalar("Train/Total_NS_Loss", total_ns_loss, epoch)
		writer.add_scalar("Train/Avg_NS_Loss_Per_Frame", avg_ns_loss_per_frame, epoch)
		
		# Print to terminal (every epoch)
		print(f"Epoch {epoch}/{num_epochs}: Total Loss: {total_epoch_loss:.6f}, Avg Loss/Frame: {avg_loss_per_frame:.6f}")
		print(f"  L1 Loss: {total_l1_loss:.6f} (avg: {avg_l1_per_frame:.6f}), DSSIM Loss: {total_dssim_loss:.6f} (avg: {avg_dssim_per_frame:.6f})")
		print(f"  Reg Loss: {total_reg_loss:.6f} (avg: {avg_reg_loss_per_frame:.6f}), NS Loss: {total_ns_loss:.6f} (avg: {avg_ns_loss_per_frame:.6f})")
		if len(frame_losses) > 0:
			print(f"  Frame losses: ", end="")
			for i, frame_loss_info in enumerate(frame_losses):
				if i < 5:  # Only print first 5 frames to avoid clutter
					print(f"Frame {frame_loss_info['frame']}: {frame_loss_info['loss']:.6f}", end=", ")
			if len(frame_losses) > 5:
				print(f"... ({len(frame_losses)} frames total)")
			else:
				print()
		
		if epoch % args.i_save == 0:
			checkpoint_start = datetime.now()
			
			# Save Velocity Model
			for i, model in enumerate(vel_models):
				global_frame_idx = start_frame + i
				model.save(f"{savedir}/ckpt/velrbf_frame_{global_frame_idx:03d}_ckpt_{epoch:06d}.pth")
				
				if use_individual_optimizers and vel_optimizers is not None and global_frame_idx in vel_optimizers:
					optimizer_path = os.path.join(savedir, "ckpt", f"velrbf_frame_{global_frame_idx:03d}_optimizer_{epoch:06d}.pth")
					torch.save(vel_optimizers[global_frame_idx].state_dict(), optimizer_path)
					
				if vel_schedulers is not None and global_frame_idx in vel_schedulers:
					scheduler_path = os.path.join(savedir, "ckpt", f"velrbf_frame_{global_frame_idx:03d}_scheduler_{epoch:06d}.pth")
					torch.save(vel_schedulers[global_frame_idx].state_dict(), scheduler_path)
			
			# Save previous frame velocity model if it was optimized
			if start_frame > 0 and prev_velocity_model_for_advection is not None:
				prev_frame_idx = start_frame - 1
				prev_velocity_model_for_advection.save(
					f"{savedir}/ckpt/velrbf_frame_{prev_frame_idx:03d}_ckpt_{epoch:06d}.pth"
				)
				
				if use_individual_optimizers and initial_velocity_optimizers_dict is not None:
					if prev_frame_idx in initial_velocity_optimizers_dict:
						optimizer_path = os.path.join(savedir, "ckpt", f"velrbf_frame_{prev_frame_idx:03d}_optimizer_{epoch:06d}.pth")
						torch.save(initial_velocity_optimizers_dict[prev_frame_idx].state_dict(), optimizer_path)
						
						if initial_velocity_schedulers_dict is not None and prev_frame_idx in initial_velocity_schedulers_dict:
							scheduler_path = os.path.join(savedir, "ckpt", f"velrbf_frame_{prev_frame_idx:03d}_scheduler_{epoch:06d}.pth")
							torch.save(initial_velocity_schedulers_dict[prev_frame_idx].state_dict(), scheduler_path)
			
			# Save Gaussian Model (using train_frame_zero_gaussian style)
			# Save to point_cloud/coarse_iteration_{epoch}/ directory with 4 files:
			# - point_cloud.ply
			# - deformation.pth
			# - deformation_table.pth
			# - deformation_accum.pth
			print(f"\n[Epoch {epoch}] Saving Gaussians")
			point_cloud_path = os.path.join(savedir, "point_cloud", f"coarse_iteration_{epoch}")
			os.makedirs(point_cloud_path, exist_ok=True)
			gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
			gaussians.save_deformation(point_cloud_path)
			# Also save checkpoint
			checkpoint_path = os.path.join(savedir, f"chkpnt_coarse_{epoch}.pth")
			checkpoint_data = (gaussians.capture(), epoch)
			
			# Save inflow Gaussians if they exist
			if inflow_ratio > 0 and inflow_gaussians is not None:
				print(f"[Epoch {epoch}] Saving Inflow Gaussians ({inflow_gaussians.num_groups} groups)")
				inflow_checkpoint_path = os.path.join(savedir, f"inflow_gaussians_epoch_{epoch}.pth")
				inflow_checkpoint_data = (inflow_gaussians.capture(), epoch)
				torch.save(inflow_checkpoint_data, inflow_checkpoint_path)
				print(f"  Saved inflow Gaussians checkpoint to {inflow_checkpoint_path}")
				# Also save to the main checkpoint for convenience
				checkpoint_data = (gaussians.capture(), epoch, inflow_gaussians.capture())
			
			torch.save(checkpoint_data, checkpoint_path)
			
			checkpoint_end = datetime.now()
			checkpoint_duration = (checkpoint_end - checkpoint_start).total_seconds()
			total_checkpoint_time += checkpoint_duration
			print(f"[Epoch {epoch}] Checkpoint saving time: {int(checkpoint_duration // 60)}m {checkpoint_duration % 60:.2f}s")
			
		if epoch % args.i_draw == 0:
			vis_start_time = datetime.now()
			
			# Visualize Velocity Slice (Middle Z)
			vel_imgs = []
			
			os.makedirs(os.path.join(savedir, "images", "vel"), exist_ok=True)
			os.makedirs(os.path.join(savedir, "video", "vel"), exist_ok=True)
			
			# Only visualize velocity fields for frames that were trained
			for frame in range(advect_frame_num - 1):
				# vis_points_np = utils_grid.generate_gridpoints(grid_shape, min_corner, max_corner)
				# vis_points = torch.from_numpy(vis_points_np).float().to(device)
	
				vel_pred_full = vel_models[frame](grid_points)
				
				im_estim = vel_pred_full.detach().cpu().numpy()
				im_estim = np.reshape(im_estim, grid_shape + (3,))  # (x, y, z, 3)
				im_estim = np.swapaxes(im_estim, 0, 2)  # (z, y, x, 3)
				
				estim_image = vel_uv2hsv(im_estim, scale=args.vel_color, is3D=True, logv=False)
				
				if frame == (advect_frame_num - 1) // 2:
					imageio.imwrite(f"{savedir}/images/vel/vel_image_{epoch:06d}.png", estim_image)
				vel_imgs.append(estim_image)
			
			vel_video = np.stack(vel_imgs, axis=0)
			imageio.mimwrite(
				f"{savedir}/video/vel/vel_video_{epoch:06d}.mp4",
				vel_video,
				fps=25,
				quality=8,
			)
			print(f"velocity fieldVideo saved: {savedir}/video/vel/vel_video_{epoch:06d}.mp4")
			
			# Visualize Vorticity Evolution (from frame 0, advected through all frames)
			print(f"\n[Vorticity Evolution Visualization] Computing evolved vorticity for {advect_frame_num} frames...")
			vorticity_evolved_imgs = []
			
			# Use no_grad for visualization (no gradients needed)
			with torch.no_grad():
				# Frame 0: Get initial vorticity from vel_models[0]
				initial_vorticity_flat = vel_models[0].vorticity(grid_points)  # [N_grid, 3]
				# Reshape to grid format: (Nx, Ny, Nz, 3) = (x, y, z, 3)
				current_vorticity_grid = initial_vorticity_flat.view(grid_shape + (3,))  # (Nx, Ny, Nz, 3)
				
				# Visualize frame 0 vorticity
				# Convert to visualization format: (x, y, z, 3) -> (z, y, x, 3) for vel_uv2hsv
				vorticity_0_np = current_vorticity_grid.detach().cpu().numpy()
				vorticity_0_vis = np.swapaxes(vorticity_0_np, 0, 2)  # (z, y, x, 3)
				vorticity_0_image = vel_uv2hsv(vorticity_0_vis, scale=getattr(args, 'vor_color', args.vel_color), is3D=True, logv=False)
				vorticity_evolved_imgs.append(vorticity_0_image)
				
				# Evolve vorticity frame by frame (from frame 1 to advect_frame_num - 1)
				for t in range(1, advect_frame_num):
					# Get velocity field at t-1
					vel_pred_full = vel_models[t - 1](grid_points)  # [N_grid, 3]
					# Reshape to grid format: (Nx, Ny, Nz, 3) = (x, y, z, 3)
					pred_velocity = vel_pred_full.view(grid_shape + (3,))  # (Nx, Ny, Nz, 3)
					
					# Advect vorticity from t-1 to t
					ns_vorticity = advect_vorticity_field_torch(
						current_vorticity_grid, pred_velocity, dt, inflow_height_ratio=0.0
					)
					
					# Compute stretching term: (omega · ∇u) * dt
					grad_u = compute_velocity_gradient_torch(pred_velocity)  # [Nx, Ny, Nz, 3, 3]
					omega_dot_grad = torch.einsum('...i,...ij->...j', ns_vorticity, grad_u)
					ns_vorticity = ns_vorticity + omega_dot_grad * dt
					
					# Update current vorticity for next iteration
					current_vorticity_grid = ns_vorticity
					
					# Visualize evolved vorticity at frame t
					# Convert to visualization format: (x, y, z, 3) -> (z, y, x, 3) for vel_uv2hsv
					vorticity_t_np = ns_vorticity.detach().cpu().numpy()
					vorticity_t_vis = np.swapaxes(vorticity_t_np, 0, 2)  # (z, y, x, 3)
					vorticity_t_image = vel_uv2hsv(vorticity_t_vis, scale=getattr(args, 'vor_color', args.vel_color), is3D=True, logv=False)
					vorticity_evolved_imgs.append(vorticity_t_image)
			
			# Save vorticity evolution video
			os.makedirs(os.path.join(savedir, "video", "vorticity_evolved"), exist_ok=True)
			vorticity_evolved_video = np.stack(vorticity_evolved_imgs, axis=0)
			imageio.mimwrite(
				f"{savedir}/video/vorticity_evolved/vorticity_evolved_video_{epoch:06d}.mp4",
				vorticity_evolved_video,
				fps=25,
				quality=8,
			)
			print(f"vorticityVideo saved: {savedir}/video/vorticity_evolved/vorticity_evolved_video_{epoch:06d}.mp4")
				
			# Visualize Gaussian Render for all frames
			# Render each frame with its corresponding camera(s) and save comparison images
			with torch.no_grad():
				vis_images_with_inflow = []  # Store images for video: Render (with inflow) | GT
				vis_images_no_inflow = []  # Store images for video: Render (no inflow) | GT
				vis_images_only_inflow = []  # Store images for video: Render (only inflow) | GT
				
				# Create epoch-specific directory for images
				epoch_dir = os.path.join(savedir, "vis_train", f"epoch_{epoch:06d}")
				os.makedirs(epoch_dir, exist_ok=True)
				
				# Create directory for VTK Gaussian visualization
				vtk_gaussian_dir = os.path.join(savedir, "vis_train", f"gaussians_vtk_epoch_{epoch:06d}")
				os.makedirs(vtk_gaussian_dir, exist_ok=True)
				print(f"\n[Saving Gaussians to VTK] Saving to {vtk_gaussian_dir}")
				
				# Only visualize frames that were advected in this epoch
				print(f"\n[Rendering Visualization] Rendering {advect_frame_num} frames for epoch {epoch} (progressive training)...")
				
				# Create directory for opacity visualizations
				opacity_vis_dir = os.path.join(savedir, "vis_train", f"opacity_epoch_{epoch:06d}")
				os.makedirs(opacity_vis_dir, exist_ok=True)
				
				# Create directory for point cloud visualizations
				pointcloud_vis_dir = os.path.join(savedir, "vis_train", f"pointcloud_epoch_{epoch:06d}")
				os.makedirs(pointcloud_vis_dir, exist_ok=True)
				
				# Get original position (frame 0, before advection)
				# Convert from world space to sim space
				xyz_world_0 = gaussians.get_xyz.detach().clone()
				xyz_smoke_0 = coord_trans.world2smoke(xyz_world_0)
				original_pos_sim = xyz_smoke_0 * lengths_tensor  # [N, 3] in sim space
				
				for t in range(advect_frame_num):
					# Get position for this frame
					if t >= len(traj_sim):
						print(f"Warning: Frame {t} not found in traj_sim (length={len(traj_sim)}), skipping...")
						continue
						
					# Convert Sim Space -> Smoke Space -> World Space for rendering
					pos_sim = traj_sim[t]
					pos_smoke = pos_sim / lengths_tensor  # Sim [0, lengths] -> Smoke [0, 1]
					pos_world = coord_trans.smoke2world(pos_smoke)
					
					# === Determine inflow groups for this frame ===
					# Determine which inflow groups should be included at frame t
					has_inflow_frame = (t > 0 and inflow_ratio > 0 and inflow_gaussians is not None)
					if has_inflow_frame:
						max_group_idx = min(t, inflow_gaussians.num_groups)
						inflow_group_indices_frame = list(range(max_group_idx))
					else:
						inflow_group_indices_frame = []
					
					# Split pos_world into original and inflow parts
					orig_num_points = gaussians.get_xyz.shape[0]
					if has_inflow_frame and pos_world.shape[0] > orig_num_points:
						# pos_world contains both original and inflow points
						pos_world_orig = pos_world[:orig_num_points]
						pos_world_inflow = pos_world[orig_num_points:]
					else:
						# Only original points
						pos_world_orig = pos_world
						pos_world_inflow = None
					
					if visualize_opacity:
						# =================== Opacity visualization Start ===================
						# Get Gaussian opacity values
						opacity_values = gaussians.opacity_activation(gaussians._opacity).detach().cpu().numpy()  # [N, 1]
						opacity_values = opacity_values.squeeze(-1) if opacity_values.ndim > 1 else opacity_values  # [N]
						
						# Use pos_sim directly (already in sim space [0, lengths])
						pos_sim_np = pos_sim.detach().cpu().numpy()  # [N, 3] in sim space [0, lengths]
						
						# Map opacity values to 3D grid
						# Create a 3D grid for opacity distribution
						opacity_grid = np.zeros(grid_shape, dtype=np.float32)  # [nx, ny, nz]
						
						# Calculate grid cell size
						cell_size = lengths_tensor.cpu().numpy() / np.array(grid_shape)  # [3]
						# Map each Gaussian's opacity to the nearest grid cell
						# For simplicity, we use nearest neighbor assignment
						for i in range(len(pos_sim_np)):
							pos = pos_sim_np[i]
							opacity = opacity_values[i]
							
							# Clamp grid indices to valid range and convert to integer
							grid_idx = np.clip(pos / s, [0, 0, 0], np.array(grid_shape) - 1).astype(np.int32)
							
							# Accumulate opacity (use max or sum - using sum for now)
							opacity_grid[grid_idx[0], grid_idx[1], grid_idx[2]] += opacity
						
						# Normalize opacity grid (optional, to prevent overflow)
						# Or use max to get peak opacity
						# opacity_grid = np.clip(opacity_grid, 0, 1)
						
						# Save VTK format for ParaView
						# Calculate origin and spacing for VTK
						origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Origin in sim space
						spacing = cell_size  # Spacing between grid points
						
						# Save VTK Image Data format
						vtk_path = os.path.join(opacity_vis_dir, f"frame_{t:03d}_opacity.vti")
						save_vtk_image_data(opacity_grid, origin, spacing, vtk_path)
						
						if t == 0 or (t + 1) % 10 == 0:
							print(f"  Saved VTK opacity grid for frame {t}: {vtk_path}")
						
						# Reshape to format expected by den_scalar2rgb: (d, h, w, 1)
						# den_scalar2rgb expects shape (d, h, w, 1) for 3D
						opacity_grid_4d = opacity_grid[..., np.newaxis]  # [nx, ny, nz, 1]
						
						# Convert to format expected by den_scalar2rgb
						# den_scalar2rgb expects (d, h, w, 1) where d is depth (z), h is height (y), w is width (x)
						# Our grid_shape is (nx, ny, nz) = (width, height, depth)
						# So we need to transpose: (nx, ny, nz) -> (nz, ny, nx) = (d, h, w)
						opacity_grid_for_vis = np.transpose(opacity_grid_4d, (2, 1, 0, 3))  # [nz, ny, nx, 1]
						
						# Visualize using den_scalar2rgb
						opacity_image = den_scalar2rgb(opacity_grid_for_vis, scale=5000, is3D=True, logv=False, mix=True)
						
						# Save opacity visualization
						opacity_path = os.path.join(opacity_vis_dir, f"frame_{t:03d}_opacity.png")
						imageio.imwrite(opacity_path, opacity_image)
						
						if t == 0 or (t + 1) % 10 == 0:
							print(f"  Saved opacity visualization for frame {t}: {opacity_path}")
						
						# === Visualize Point Cloud Three Views ===
						# Visualize current frame positions colored by opacity
						pointcloud_path = os.path.join(pointcloud_vis_dir, f"frame_{t:03d}_pointcloud.png")
						visualize_pointcloud_with_opacity(
							positions=pos_sim,
							opacity_values=opacity_values,
							save_path=pointcloud_path,
							title=f"Gaussian Point Cloud - Frame {t} (Epoch {epoch})"
						)
						
						if t == 0 or (t + 1) % 10 == 0:
							print(f"  Saved point cloud visualization for frame {t}: {pointcloud_path}")
	  
						# =================== Opacity visualization Finish ===================
					
					# Get pre-computed matching cameras for this frame
					# Note: In progressive training, advect_frame_num may exceed frame_range
					relative_frame_idx = t
					if relative_frame_idx < len(frame_to_cameras):
						matching_cameras = frame_to_cameras[relative_frame_idx]
					else:
						# Frame beyond frame_range (progressive training)
						# Map to the last frame's cameras or use empty list
						if len(frame_to_cameras) > 0:
							matching_cameras = frame_to_cameras[len(frame_to_cameras) - 1]
						else:
							matching_cameras = []
					
					# Render all matching cameras for this frame
					frame_comparisons_with_inflow = []  # Store comparisons with inflow: Render (with inflow) | GT
					frame_comparisons_no_inflow = []  # Store comparisons without inflow: Render (no inflow) | GT
					frame_comparisons_only_inflow = []  # Store comparisons with only inflow: Render (only inflow) | GT
					frame_renders = []  # Store individual render images
					frame_gts = []  # Store individual GT images
					
					for cam_idx, cam in matching_cameras:
						# Render with both original and inflow Gaussians (if any)
						if has_inflow_frame and pos_world_inflow is not None:
							# Use ExtendedGaussianWrapper to merge original and inflow GS
							# pos_world already contains both original and inflow points in correct order
							wrapped_vis = ExtendedGaussianWrapper(gaussians, pos_world, inflow_gaussians, inflow_group_indices_frame)
							render_result = render(cam, wrapped_vis, pipe, background, stage="coarse")["render"]
							
							# Also render without inflow for comparison
							wrapped_vis_no_inflow = GaussianOverrideWrapper(gaussians, pos_world_orig)
							render_result_no_inflow = render(cam, wrapped_vis_no_inflow, pipe, background, stage="coarse")["render"]
							
							# For multiple groups, we need to merge them
							# Create a simple wrapper that merges all inflow groups
							class InflowOnlyExtendedWrapper:
								"""Wrapper that only contains inflow points from multiple groups"""
								def __init__(self, inflow_gaussians, inflow_group_indices, merged_pos_world):
									self.inflow = inflow_gaussians
									self.inflow_group_indices = inflow_group_indices
									self.merged_pos = merged_pos_world
									
									# Merge all inflow groups' properties
									inflow_opacities = []
									inflow_scalings = []
									inflow_rotations = []
									inflow_features_dc_list = []
									inflow_features_rest_list = []
									
									for group_idx in inflow_group_indices:
										inflow_opacities.append(inflow_gaussians._opacity_groups[group_idx])
										inflow_scalings.append(inflow_gaussians._scaling_groups[group_idx])
										inflow_rotations.append(inflow_gaussians.get_group_rotation(group_idx))
										inflow_features_dc_list.append(inflow_gaussians.get_group_features_dc(group_idx))
										inflow_features_rest_list.append(inflow_gaussians.get_group_features_rest(group_idx))
									
									self._opacity = torch.cat(inflow_opacities, dim=0) if inflow_opacities else torch.empty(0, 1, device=merged_pos_world.device)
									self._scaling = torch.cat(inflow_scalings, dim=0) if inflow_scalings else torch.empty(0, 3, device=merged_pos_world.device)
									self._rotation = torch.cat(inflow_rotations, dim=0) if inflow_rotations else torch.empty(0, 4, device=merged_pos_world.device)
									self._features_dc = torch.cat(inflow_features_dc_list, dim=0) if inflow_features_dc_list else torch.empty(0, 1, 3, device=merged_pos_world.device)
									self._features_rest = torch.cat(inflow_features_rest_list, dim=0) if inflow_features_rest_list else torch.empty(0, (inflow_gaussians.max_sh_degree + 1) ** 2 - 1, 3, device=merged_pos_world.device)
									
									num_points = merged_pos_world.shape[0]
									self._deformation_table = torch.zeros(num_points, dtype=torch.bool, device=merged_pos_world.device)
								
								@property
								def get_xyz(self):
									return self.merged_pos
								
								@property
								def get_opacity(self):
									return torch.sigmoid(self._opacity)
								
								@property
								def get_scaling(self):
									return torch.exp(self._scaling)
								
								@property
								def get_rotation(self):
									q = self._rotation
									return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
								
								@property
								def get_features_dc(self):
									return self._features_dc
								
								@property
								def get_features_rest(self):
									return self._features_rest
								
								@property
								def get_features(self):
									"""returnfull features，render attributes"""
									return torch.cat([self._features_dc, self._features_rest], dim=1)
								
								@property
								def active_sh_degree(self):
									return self.inflow.max_sh_degree
								
								@property
								def max_sh_degree(self):
									return self.inflow.max_sh_degree
								
								@property
								def opacity_activation(self):
									return torch.sigmoid
								
								@property
								def scaling_activation(self):
									return torch.exp
								
								@property
								def rotation_activation(self):
									def normalize_quaternion(q):
										return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
									return normalize_quaternion
								
								def get_covariance(self, scaling_modifier=1.0):
									"""（）"""
									return None
								
								def __getattr__(self, name):
									if hasattr(self.inflow, name):
										return getattr(self.inflow, name)
									raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
							
							wrapped_vis_only_inflow = InflowOnlyExtendedWrapper(inflow_gaussians, inflow_group_indices_frame, pos_world_inflow)
							render_result_only_inflow = render(cam, wrapped_vis_only_inflow, pipe, background, stage="coarse")["render"]
						else:
							# Only original GS
							wrapped_vis = GaussianOverrideWrapper(gaussians, pos_world_orig)
							render_result = render(cam, wrapped_vis, pipe, background, stage="coarse")["render"]
							render_result_no_inflow = None  # No inflow to compare
							render_result_only_inflow = None  # No inflow to compare
						
						# Get GT image
						gt_image = cam.original_image.cuda()
						
						# Convert to numpy for saving
						render_np = render_result.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
						gt_np = gt_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
						
						# Ensure images have the same height
						if render_np.shape[:2] != gt_np.shape[:2]:
							target_h, target_w = render_np.shape[:2]
							gt_np = cv.resize(gt_np, (target_w, target_h), interpolation=cv.INTER_LINEAR)
						
						# Create comparison image with inflow: Render (with inflow) | GT
						comparison_with_inflow = np.concatenate([render_np, gt_np], axis=1)  # [H, 2*W, 3]
						comparison_with_inflow_uint8 = (comparison_with_inflow * 255).astype(np.uint8)
						frame_comparisons_with_inflow.append(comparison_with_inflow_uint8)
						
						# Create comparison image without inflow
						if render_result_no_inflow is not None:
							render_no_inflow_np = render_result_no_inflow.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
							# Ensure images have the same height
							if render_no_inflow_np.shape[:2] != gt_np.shape[:2]:
								target_h, target_w = render_np.shape[:2]
								render_no_inflow_np = cv.resize(render_no_inflow_np, (target_w, target_h), interpolation=cv.INTER_LINEAR)
							comparison_no_inflow = np.concatenate([render_no_inflow_np, gt_np], axis=1)  # [H, 2*W, 3]
						else:
							# No inflow, so render_no_inflow is the same as render
							comparison_no_inflow = comparison_with_inflow.copy()
						
						comparison_no_inflow_uint8 = (comparison_no_inflow * 255).astype(np.uint8)
						frame_comparisons_no_inflow.append(comparison_no_inflow_uint8)
						
						# Create comparison image with only inflow: Render (only inflow) | GT
						if render_result_only_inflow is not None:
							render_only_inflow_np = render_result_only_inflow.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
							# Ensure images have the same height
							if render_only_inflow_np.shape[:2] != gt_np.shape[:2]:
								target_h, target_w = render_np.shape[:2]
								render_only_inflow_np = cv.resize(render_only_inflow_np, (target_w, target_h), interpolation=cv.INTER_LINEAR)
							comparison_only_inflow = np.concatenate([render_only_inflow_np, gt_np], axis=1)  # [H, 2*W, 3]
						else:
							# No inflow, so render_only_inflow is the same as render (or empty)
							if has_inflow_frame:
								# Create a black image for comparison
								comparison_only_inflow = np.zeros_like(comparison_with_inflow)
							else:
								comparison_only_inflow = comparison_with_inflow.copy()
						
						comparison_only_inflow_uint8 = (comparison_only_inflow * 255).astype(np.uint8)
						frame_comparisons_only_inflow.append(comparison_only_inflow_uint8)
						
						frame_renders.append((render_np * 255).astype(np.uint8))
						frame_gts.append((gt_np * 255).astype(np.uint8))
						
						# Save comparison images for this specific camera (in epoch folder)
						comparison_with_inflow_path = os.path.join(epoch_dir, f"frame_{t:03d}_cam_{cam_idx:03d}_comparison_with_inflow.png")
						comparison_no_inflow_path = os.path.join(epoch_dir, f"frame_{t:03d}_cam_{cam_idx:03d}_comparison_no_inflow.png")
						comparison_only_inflow_path = os.path.join(epoch_dir, f"frame_{t:03d}_cam_{cam_idx:03d}_comparison_only_inflow.png")
						imageio.imwrite(comparison_with_inflow_path, comparison_with_inflow_uint8)
						imageio.imwrite(comparison_no_inflow_path, comparison_no_inflow_uint8)
						imageio.imwrite(comparison_only_inflow_path, comparison_only_inflow_uint8)
						
						# Also save individual images for this camera (in epoch folder)
						render_path = os.path.join(epoch_dir, f"frame_{t:03d}_cam_{cam_idx:03d}_render.png")
						gt_path = os.path.join(epoch_dir, f"frame_{t:03d}_cam_{cam_idx:03d}_gt.png")
						imageio.imwrite(render_path, frame_renders[-1])
						imageio.imwrite(gt_path, frame_gts[-1])
						
						# If we have inflow, also save render without inflow and only inflow for comparison
						if render_result_no_inflow is not None:
							render_no_inflow_path = os.path.join(epoch_dir, f"frame_{t:03d}_cam_{cam_idx:03d}_render_no_inflow.png")
							# render_no_inflow_np was already computed above
							imageio.imwrite(render_no_inflow_path, (render_no_inflow_np * 255).astype(np.uint8))
						
						if render_result_only_inflow is not None:
							render_only_inflow_path = os.path.join(epoch_dir, f"frame_{t:03d}_cam_{cam_idx:03d}_render_only_inflow.png")
							# render_only_inflow_np was already computed above
							imageio.imwrite(render_only_inflow_path, (render_only_inflow_np * 255).astype(np.uint8))
					
					# Check if we have any cameras for this frame
					if len(frame_comparisons_with_inflow) == 0:
						# No cameras matched for this frame, skip it
						print(f"  Warning: No cameras found for frame {t}, skipping visualization")
						continue
					
					# If multiple cameras, create a grid layout showing all views
					if len(frame_comparisons_with_inflow) > 1:
						# Create a grid: arrange cameras in rows
						# For simplicity, arrange horizontally (all cameras side by side)
						num_cams = len(frame_comparisons_with_inflow)
						
						# Each comparison has 2 columns: Render | GT
						grid_comparison_with_inflow = np.concatenate(frame_comparisons_with_inflow, axis=1)  # [H, num_cams*2*W, 3]
						grid_comparison_no_inflow = np.concatenate(frame_comparisons_no_inflow, axis=1)  # [H, num_cams*2*W, 3]
						grid_comparison_only_inflow = np.concatenate(frame_comparisons_only_inflow, axis=1)  # [H, num_cams*2*W, 3]
						
						# Save grid comparisons (in epoch folder)
						grid_with_inflow_path = os.path.join(epoch_dir, f"frame_{t:03d}_all_cams_comparison_with_inflow.png")
						grid_no_inflow_path = os.path.join(epoch_dir, f"frame_{t:03d}_all_cams_comparison_no_inflow.png")
						grid_only_inflow_path = os.path.join(epoch_dir, f"frame_{t:03d}_all_cams_comparison_only_inflow.png")
						imageio.imwrite(grid_with_inflow_path, grid_comparison_with_inflow)
						imageio.imwrite(grid_no_inflow_path, grid_comparison_no_inflow)
						imageio.imwrite(grid_only_inflow_path, grid_comparison_only_inflow)
						
						# For video, use the first camera's comparison (or could use grid if not too wide)
						# Use grid if width is reasonable (e.g., < 4000 pixels), otherwise use first camera
						max_width = 4000
						if grid_comparison_with_inflow.shape[1] <= max_width:
							vis_images_with_inflow.append(grid_comparison_with_inflow)
							vis_images_no_inflow.append(grid_comparison_no_inflow)
							vis_images_only_inflow.append(grid_comparison_only_inflow)
						else:
							vis_images_with_inflow.append(frame_comparisons_with_inflow[0])
							vis_images_no_inflow.append(frame_comparisons_no_inflow[0])
							vis_images_only_inflow.append(frame_comparisons_only_inflow[0])
					else:
						# Single camera: use its comparison
						vis_images_with_inflow.append(frame_comparisons_with_inflow[0])
						vis_images_no_inflow.append(frame_comparisons_no_inflow[0])
						vis_images_only_inflow.append(frame_comparisons_only_inflow[0])
						
						# Also save with simpler naming (backward compatibility, in epoch folder)
						comparison_with_inflow_path = os.path.join(epoch_dir, f"frame_{t:03d}_comparison_with_inflow.png")
						comparison_no_inflow_path = os.path.join(epoch_dir, f"frame_{t:03d}_comparison_no_inflow.png")
						comparison_only_inflow_path = os.path.join(epoch_dir, f"frame_{t:03d}_comparison_only_inflow.png")
						imageio.imwrite(comparison_with_inflow_path, frame_comparisons_with_inflow[0])
						imageio.imwrite(comparison_no_inflow_path, frame_comparisons_no_inflow[0])
						imageio.imwrite(comparison_only_inflow_path, frame_comparisons_only_inflow[0])
						
						# Save individual images using already computed values (in epoch folder)
						render_path = os.path.join(epoch_dir, f"frame_{t:03d}_render.png")
						gt_path = os.path.join(epoch_dir, f"frame_{t:03d}_gt.png")
						imageio.imwrite(render_path, frame_renders[0])
						imageio.imwrite(gt_path, frame_gts[0])
					
					if (t + 1) % 10 == 0 or t == advect_frame_num - 1:
						print(f"  Rendered {t+1}/{advect_frame_num} frames (global: {start_frame+t+1}/{end_frame})...")
				
				# Create videos from all frames
				# Video 1: Render (with inflow) | GT
				if len(vis_images_with_inflow) > 0:
					video_path_with_inflow = os.path.join(savedir, "vis_train", f"comparison_with_inflow_epoch_{epoch:06d}.mp4")
					imageio.mimwrite(video_path_with_inflow, vis_images_with_inflow, fps=10, quality=8)
					print(f"Saved comparison video (with inflow): {video_path_with_inflow} with {len(vis_images_with_inflow)} frames")
				else:
					print(f"Warning: No images to create video (with inflow) for epoch {epoch}")
				
				# Video 2: Render (no inflow) | GT
				if len(vis_images_no_inflow) > 0:
					video_path_no_inflow = os.path.join(savedir, "vis_train", f"comparison_no_inflow_epoch_{epoch:06d}.mp4")
					imageio.mimwrite(video_path_no_inflow, vis_images_no_inflow, fps=10, quality=8)
					print(f"Saved comparison video (no inflow): {video_path_no_inflow} with {len(vis_images_no_inflow)} frames")
				else:
					print(f"Warning: No images to create video (no inflow) for epoch {epoch}")
				
				# Video 3: Render (only inflow) | GT
				if len(vis_images_only_inflow) > 0:
					video_path_only_inflow = os.path.join(savedir, "vis_train", f"comparison_only_inflow_epoch_{epoch:06d}.mp4")
					imageio.mimwrite(video_path_only_inflow, vis_images_only_inflow, fps=10, quality=8)
					print(f"Saved comparison video (only inflow): {video_path_only_inflow} with {len(vis_images_only_inflow)} frames")
				else:
					print(f"Warning: No images to create video (only inflow) for epoch {epoch}")
				
				vis_end_time = datetime.now()
				vis_duration = (vis_end_time - vis_start_time).total_seconds()
				total_visualization_time += vis_duration
				print(f"[Epoch {epoch}] Visualization time: {int(vis_duration // 60)}m {vis_duration % 60:.2f}s")
	
	if num_epochs % args.i_save != 0:
		print(f"\n[Final Epoch {num_epochs}] Saving final checkpoint...")
		final_checkpoint_start = datetime.now()
		
		# Save Velocity Model
		for i, model in enumerate(vel_models):
			global_frame_idx = start_frame + i
			model.save(f"{savedir}/ckpt/velrbf_frame_{global_frame_idx:03d}_ckpt_{num_epochs:06d}.pth")
			
			if use_individual_optimizers and vel_optimizers is not None and global_frame_idx in vel_optimizers:
				optimizer_path = os.path.join(savedir, "ckpt", f"velrbf_frame_{global_frame_idx:03d}_optimizer_{num_epochs:06d}.pth")
				torch.save(vel_optimizers[global_frame_idx].state_dict(), optimizer_path)
				
				if vel_schedulers is not None and global_frame_idx in vel_schedulers:
					scheduler_path = os.path.join(savedir, "ckpt", f"velrbf_frame_{global_frame_idx:03d}_scheduler_{num_epochs:06d}.pth")
					torch.save(vel_schedulers[global_frame_idx].state_dict(), scheduler_path)
		
		# Save previous frame velocity model if it was optimized
		if start_frame > 0 and prev_velocity_model_for_advection is not None:
			prev_frame_idx = start_frame - 1
			prev_velocity_model_for_advection.save(
				f"{savedir}/ckpt/velrbf_frame_{prev_frame_idx:03d}_ckpt_{num_epochs:06d}.pth"
			)
			
			if use_individual_optimizers and initial_velocity_optimizers_dict is not None:
				if prev_frame_idx in initial_velocity_optimizers_dict:
					optimizer_path = os.path.join(savedir, "ckpt", f"velrbf_frame_{prev_frame_idx:03d}_optimizer_{num_epochs:06d}.pth")
					torch.save(initial_velocity_optimizers_dict[prev_frame_idx].state_dict(), optimizer_path)
					
					if initial_velocity_schedulers_dict is not None and prev_frame_idx in initial_velocity_schedulers_dict:
						scheduler_path = os.path.join(savedir, "ckpt", f"velrbf_frame_{prev_frame_idx:03d}_scheduler_{num_epochs:06d}.pth")
						torch.save(initial_velocity_schedulers_dict[prev_frame_idx].state_dict(), scheduler_path)
		
		# Save Gaussian Model
		point_cloud_path = os.path.join(savedir, "point_cloud", f"coarse_iteration_{num_epochs}")
		os.makedirs(point_cloud_path, exist_ok=True)
		gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
		gaussians.save_deformation(point_cloud_path)
		# Save checkpoint
		checkpoint_path = os.path.join(savedir, f"chkpnt_coarse_{num_epochs}.pth")
		checkpoint_data = (gaussians.capture(), num_epochs)
		
		# Save inflow Gaussians if they exist
		if inflow_ratio > 0 and inflow_gaussians is not None:
			inflow_checkpoint_path = os.path.join(savedir, f"inflow_gaussians_epoch_{num_epochs}.pth")
			inflow_checkpoint_data = (inflow_gaussians.capture(), num_epochs)
			torch.save(inflow_checkpoint_data, inflow_checkpoint_path)
			checkpoint_data = (gaussians.capture(), num_epochs, inflow_gaussians.capture())
		
		torch.save(checkpoint_data, checkpoint_path)
		print(f"  Saved final checkpoint to {checkpoint_path}")
		
		final_checkpoint_end = datetime.now()
		final_checkpoint_duration = (final_checkpoint_end - final_checkpoint_start).total_seconds()
		total_checkpoint_time += final_checkpoint_duration
		print(f"[Final Epoch {num_epochs}] Final checkpoint saving time: {int(final_checkpoint_duration // 60)}m {final_checkpoint_duration % 60:.2f}s")

	print("Training Complete.")
	
	return {
		'train_time': total_train_time,
		'checkpoint_time': total_checkpoint_time,
		'visualization_time': total_visualization_time,
		'total_time': total_train_time + total_checkpoint_time + total_visualization_time
	}


def train_velocity_model_sliding_window(args, savedir=None, scale: int = 1, gaussian_ckpt_path: str = None, dt = None, 
										visualize_opacity=False, inflow_ratio=0.05, insert_ratio=0.01, 
										visualize_inflow_region=False, w: int = None):
	"""
	： 0->w start， i->i+w， frame_i  GS  velocity
	
	Args:
		args: training arguments
		savedir: output directory
		scale: scaling
		gaussian_ckpt_path:  Gaussian checkpoint path（optional）
		dt: time steps
		visualize_opacity: visualization opacity
		inflow_ratio: inflow 
		insert_ratio: 
		visualize_inflow_region: visualization inflow region
		w: window size（frame）
	"""
	device = set_device(args)
	
	# Load Meta Info to get frame_num
	with open(os.path.join(args.datadir, 'info.json'), 'r') as fp:
		meta = json.load(fp)
		train_video = meta['train_videos'][0]
		total_frame_num = train_video['frame_num']
		args.frame_num = total_frame_num
	
	if w is None:
		w = getattr(args, 'sliding_window_size', total_frame_num)
	
	if w <= 0 or w > total_frame_num:
		raise ValueError(f"Invalid window size: w={w}, total_frame_num={total_frame_num}")
	
	print(f"\n[Sliding Window Training] Window size: {w}, Total frames: {total_frame_num}")
	print(f"  Will train windows: 0->{w}, 1->{w+1}, ..., {total_frame_num-w}->{total_frame_num}")
	
	velocity_optimization_counts = calculate_velocity_optimization_windows(w, total_frame_num)
	print(f"  Velocity optimization counts: {velocity_optimization_counts}")
	
	frame_gaussian_states = {}
	frame_velocity_models = {}
	frame_velocity_optimizers = {}
	frame_velocity_schedulers = {}
	
	os.makedirs(os.path.join(savedir, "frame_gaussians"), exist_ok=True)
	os.makedirs(os.path.join(savedir, "frame_velocities"), exist_ok=True)
	os.makedirs(os.path.join(savedir, "frame_inflow_gaussians"), exist_ok=True)
	
	timing_records = []
	total_start_time = datetime.now()
	
	max_windows = getattr(args, 'max_windows', None)
	for window_count, start_frame in enumerate(range(0, total_frame_num - w + 1)):
		if max_windows is not None and window_count >= max_windows:
			print(f"[Quick Test] Reached max_windows={max_windows}; stopping sliding-window training early.")
			break
		end_frame = start_frame + w
		print(f"\n{'='*80}")
		print(f"[Window {start_frame}->{end_frame}] Training frame {start_frame} to {end_frame-1}")
		print(f"{'='*80}")
		
		initial_gaussians_state = None
		initial_velocity_models_dict = None
		initial_velocity_optimizers_dict = None
		initial_velocity_schedulers_dict = None
		freeze_initial_gaussians_xyz = False
		
		prev_inflow_gaussians_to_merge = None
		prev_inflow_gaussians_to_continue = None
		
		if start_frame > 0:
			if (start_frame - 1) in frame_gaussian_states:
				initial_gaussians_state = frame_gaussian_states[start_frame - 1]
				print(f"  Loading initial GS state from frame {start_frame-1}")
			else:
				gaussian_state_path = os.path.join(savedir, "frame_gaussians", f"frame_{start_frame-1:03d}_gaussian.pth")
				if os.path.exists(gaussian_state_path):
					initial_gaussians_state, _ = torch.load(gaussian_state_path)
					print(f"  Loaded GS state from file: {gaussian_state_path}")
			
			initial_velocity_models_dict = {}
			initial_velocity_optimizers_dict = {}
			initial_velocity_schedulers_dict = {}
			
			for vel_frame_idx in range(start_frame - 1, end_frame - 1):
				if vel_frame_idx in frame_velocity_models:
					initial_velocity_models_dict[vel_frame_idx] = frame_velocity_models[vel_frame_idx]
				else:
					vel_model_path = os.path.join(savedir, "frame_velocities", f"frame_{vel_frame_idx:03d}_velocity.pth")
					if not os.path.exists(vel_model_path):
						prev_window_start = max(0, vel_frame_idx - w + 1)
						prev_window_end = prev_window_start + w
						prev_window_savedir = os.path.join(savedir, f"window_{prev_window_start}_{prev_window_end}")
						import glob
						prev_vel_checkpoint_pattern = os.path.join(prev_window_savedir, "ckpt", f"velrbf_frame_{vel_frame_idx:03d}_ckpt_*.pth")
						prev_vel_checkpoint_files = glob.glob(prev_vel_checkpoint_pattern)
						if prev_vel_checkpoint_files:
							vel_model_path = max(prev_vel_checkpoint_files, key=os.path.getctime)
							print(f"  Found velocity model in previous window: {vel_model_path}")
					
					if os.path.exists(vel_model_path):
						from velocity_common.dfrbf import TiDFRBF
						model = TiDFRBF.load(vel_model_path, device=device)
						initial_velocity_models_dict[vel_frame_idx] = model
						frame_velocity_models[vel_frame_idx] = model
						print(f"  Loaded velocity model from: {vel_model_path}")
				
				if vel_frame_idx in frame_velocity_optimizers:
					initial_velocity_optimizers_dict[vel_frame_idx] = frame_velocity_optimizers[vel_frame_idx]
				else:
					optimizer_path = os.path.join(savedir, "frame_velocities", f"frame_{vel_frame_idx:03d}_optimizer.pth")
					if not os.path.exists(optimizer_path):
						prev_window_start = max(0, vel_frame_idx - w + 1)
						prev_window_end = prev_window_start + w
						prev_window_savedir = os.path.join(savedir, f"window_{prev_window_start}_{prev_window_end}")
						import glob
						prev_optimizer_pattern = os.path.join(prev_window_savedir, "ckpt", f"velrbf_frame_{vel_frame_idx:03d}_optimizer_*.pth")
						prev_optimizer_files = glob.glob(prev_optimizer_pattern)
						if prev_optimizer_files:
							optimizer_path = max(prev_optimizer_files, key=os.path.getctime)
					
					if os.path.exists(optimizer_path) and vel_frame_idx in initial_velocity_models_dict:
						optimizer_state = torch.load(optimizer_path)
						optimizer = torch.optim.Adam(initial_velocity_models_dict[vel_frame_idx].parameters(), lr=args.lrate_vel)
						optimizer.load_state_dict(optimizer_state)
						initial_velocity_optimizers_dict[vel_frame_idx] = optimizer
						frame_velocity_optimizers[vel_frame_idx] = optimizer
						print(f"  Loaded optimizer state from: {optimizer_path}")
				
				if vel_frame_idx in frame_velocity_schedulers:
					initial_velocity_schedulers_dict[vel_frame_idx] = frame_velocity_schedulers[vel_frame_idx]
				else:
					scheduler_path = os.path.join(savedir, "frame_velocities", f"frame_{vel_frame_idx:03d}_scheduler.pth")
					if not os.path.exists(scheduler_path):
						prev_window_start = max(0, vel_frame_idx - w + 1)
						prev_window_end = prev_window_start + w
						prev_window_savedir = os.path.join(savedir, f"window_{prev_window_start}_{prev_window_end}")
						import glob
						prev_scheduler_pattern = os.path.join(prev_window_savedir, "ckpt", f"velrbf_frame_{vel_frame_idx:03d}_scheduler_*.pth")
						prev_scheduler_files = glob.glob(prev_scheduler_pattern)
						if prev_scheduler_files:
							scheduler_path = max(prev_scheduler_files, key=os.path.getctime)
					
					if os.path.exists(scheduler_path) and vel_frame_idx in initial_velocity_optimizers_dict:
						scheduler_state = torch.load(scheduler_path)
						num_windows = velocity_optimization_counts.get(vel_frame_idx, 1)
						total_epochs = num_windows * args.num_epochs
						gamma = 0.01 ** (1.0 / total_epochs)
						scheduler = torch.optim.lr_scheduler.ExponentialLR(
							initial_velocity_optimizers_dict[vel_frame_idx], gamma, verbose=False
						)
						scheduler.load_state_dict(scheduler_state)
						initial_velocity_schedulers_dict[vel_frame_idx] = scheduler
						frame_velocity_schedulers[vel_frame_idx] = scheduler
						print(f"  Loaded scheduler state from: {scheduler_path}")
			
			freeze_initial_gaussians_xyz = True
			
			if inflow_ratio > 0:
				prev_window_start = start_frame - 1
				prev_window_end = prev_window_start + w
				prev_window_savedir = os.path.join(savedir, f"window_{prev_window_start}_{prev_window_end}")
				
				prev_inflow_file = None
				if os.path.exists(prev_window_savedir):
					import glob
					prev_window_inflow_pattern = os.path.join(prev_window_savedir, "inflow_gaussians_epoch_*.pth")
					prev_window_inflow_files = glob.glob(prev_window_inflow_pattern)
					if prev_window_inflow_files:
						prev_inflow_file = max(prev_window_inflow_files, key=os.path.getctime)
						print(f"  Loading previous window inflow from: {prev_inflow_file}")
				
				if prev_inflow_file is not None:
					
					inflow_checkpoint_data = torch.load(prev_inflow_file)
					if isinstance(inflow_checkpoint_data, tuple) and len(inflow_checkpoint_data) >= 2:
						prev_inflow_state = inflow_checkpoint_data[0]
						
						from argparse import ArgumentParser
						from arguments import ModelParams, ModelHiddenParams
						from scene.gaussian_model import GaussianModel
						from velocity_training.models import InflowGaussians
						
						parser = ArgumentParser()
						model_params_obj = ModelParams(parser, sentinel=True)
						hyperparam_obj = ModelHiddenParams(parser)
						model_params = model_params_obj.extract(args)
						hyperparam = hyperparam_obj.extract(args)
						
						temp_gaussians = GaussianModel(model_params.sh_degree, hyperparam)
						if initial_gaussians_state is not None:
							(active_sh_degree, 
							 xyz, 
							 deform_state,
							 deformation_table,
							 features_dc, 
							 features_rest,
							 scaling, 
							 rotation, 
							 opacity,
							 max_radii2D, 
							 xyz_gradient_accum, 
							 denom,
							 opt_dict,
							 spatial_lr_scale) = initial_gaussians_state
							
							temp_gaussians.active_sh_degree = active_sh_degree
							temp_gaussians._xyz = nn.Parameter(xyz.clone().to(device))
							temp_gaussians._deformation.load_state_dict(deform_state)
							temp_gaussians._deformation_table = deformation_table.clone().to(device)
							temp_gaussians._features_dc = nn.Parameter(features_dc.clone().to(device))
							temp_gaussians._features_rest = nn.Parameter(features_rest.clone().to(device))
							temp_gaussians._scaling = nn.Parameter(scaling.clone().to(device))
							temp_gaussians._rotation = nn.Parameter(rotation.clone().to(device))
							temp_gaussians._opacity = nn.Parameter(opacity.clone().to(device))
							temp_gaussians.max_radii2D = max_radii2D.clone().to(device)
							temp_gaussians.spatial_lr_scale = spatial_lr_scale
						
						with open(os.path.join(args.datadir, 'info.json'), 'r') as fp:
							meta = json.load(fp)
							voxel_tran = np.float32(meta['voxel_matrix'])
							voxel_tran = np.stack([voxel_tran[:,2],voxel_tran[:,1],voxel_tran[:,0],voxel_tran[:,3]],axis=1)
							voxel_scale = np.broadcast_to(meta['voxel_scale'],[3])
							voxel_scale = voxel_scale * args.scene_scale
							voxel_tran[:3,3] *= args.scene_scale
						
						from velocity_common.coordinate_transform import CoordinateTransform
						coord_trans_temp = CoordinateTransform(voxel_tran, voxel_scale, device)
						lengths = np.array([args.Nx, args.Ny, args.Nz])
						lengths_tensor_temp = torch.from_numpy(lengths).float().to(device)
						
						prev_inflow_gaussians_full = InflowGaussians.restore(
							prev_inflow_state, temp_gaussians, coord_trans_temp, device
						)
						
						merge_group_indices = []
						continue_group_indices = []
						
						for group_idx in range(prev_inflow_gaussians_full.num_groups):
							origin_frame = prev_inflow_gaussians_full.get_group_origin_frame(group_idx)
							if origin_frame is not None:
								if origin_frame == start_frame - 1:
									merge_group_indices.append(group_idx)
								elif start_frame <= origin_frame < end_frame - 1:
									continue_group_indices.append(group_idx)
						
						print(f"  Found {len(merge_group_indices)} inflow groups to merge directly (origin_frame == {start_frame-1})")
						print(f"  Found {len(continue_group_indices)} inflow groups to continue training ({start_frame} <= origin_frame < {end_frame-1})")
						
						if len(merge_group_indices) > 0:
							merge_inflow_state = {
								'num_groups': len(merge_group_indices),
								'num_points_per_group': prev_inflow_state['num_points_per_group'],
								'lengths_tensor': prev_inflow_state['lengths_tensor'],
								'inflow_ratio': prev_inflow_state['inflow_ratio'],
								'max_sh_degree': prev_inflow_state['max_sh_degree'],
								'initialized_groups': [prev_inflow_state['initialized_groups'][g] for g in merge_group_indices],
								'group_origin_frames': [prev_inflow_state['group_origin_frames'][g] for g in merge_group_indices],
								'xyz_groups': [prev_inflow_state['xyz_groups'][g] for g in merge_group_indices],
								'features_dc_groups': [prev_inflow_state['features_dc_groups'][g] for g in merge_group_indices],
								'features_rest_groups': [prev_inflow_state['features_rest_groups'][g] for g in merge_group_indices],
								'opacity_groups': [prev_inflow_state['opacity_groups'][g] for g in merge_group_indices],
								'scaling_groups': [prev_inflow_state['scaling_groups'][g] for g in merge_group_indices],
								'rotation_groups': [prev_inflow_state['rotation_groups'][g] for g in merge_group_indices],
							}
							
							prev_inflow_gaussians_to_merge = InflowGaussians.restore(
								merge_inflow_state, temp_gaussians, coord_trans_temp, device
							)
							
							if initial_gaussians_state is not None:
								from arguments import OptimizationParams
								op = OptimizationParams(parser)
								opt = op.extract(args)
								gaussians = GaussianModel(model_params.sh_degree, hyperparam)
								gaussians.restore(initial_gaussians_state, opt)
								
								all_merge_indices = list(range(len(merge_group_indices)))
								gaussians, inflow_point_indices = _merge_inflow_to_gaussians(
									gaussians, prev_inflow_gaussians_to_merge, all_merge_indices,
									coord_trans_temp, lengths_tensor_temp, device
								)
								
								gaussians.training_setup(opt)
								
								initial_gaussians_state = gaussians.capture()
								print(f"  Merged {inflow_point_indices.sum().item()} inflow points (origin_frame == {start_frame-1}) into initial GS state")
						
						if len(continue_group_indices) > 0:
							continue_inflow_state = {
								'num_groups': len(continue_group_indices),
								'num_points_per_group': prev_inflow_state['num_points_per_group'],
								'lengths_tensor': prev_inflow_state['lengths_tensor'],
								'inflow_ratio': prev_inflow_state['inflow_ratio'],
								'max_sh_degree': prev_inflow_state['max_sh_degree'],
								'initialized_groups': [prev_inflow_state['initialized_groups'][g] for g in continue_group_indices],
								'group_origin_frames': [prev_inflow_state['group_origin_frames'][g] for g in continue_group_indices],
								'xyz_groups': [prev_inflow_state['xyz_groups'][g] for g in continue_group_indices],
								'features_dc_groups': [prev_inflow_state['features_dc_groups'][g] for g in continue_group_indices],
								'features_rest_groups': [prev_inflow_state['features_rest_groups'][g] for g in continue_group_indices],
								'opacity_groups': [prev_inflow_state['opacity_groups'][g] for g in continue_group_indices],
								'scaling_groups': [prev_inflow_state['scaling_groups'][g] for g in continue_group_indices],
								'rotation_groups': [prev_inflow_state['rotation_groups'][g] for g in continue_group_indices],
							}
							
							prev_inflow_gaussians_to_continue = InflowGaussians.restore(
								continue_inflow_state, temp_gaussians, coord_trans_temp, device
							)
							print(f"  Created InflowGaussians object with {len(continue_group_indices)} groups to continue training")
		
		window_savedir = os.path.join(savedir, f"window_{start_frame}_{end_frame}")
		os.makedirs(window_savedir, exist_ok=True)
		
		use_progressive = (start_frame == 0)
		subsequent_window_epochs = getattr(args, 'sliding_window_subsequent_epochs', None)
		if subsequent_window_epochs is None:
			subsequent_window_epochs = max(1, args.num_epochs // 2)
		
		num_epochs_for_window = args.num_epochs if start_frame == 0 else subsequent_window_epochs
		
		if start_frame > 0:
			print(f"  Using {num_epochs_for_window} epochs for subsequent window (no progressive training)")
		
		window_train_start = datetime.now()
		
		timing_info = train_velocity_model_with_gaussian(
			args, savedir=window_savedir, scale=scale, gaussian_ckpt_path=gaussian_ckpt_path if start_frame == 0 else None,
			dt=dt, visualize_opacity=visualize_opacity, inflow_ratio=inflow_ratio, insert_ratio=insert_ratio,
			visualize_inflow_region=visualize_inflow_region, start_frame=start_frame, end_frame=end_frame,
			initial_gaussians_state=initial_gaussians_state,
			initial_velocity_models_dict=initial_velocity_models_dict,
			initial_velocity_optimizers_dict=initial_velocity_optimizers_dict,
			initial_velocity_schedulers_dict=initial_velocity_schedulers_dict,
			freeze_initial_gaussians_xyz=freeze_initial_gaussians_xyz,
			velocity_optimization_counts=velocity_optimization_counts,
			use_individual_optimizers=True,
			use_progressive_training=use_progressive,
			num_epochs_override=num_epochs_for_window,
			existing_inflow_gaussians=prev_inflow_gaussians_to_continue
		)
		
		window_train_end = datetime.now()
		window_train_duration = (window_train_end - window_train_start).total_seconds()
		
		if timing_info is None:
			train_time = window_train_duration
			checkpoint_time = 0.0
			visualization_time = 0.0
		else:
			train_time = timing_info['train_time']
			checkpoint_time = timing_info['checkpoint_time']
			visualization_time = timing_info['visualization_time']
		
		window_save_start = datetime.now()
		
		checkpoint_path = os.path.join(window_savedir, f"chkpnt_coarse_{num_epochs_for_window}.pth")
		if not os.path.exists(checkpoint_path):
			import glob
			checkpoint_pattern = os.path.join(window_savedir, "chkpnt_coarse_*.pth")
			checkpoint_files = glob.glob(checkpoint_pattern)
			if checkpoint_files:
				checkpoint_path = max(checkpoint_files, key=os.path.getctime)
				print(f"  Using latest checkpoint: {checkpoint_path}")
		
		if os.path.exists(checkpoint_path):
			checkpoint_data = torch.load(checkpoint_path)
			if isinstance(checkpoint_data, tuple) and len(checkpoint_data) >= 2:
				gaussian_state = checkpoint_data[0]
				frame_gaussian_states[start_frame] = gaussian_state
				
				gaussian_state_path = os.path.join(savedir, "frame_gaussians", f"frame_{start_frame:03d}_gaussian.pth")
				torch.save((gaussian_state, num_epochs_for_window), gaussian_state_path)
				print(f"  Saved GS state to: {gaussian_state_path}")
			else:
				print(f"  Warning: Invalid checkpoint format at {checkpoint_path}")
		else:
			print(f"  Warning: Checkpoint not found at {checkpoint_path}")
		
		for vel_frame_idx in range(start_frame, end_frame - 1):
			vel_checkpoint_path = os.path.join(window_savedir, "ckpt", f"velrbf_frame_{vel_frame_idx:03d}_ckpt_{num_epochs_for_window:06d}.pth")
			if not os.path.exists(vel_checkpoint_path):
				import glob
				vel_checkpoint_pattern = os.path.join(window_savedir, "ckpt", f"velrbf_frame_{vel_frame_idx:03d}_ckpt_*.pth")
				vel_checkpoint_files = glob.glob(vel_checkpoint_pattern)
				if vel_checkpoint_files:
					vel_checkpoint_path = max(vel_checkpoint_files, key=os.path.getctime)
					print(f"  Using latest velocity checkpoint: {vel_checkpoint_path}")
			
			if os.path.exists(vel_checkpoint_path):
				from velocity_common.dfrbf import TiDFRBF
				model = TiDFRBF.load(vel_checkpoint_path, device=device)
				frame_velocity_models[vel_frame_idx] = model
				
				vel_model_path = os.path.join(savedir, "frame_velocities", f"frame_{vel_frame_idx:03d}_velocity.pth")
				model.save(vel_model_path)
				print(f"  Saved velocity model to: {vel_model_path}")
				
				optimizer_checkpoint_path = os.path.join(window_savedir, "ckpt", f"velrbf_frame_{vel_frame_idx:03d}_optimizer_{num_epochs_for_window:06d}.pth")
				if not os.path.exists(optimizer_checkpoint_path):
					import glob
					optimizer_pattern = os.path.join(window_savedir, "ckpt", f"velrbf_frame_{vel_frame_idx:03d}_optimizer_*.pth")
					optimizer_files = glob.glob(optimizer_pattern)
					if optimizer_files:
						optimizer_checkpoint_path = max(optimizer_files, key=os.path.getctime)
				
				if os.path.exists(optimizer_checkpoint_path):
					optimizer_state = torch.load(optimizer_checkpoint_path)
					optimizer_save_path = os.path.join(savedir, "frame_velocities", f"frame_{vel_frame_idx:03d}_optimizer.pth")
					torch.save(optimizer_state, optimizer_save_path)
					print(f"  Saved optimizer state to: {optimizer_save_path}")
				
				scheduler_checkpoint_path = os.path.join(window_savedir, "ckpt", f"velrbf_frame_{vel_frame_idx:03d}_scheduler_{num_epochs_for_window:06d}.pth")
				if not os.path.exists(scheduler_checkpoint_path):
					import glob
					scheduler_pattern = os.path.join(window_savedir, "ckpt", f"velrbf_frame_{vel_frame_idx:03d}_scheduler_*.pth")
					scheduler_files = glob.glob(scheduler_pattern)
					if scheduler_files:
						scheduler_checkpoint_path = max(scheduler_files, key=os.path.getctime)
				
				if os.path.exists(scheduler_checkpoint_path):
					scheduler_state = torch.load(scheduler_checkpoint_path)
					scheduler_save_path = os.path.join(savedir, "frame_velocities", f"frame_{vel_frame_idx:03d}_scheduler.pth")
					torch.save(scheduler_state, scheduler_save_path)
					print(f"  Saved scheduler state to: {scheduler_save_path}")
			else:
				print(f"  Warning: Velocity checkpoint not found for frame {vel_frame_idx}")
		
		if start_frame > 0:
			prev_frame_idx = start_frame - 1
			prev_vel_checkpoint_path = os.path.join(window_savedir, "ckpt", f"velrbf_frame_{prev_frame_idx:03d}_ckpt_{num_epochs_for_window:06d}.pth")
			if not os.path.exists(prev_vel_checkpoint_path):
				import glob
				prev_vel_checkpoint_pattern = os.path.join(window_savedir, "ckpt", f"velrbf_frame_{prev_frame_idx:03d}_ckpt_*.pth")
				prev_vel_checkpoint_files = glob.glob(prev_vel_checkpoint_pattern)
				if prev_vel_checkpoint_files:
					prev_vel_checkpoint_path = max(prev_vel_checkpoint_files, key=os.path.getctime)
					print(f"  Using latest previous frame velocity checkpoint: {prev_vel_checkpoint_path}")
			
			if os.path.exists(prev_vel_checkpoint_path):
				from velocity_common.dfrbf import TiDFRBF
				prev_model = TiDFRBF.load(prev_vel_checkpoint_path, device=device)
				frame_velocity_models[prev_frame_idx] = prev_model
				
				prev_vel_model_path = os.path.join(savedir, "frame_velocities", f"frame_{prev_frame_idx:03d}_velocity.pth")
				prev_model.save(prev_vel_model_path)
				print(f"  Saved updated velocity model for frame {prev_frame_idx} to: {prev_vel_model_path}")
				
				prev_optimizer_checkpoint_path = os.path.join(window_savedir, "ckpt", f"velrbf_frame_{prev_frame_idx:03d}_optimizer_{num_epochs_for_window:06d}.pth")
				if not os.path.exists(prev_optimizer_checkpoint_path):
					import glob
					prev_optimizer_pattern = os.path.join(window_savedir, "ckpt", f"velrbf_frame_{prev_frame_idx:03d}_optimizer_*.pth")
					prev_optimizer_files = glob.glob(prev_optimizer_pattern)
					if prev_optimizer_files:
						prev_optimizer_checkpoint_path = max(prev_optimizer_files, key=os.path.getctime)
				
				if os.path.exists(prev_optimizer_checkpoint_path):
					prev_optimizer_state = torch.load(prev_optimizer_checkpoint_path)
					prev_optimizer_save_path = os.path.join(savedir, "frame_velocities", f"frame_{prev_frame_idx:03d}_optimizer.pth")
					torch.save(prev_optimizer_state, prev_optimizer_save_path)
					print(f"  Saved optimizer state for frame {prev_frame_idx} to: {prev_optimizer_save_path}")
				
				prev_scheduler_checkpoint_path = os.path.join(window_savedir, "ckpt", f"velrbf_frame_{prev_frame_idx:03d}_scheduler_{num_epochs_for_window:06d}.pth")
				if not os.path.exists(prev_scheduler_checkpoint_path):
					import glob
					prev_scheduler_pattern = os.path.join(window_savedir, "ckpt", f"velrbf_frame_{prev_frame_idx:03d}_scheduler_*.pth")
					prev_scheduler_files = glob.glob(prev_scheduler_pattern)
					if prev_scheduler_files:
						prev_scheduler_checkpoint_path = max(prev_scheduler_files, key=os.path.getctime)
				
				if os.path.exists(prev_scheduler_checkpoint_path):
					prev_scheduler_state = torch.load(prev_scheduler_checkpoint_path)
					prev_scheduler_save_path = os.path.join(savedir, "frame_velocities", f"frame_{prev_frame_idx:03d}_scheduler.pth")
					torch.save(prev_scheduler_state, prev_scheduler_save_path)
					print(f"  Saved scheduler state for frame {prev_frame_idx} to: {prev_scheduler_save_path}")
		
		next_window_exists = (start_frame + 1 <= total_frame_num - w)
		if next_window_exists:
			keep_vel_keys = set(range(start_frame, end_frame))
			keep_gaussian_keys = {start_frame}
		else:
			keep_vel_keys = set()
			keep_gaussian_keys = set()
		for key in list(frame_velocity_models.keys()):
			if key not in keep_vel_keys:
				del frame_velocity_models[key]
		for key in list(frame_velocity_optimizers.keys()):
			if key not in keep_vel_keys:
				del frame_velocity_optimizers[key]
		for key in list(frame_velocity_schedulers.keys()):
			if key not in keep_vel_keys:
				del frame_velocity_schedulers[key]
		for key in list(frame_gaussian_states.keys()):
			if key not in keep_gaussian_keys:
				del frame_gaussian_states[key]
		_release_unused_memory()
		if next_window_exists:
			print(f"  [Memory] Kept velocity models for frames {sorted(keep_vel_keys)}, GS state for frame {start_frame}; cleared the rest.")
		else:
			print(f"  [Memory] Last window: cleared all in-memory velocity/optimizer/scheduler/GS state.")
		
		window_save_end = datetime.now()
		window_save_duration = (window_save_end - window_save_start).total_seconds()
		
		def format_time(seconds):
			return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {seconds % 60:.2f}s"
		
		timing_record = {
			'window': f"{start_frame}->{end_frame}",
			'start_frame': start_frame,
			'end_frame': end_frame,
			'num_epochs': num_epochs_for_window,
			'train_time_seconds': train_time,
			'train_time_formatted': format_time(train_time),
			'checkpoint_time_seconds': checkpoint_time,
			'checkpoint_time_formatted': format_time(checkpoint_time),
			'visualization_time_seconds': visualization_time,
			'visualization_time_formatted': format_time(visualization_time),
			'save_duration_seconds': window_save_duration,
			'save_duration_formatted': format_time(window_save_duration),
			'train_total_time_seconds': train_time + checkpoint_time + visualization_time,
			'train_total_time_formatted': format_time(train_time + checkpoint_time + visualization_time),
			'total_duration_seconds': train_time + checkpoint_time + visualization_time + window_save_duration,
			'total_duration_formatted': format_time(train_time + checkpoint_time + visualization_time + window_save_duration)
		}
		timing_records.append(timing_record)
		
		print(f"\n  [Window {start_frame}->{end_frame}] Time breakdown:")
		print(f"    Training time: {timing_record['train_time_formatted']}")
		print(f"    Checkpoint saving time: {timing_record['checkpoint_time_formatted']}")
		print(f"    Visualization time: {timing_record['visualization_time_formatted']}")
		print(f"    Model saving time: {timing_record['save_duration_formatted']}")
		print(f"    Total time: {timing_record['total_duration_formatted']}")
	
	total_end_time = datetime.now()
	total_duration = (total_end_time - total_start_time).total_seconds()
	
	print(f"\n{'='*80}")
	print("Sliding Window Training Complete!")
	print(f"{'='*80}")
	
	timing_file_path = os.path.join(savedir, "timing.txt")
	with open(timing_file_path, 'w', encoding='utf-8') as f:
		f.write("=" * 80 + "\n")
		f.write("Sliding Window Training Timing Report\n")
		f.write("=" * 80 + "\n\n")
		f.write(f"Window size: {w}\n")
		f.write(f"Total frames: {total_frame_num}\n")
		f.write(f"Total windows: {len(timing_records)}\n")
		f.write(f"Start time: {total_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
		f.write(f"End time: {total_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
		f.write(f"Total duration: {int(total_duration // 3600)}h {int((total_duration % 3600) // 60)}m {total_duration % 60:.2f}s\n")
		f.write(f"Total duration (seconds): {total_duration:.2f}\n\n")
		
		f.write("-" * 80 + "\n")
		f.write("Per-Window Timing Details:\n")
		f.write("-" * 80 + "\n\n")
		
		for i, record in enumerate(timing_records, 1):
			f.write(f"Window {i}: {record['window']}\n")
			f.write(f"  Frames: {record['start_frame']} to {record['end_frame']-1}\n")
			f.write(f"  Epochs: {record['num_epochs']}\n")
			f.write(f"  Training time (forward + backward): {record['train_time_formatted']} ({record['train_time_seconds']:.2f}s)\n")
			f.write(f"  Checkpoint saving time: {record['checkpoint_time_formatted']} ({record['checkpoint_time_seconds']:.2f}s)\n")
			f.write(f"  Visualization time: {record['visualization_time_formatted']} ({record['visualization_time_seconds']:.2f}s)\n")
			f.write(f"  Model saving time (sliding window): {record['save_duration_formatted']} ({record['save_duration_seconds']:.2f}s)\n")
			f.write(f"  Total time: {record['total_duration_formatted']} ({record['total_duration_seconds']:.2f}s)\n")
			f.write("\n")
		
		f.write("-" * 80 + "\n")
		f.write("Statistics:\n")
		f.write("-" * 80 + "\n\n")
		
		total_train_time = sum(r['train_time_seconds'] for r in timing_records)
		total_checkpoint_time = sum(r['checkpoint_time_seconds'] for r in timing_records)
		total_visualization_time = sum(r['visualization_time_seconds'] for r in timing_records)
		total_save_time = sum(r['save_duration_seconds'] for r in timing_records)
		
		avg_train_time = total_train_time / len(timing_records) if timing_records else 0
		avg_checkpoint_time = total_checkpoint_time / len(timing_records) if timing_records else 0
		avg_visualization_time = total_visualization_time / len(timing_records) if timing_records else 0
		avg_save_time = total_save_time / len(timing_records) if timing_records else 0
		
		def format_time_for_file(seconds):
			return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {seconds % 60:.2f}s"
		
		f.write(f"Total training time (forward + backward): {format_time_for_file(total_train_time)} ({total_train_time:.2f}s)\n")
		f.write(f"Total checkpoint saving time: {format_time_for_file(total_checkpoint_time)} ({total_checkpoint_time:.2f}s)\n")
		f.write(f"Total visualization time: {format_time_for_file(total_visualization_time)} ({total_visualization_time:.2f}s)\n")
		f.write(f"Total model saving time (sliding window): {format_time_for_file(total_save_time)} ({total_save_time:.2f}s)\n")
		f.write(f"Total time: {format_time_for_file(total_train_time + total_checkpoint_time + total_visualization_time + total_save_time)} ({total_train_time + total_checkpoint_time + total_visualization_time + total_save_time:.2f}s)\n")
		f.write("\n")
		f.write(f"Average training time per window: {format_time_for_file(avg_train_time)} ({avg_train_time:.2f}s)\n")
		f.write(f"Average checkpoint saving time per window: {format_time_for_file(avg_checkpoint_time)} ({avg_checkpoint_time:.2f}s)\n")
		f.write(f"Average visualization time per window: {format_time_for_file(avg_visualization_time)} ({avg_visualization_time:.2f}s)\n")
		f.write(f"Average model saving time per window: {format_time_for_file(avg_save_time)} ({avg_save_time:.2f}s)\n")
	
	print(f"\nTiming report saved to: {timing_file_path}")
	print(f"Total training time: {int(total_duration // 3600)}h {int((total_duration % 3600) // 60)}m {total_duration % 60:.2f}s")


def load_sliding_window_gaussians(args, savedir, target_frame, device="cuda"):
	"""
	 GS，used for
	
	Args:
		args: training arguments
		savedir: output directory
		target_frame: frame index
		device: device
	
	Returns:
		gaussians: GaussianModel object，position advection ，attributes
	"""
	from argparse import ArgumentParser
	from arguments import ModelParams, ModelHiddenParams, OptimizationParams
	from scene.gaussian_model import GaussianModel
	from velocity_common.dfrbf import TiDFRBF
	import utils.grid_utils as utils_grid
	
	gaussian_state_path = os.path.join(savedir, "frame_gaussians", f"frame_000_gaussian.pth")
	if not os.path.exists(gaussian_state_path):
		raise FileNotFoundError(f"Frame 0 Gaussian state not found: {gaussian_state_path}")
	
	gaussian_state, _ = torch.load(gaussian_state_path)
	
	parser = ArgumentParser()
	model_params_obj = ModelParams(parser, sentinel=True)
	hyperparam_obj = ModelHiddenParams(parser)
	op = OptimizationParams(parser)
	
	model_params = model_params_obj.extract(args)
	hyperparam = hyperparam_obj.extract(args)
	opt = op.extract(args)
	
	gaussians = GaussianModel(model_params.sh_degree, hyperparam)
	gaussians.restore(gaussian_state, opt)
	
	target_gaussian_state_path = os.path.join(savedir, "frame_gaussians", f"frame_{target_frame:03d}_gaussian.pth")
	if os.path.exists(target_gaussian_state_path):
		target_gaussian_state, _ = torch.load(target_gaussian_state_path)
		(active_sh_degree, xyz, deform_state, deformation_table, features_dc, features_rest,
		 scaling, rotation, opacity, max_radii2D, xyz_gradient_accum, denom, opt_dict, spatial_lr_scale) = target_gaussian_state
		
		gaussians._features_dc.data = features_dc
		gaussians._features_rest.data = features_rest
		gaussians._scaling.data = scaling
		gaussians._rotation.data = rotation
		gaussians._opacity.data = opacity
		gaussians.max_radii2D = max_radii2D
	
	with open(os.path.join(args.datadir, 'info.json'), 'r') as fp:
		meta = json.load(fp)
		voxel_tran = np.float32(meta['voxel_matrix'])
		voxel_tran = np.stack([voxel_tran[:,2],voxel_tran[:,1],voxel_tran[:,0],voxel_tran[:,3]],axis=1)
		voxel_scale = np.broadcast_to(meta['voxel_scale'],[3])
		voxel_scale = voxel_scale * args.scene_scale
		voxel_tran[:3,3] *= args.scene_scale
	
	coord_trans = CoordinateTransform(voxel_tran, voxel_scale, device)
	lengths = np.array([args.Nx, args.Ny, args.Nz])
	lengths_tensor = torch.from_numpy(lengths).float().to(device)
	
	scale = getattr(args, 'scale', 1)
	s = float(scale)
	nx, ny, nz = int(args.Nx/s), int(args.Ny/s), int(args.Nz/s)
	grid_shape = (nx, ny, nz)
	min_corner = np.zeros(3)
	max_corner = lengths
	grid_points_np = utils_grid.generate_gridpoints(grid_shape, min_corner, max_corner)
	grid_points = torch.from_numpy(grid_points_np).float().to(device)
	
	dt = getattr(args, 'dt', None)
	if dt is None:
		dt = (args.sim_steps / s)
	
	xyz_world = gaussians.get_xyz.detach().clone()
	xyz_smoke = coord_trans.world2smoke(xyz_world)
	xyz_sim = xyz_smoke * lengths_tensor
	
	with torch.no_grad():
		for frame_idx in range(target_frame):
			vel_model_path = os.path.join(savedir, "frame_velocities", f"frame_{frame_idx:03d}_velocity.pth")
			if not os.path.exists(vel_model_path):
				raise FileNotFoundError(f"Velocity model not found: {vel_model_path}")
			
			vel_model = TiDFRBF.load(vel_model_path, device=device)
			
			v_flat = vel_model(grid_points)
			v_vol = v_flat.view(nx, ny, nz, 3).permute(3, 2, 1, 0).unsqueeze(0)
			
			norm_pos = xyz_sim.clone()
			norm_pos = 2 * (norm_pos / lengths_tensor) - 1.0
			grid_in = norm_pos.view(1, 1, 1, -1, 3)
			v_part = F.grid_sample(v_vol, grid_in, align_corners=True, mode='bilinear')
			v_part = v_part.view(3, -1).permute(1, 0)
			
			xyz_sim = xyz_sim + v_part * dt
		
		xyz_smoke = xyz_sim / lengths_tensor
		xyz_world = coord_trans.smoke2world(xyz_smoke)
		
		gaussians._xyz.data = xyz_world
	
	return gaussians
