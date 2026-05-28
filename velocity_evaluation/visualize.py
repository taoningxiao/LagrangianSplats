import os
import json
import torch
import numpy as np
from tqdm import tqdm
import imageio
import glob
import re
import cv2 as cv
import torch.nn.functional as F
from velocity_common.dfrbf import TiDFRBF
from velocity_common.rbf import WendlandC4
import utils.grid_utils as utils_grid
import torch.nn as nn
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky, LinAlgError

try:
	import cupy as cp
	import cupyx.scipy.sparse as cusp
	from cupyx.scipy.sparse.linalg import cg as cupy_cg
	CUPY_AVAILABLE = True
except ImportError:
	CUPY_AVAILABLE = False
	print("Warning: cupy is unavailable，using the CPU version scipy.sparse.linalg.cg")
	from scipy.sparse.linalg import cg as scipy_cg

from velocity_common.coordinate_transform import CoordinateTransform
from velocity_common.kernels import generate_kernels
from velocity_common.utils import get_background_color, set_device

# Visualization helper functions (from recons_vel_kernel.py)
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


def jacobian3D_np(x):
	# x, (b,)d,h,w,ch
	# return jacobian and curl

	if len(x.shape) < 5:
		x = np.expand_dims(x, axis=0)
	dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
	dvdx = x[:,:,:,1:,1] - x[:,:,:,:-1,1]
	dwdx = x[:,:,:,1:,2] - x[:,:,:,:-1,2]
	dudy = x[:,:,1:,:,0] - x[:,:,:-1,:,0]
	dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
	dwdy = x[:,:,1:,:,2] - x[:,:,:-1,:,2]
	dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
	dvdz = x[:,1:,:,:,1] - x[:,:-1,:,:,1]
	dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

	# u = dwdy[:,:-1,:,:-1] - dvdz[:,:,1:,:-1]
	# v = dudz[:,:,1:,:-1] - dwdx[:,:-1,1:,:]
	# w = dvdx[:,:-1,1:,:] - dudy[:,:-1,:,:-1]

	dudx = np.concatenate((dudx, np.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
	dvdx = np.concatenate((dvdx, np.expand_dims(dvdx[:,:,:,-1], axis=3)), axis=3)
	dwdx = np.concatenate((dwdx, np.expand_dims(dwdx[:,:,:,-1], axis=3)), axis=3)

	dudy = np.concatenate((dudy, np.expand_dims(dudy[:,:,-1,:], axis=2)), axis=2)
	dvdy = np.concatenate((dvdy, np.expand_dims(dvdy[:,:,-1,:], axis=2)), axis=2)
	dwdy = np.concatenate((dwdy, np.expand_dims(dwdy[:,:,-1,:], axis=2)), axis=2)

	dudz = np.concatenate((dudz, np.expand_dims(dudz[:,-1,:,:], axis=1)), axis=1)
	dvdz = np.concatenate((dvdz, np.expand_dims(dvdz[:,-1,:,:], axis=1)), axis=1)
	dwdz = np.concatenate((dwdz, np.expand_dims(dwdz[:,-1,:,:], axis=1)), axis=1)

	u = dwdy - dvdz
	v = dudz - dwdx
	w = dvdx - dudy
	
	j = np.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], axis=-1)
	c = np.stack([u,v,w], axis=-1)
	
	return j, c


def div_np(x):
	"""
	velocity field（finite differences）
	
	Args:
		x: velocity field，shape (z, y, x, 3)  (batch, z, y, x, 3)
		   where velocity[..., 0] = vx, velocity[..., 1] = vy, velocity[..., 2] = vz
		   
	Note:
		This function expects input format (z, y, x, 3)，：
		- axis=1  z direction
		- axis=2  y direction  
		- axis=3  x direction
		 GT  vel_uv2hsv 。
		
		If model output is (x, y, z, 3) ，first swapaxes(0, 2) convert to (z, y, x, 3)。
	
	Returns:
		div: ，shape (batch, z, y, x)  (z, y, x)
	"""
	if len(x.shape) < 5:
		x = np.expand_dims(x, axis=0)
	dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
	dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
	dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]
	
	dudx = np.concatenate((dudx, np.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
	dvdy = np.concatenate((dvdy, np.expand_dims(dvdy[:,:,-1,:], axis=2)), axis=2)
	dwdz = np.concatenate((dwdz, np.expand_dims(dwdz[:,-1,:,:], axis=1)), axis=1)
	
	div = dudx + dvdy + dwdz
	return div


def create_inflow_mask_torch(density_shape, inflow_height_ratio, device, dtype):
	"""
	Create the inflow-region mask（PyTorch）
	
	Args:
		density_shape: shape (Nx, Ny, Nz)
		inflow_height_ratio: inflow-height ratio
		device: device
		dtype: dtype
	
	Returns:
		mask:  [Nx, Ny, Nz] (torch.Tensor)
	"""
	Nx, Ny, Nz = density_shape
	Ly = Ny
	
	y = torch.linspace(0, Ly, Ny, device=device, dtype=dtype)
	Y = y.unsqueeze(0).unsqueeze(-1).expand(Nx, Ny, Nz)
	
	mask = Y < (inflow_height_ratio * Ly)
	
	return mask


def compute_velocity_gradient_np(velocity_field):
	"""
	 ∇u（numpy）
	
	Args:
		velocity_field: velocity field，shape (Nz, Ny, Nx, 3)
	
	Returns:
		grad_v: ，shape (Nz, Ny, Nx, 3, 3)
		framevelocity component(u,v,w)，framecoordinate(x,y,z)
	"""
	assert velocity_field.ndim == 4 and velocity_field.shape[-1] == 3, "velocity_field must be (Nz,Ny,Nx,3)"
	
	Nz, Ny, Nx = velocity_field.shape[:3]
	vx = velocity_field[:, :, :, 0]  # (Nz, Ny, Nx)
	vy = velocity_field[:, :, :, 1]
	vz = velocity_field[:, :, :, 2]
	
	grad_vx = np.gradient(vx)  # [grad_z, grad_y, grad_x]
	grad_vy = np.gradient(vy)
	grad_vz = np.gradient(vz)
	
	dx_vx, dy_vx, dz_vx = grad_vx[2], grad_vx[1], grad_vx[0]
	dx_vy, dy_vy, dz_vy = grad_vy[2], grad_vy[1], grad_vy[0]
	dx_vz, dy_vz, dz_vz = grad_vz[2], grad_vz[1], grad_vz[0]
	
	grad_v = np.stack([
		np.stack([dx_vx, dy_vx, dz_vx], axis=-1),
		np.stack([dx_vy, dy_vy, dz_vy], axis=-1),
		np.stack([dx_vz, dy_vz, dz_vz], axis=-1),
	], axis=-2)
	
	return grad_v


def compute_convective_term_np(velocity_field, grad_v):
	"""
	 u·∇u
	
	Args:
		velocity_field: velocity field，shape (Nz, Ny, Nx, 3)
		grad_v: ，shape (Nz, Ny, Nx, 3, 3)
	
	Returns:
		convective: ，shape (Nz, Ny, Nx, 3)
	"""
	# u·∇u = u_x * ∂u/∂x + u_y * ∂u/∂y + u_z * ∂u/∂z
	
	convective = np.zeros_like(velocity_field)
	
	for comp in range(3):
		# u·∇u[comp] = u_x * ∂u[comp]/∂x + u_y * ∂u[comp]/∂y + u_z * ∂u[comp]/∂z
		for coord in range(3):
			convective[:, :, :, comp] += velocity_field[:, :, :, coord] * grad_v[:, :, :, comp, coord]
	
	return convective


def build_laplacian_3d(grid_shape, dx=1.0, dy=1.0, dz=1.0):
	"""
	3DLaplacian
	
	Args:
		grid_shape: shape (Nx, Ny, Nz)
		dx, dy, dz: （1.0）
	
	Returns:
		laplacian: scipy（CSC），shape (Nx*Ny*Nz, Nx*Ny*Nz)
	"""
	Nx, Ny, Nz = grid_shape
	N = Nx * Ny * Nz
	
	# (p[i+1,j,k] - 2*p[i,j,k] + p[i-1,j,k])/dx^2 +
	# (p[i,j+1,k] - 2*p[i,j,k] + p[i,j-1,k])/dy^2 +
	# (p[i,j,k+1] - 2*p[i,j,k] + p[i,j,k-1])/dz^2
	
	
	data = []
	row_indices = []
	col_indices = []
	
	coeff_x = 1.0 / (dx * dx)
	coeff_y = 1.0 / (dy * dy)
	coeff_z = 1.0 / (dz * dz)
	coeff_center = 2.0 * (coeff_x + coeff_y + coeff_z)
	
	for k in range(Nz):
		for j in range(Ny):
			for i in range(Nx):
				idx = i + j * Nx + k * Nx * Ny
				
				data.append(coeff_center)
				row_indices.append(idx)
				col_indices.append(idx)
				
				if i > 0:
					idx_xm = (i-1) + j * Nx + k * Nx * Ny
					data.append(coeff_x)
					row_indices.append(idx)
					col_indices.append(idx_xm)
				if i < Nx - 1:
					idx_xp = (i+1) + j * Nx + k * Nx * Ny
					data.append(coeff_x)
					row_indices.append(idx)
					col_indices.append(idx_xp)
				
				if j > 0:
					idx_ym = i + (j-1) * Nx + k * Nx * Ny
					data.append(coeff_y)
					row_indices.append(idx)
					col_indices.append(idx_ym)
				if j < Ny - 1:
					idx_yp = i + (j+1) * Nx + k * Nx * Ny
					data.append(coeff_y)
					row_indices.append(idx)
					col_indices.append(idx_yp)
				
				if k > 0:
					idx_zm = i + j * Nx + (k-1) * Nx * Ny
					data.append(coeff_z)
					row_indices.append(idx)
					col_indices.append(idx_zm)
				if k < Nz - 1:
					idx_zp = i + j * Nx + (k+1) * Nx * Ny
					data.append(coeff_z)
					row_indices.append(idx)
					col_indices.append(idx_zp)
	
	laplacian = csc_matrix((data, (row_indices, col_indices)), shape=(N, N))
	
	return laplacian


def apply_boundary_conditions(laplacian, rhs, grid_shape, obstacle_mask, rhs_vector_field):
	"""
	boundary conditionsLaplacianvector
	
	Args:
		laplacian: Laplacian
		rhs: vector（）
		grid_shape: shape (Nx, Ny, Nz)
		obstacle_mask: mask，shape (Nx, Ny, Nz)，True
		rhs_vector_field: vector (du/dt + u·∇u)，shape (Nz, Ny, Nx, 3)，used forNeumannboundary conditions
	
	Returns:
		laplacian: Laplacian
		rhs: vector
	"""
	Nx, Ny, Nz = grid_shape
	N = Nx * Ny * Nz
	
	dirichlet_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
	dirichlet_mask[0, :, :] = True
	dirichlet_mask[-1, :, :] = True
	dirichlet_mask[:, 0, :] = True
	dirichlet_mask[:, -1, :] = True
	dirichlet_mask[:, :, 0] = True
	dirichlet_mask[:, :, -1] = True
	
	obstacle_boundary = np.zeros((Nx, Ny, Nz), dtype=bool)
	if obstacle_mask is not None and obstacle_mask.any():
		neighbor_xm = np.zeros_like(obstacle_mask)
		neighbor_xp = np.zeros_like(obstacle_mask)
		neighbor_ym = np.zeros_like(obstacle_mask)
		neighbor_yp = np.zeros_like(obstacle_mask)
		neighbor_zm = np.zeros_like(obstacle_mask)
		neighbor_zp = np.zeros_like(obstacle_mask)
		
		neighbor_xm[1:, :, :] = ~obstacle_mask[:-1, :, :]
		neighbor_xp[:-1, :, :] = ~obstacle_mask[1:, :, :]
		neighbor_ym[:, 1:, :] = ~obstacle_mask[:, :-1, :]
		neighbor_yp[:, :-1, :] = ~obstacle_mask[:, 1:, :]
		neighbor_zm[:, :, 1:] = ~obstacle_mask[:, :, :-1]
		neighbor_zp[:, :, :-1] = ~obstacle_mask[:, :, 1:]
		
		obstacle_boundary = obstacle_mask & (
			neighbor_xm | neighbor_xp | neighbor_ym | neighbor_yp | neighbor_zm | neighbor_zp
		)
	
	dirichlet_indices = np.where(dirichlet_mask.ravel())[0]
	
	laplacian = laplacian.tocsr()
	
	for idx in dirichlet_indices:
		laplacian.data[laplacian.indptr[idx]:laplacian.indptr[idx+1]] = 0
	
	laplacian[dirichlet_indices, dirichlet_indices] = 1.0
	
	laplacian_csc = laplacian.tocsc()
	for idx in dirichlet_indices:
		col_start = laplacian_csc.indptr[idx]
		col_end = laplacian_csc.indptr[idx+1]
		diag_pos = np.where(laplacian_csc.indices[col_start:col_end] == idx)[0]
		if len(diag_pos) > 0:
			diag_pos = col_start + diag_pos[0]
			laplacian_csc.data[col_start:col_end] = 0
			laplacian_csc.data[diag_pos] = 1.0
	
	laplacian = laplacian_csc.tocsr()
	
	rhs[dirichlet_indices] = 0.0
	
	if obstacle_mask is not None and obstacle_boundary.any():
		dx, dy, dz = 1.0, 1.0, 1.0
		
		neumann_boundary_mask = obstacle_boundary & (~dirichlet_mask)
		neumann_boundary_indices = np.where(neumann_boundary_mask.ravel())[0]
		
		for idx in neumann_boundary_indices:
			laplacian.data[laplacian.indptr[idx]:laplacian.indptr[idx+1]] = 0
		
		laplacian[neumann_boundary_indices, neumann_boundary_indices] = 1.0
		
		laplacian_csc = laplacian.tocsc()
		for idx in neumann_boundary_indices:
			col_start = laplacian_csc.indptr[idx]
			col_end = laplacian_csc.indptr[idx+1]
			diag_pos = np.where(laplacian_csc.indices[col_start:col_end] == idx)[0]
			if len(diag_pos) > 0:
				diag_pos = col_start + diag_pos[0]
				laplacian_csc.data[col_start:col_end] = 0
				laplacian_csc.data[diag_pos] = 1.0
		
		laplacian = laplacian_csc.tocsr()
		
		rhs[neumann_boundary_indices] = 0.0
		
		fluid_mask = ~obstacle_mask & ~obstacle_boundary
		fluid_indices_3d = np.where(fluid_mask)
		fluid_indices_flat = np.ravel_multi_index(fluid_indices_3d, grid_shape)
		
		modify_indices = []
		modify_coeffs = []
		modify_rhs_additions = []
		
		neighbor_xm_mask = np.zeros_like(obstacle_boundary, dtype=bool)
		neighbor_xp_mask = np.zeros_like(obstacle_boundary, dtype=bool)
		neighbor_xm_mask[1:, :, :] = obstacle_boundary[:-1, :, :]
		neighbor_xp_mask[:-1, :, :] = obstacle_boundary[1:, :, :]
		
		xm_modify_mask = fluid_mask & neighbor_xm_mask
		xp_modify_mask = fluid_mask & neighbor_xp_mask
		
		xm_modify_indices = np.where(xm_modify_mask.ravel())[0]
		for idx_flat in xm_modify_indices:
			i, j, k = np.unravel_index(idx_flat, grid_shape)
			bidx_flat = np.ravel_multi_index((i-1, j, k), grid_shape)
			bidx_k, bidx_j, bidx_i = k, j, i-1
			rhs_vec = rhs_vector_field[bidx_k, bidx_j, bidx_i, :]
			n_dot_rhs = -rhs_vec[0]  # nx = -1
			coeff = 1.0 / (dx * dx)
			modify_indices.append(idx_flat)
			modify_coeffs.append(coeff)
			modify_rhs_additions.append(-coeff * dx * n_dot_rhs)
		
		xp_modify_indices = np.where(xp_modify_mask.ravel())[0]
		for idx_flat in xp_modify_indices:
			i, j, k = np.unravel_index(idx_flat, grid_shape)
			bidx_flat = np.ravel_multi_index((i+1, j, k), grid_shape)
			bidx_k, bidx_j, bidx_i = k, j, i+1
			rhs_vec = rhs_vector_field[bidx_k, bidx_j, bidx_i, :]
			n_dot_rhs = rhs_vec[0]  # nx = 1
			coeff = 1.0 / (dx * dx)
			modify_indices.append(idx_flat)
			modify_coeffs.append(coeff)
			modify_rhs_additions.append(-coeff * dx * n_dot_rhs)
		
		neighbor_ym_mask = np.zeros_like(obstacle_boundary, dtype=bool)
		neighbor_yp_mask = np.zeros_like(obstacle_boundary, dtype=bool)
		neighbor_ym_mask[:, 1:, :] = obstacle_boundary[:, :-1, :]
		neighbor_yp_mask[:, :-1, :] = obstacle_boundary[:, 1:, :]
		
		ym_modify_mask = fluid_mask & neighbor_ym_mask
		yp_modify_mask = fluid_mask & neighbor_yp_mask
		
		ym_modify_indices = np.where(ym_modify_mask.ravel())[0]
		for idx_flat in ym_modify_indices:
			i, j, k = np.unravel_index(idx_flat, grid_shape)
			bidx_k, bidx_j, bidx_i = k, j-1, i
			rhs_vec = rhs_vector_field[bidx_k, bidx_j, bidx_i, :]
			n_dot_rhs = -rhs_vec[1]  # ny = -1
			coeff = 1.0 / (dy * dy)
			modify_indices.append(idx_flat)
			modify_coeffs.append(coeff)
			modify_rhs_additions.append(-coeff * dy * n_dot_rhs)
		
		yp_modify_indices = np.where(yp_modify_mask.ravel())[0]
		for idx_flat in yp_modify_indices:
			i, j, k = np.unravel_index(idx_flat, grid_shape)
			bidx_k, bidx_j, bidx_i = k, j+1, i
			rhs_vec = rhs_vector_field[bidx_k, bidx_j, bidx_i, :]
			n_dot_rhs = rhs_vec[1]  # ny = 1
			coeff = 1.0 / (dy * dy)
			modify_indices.append(idx_flat)
			modify_coeffs.append(coeff)
			modify_rhs_additions.append(-coeff * dy * n_dot_rhs)
		
		neighbor_zm_mask = np.zeros_like(obstacle_boundary, dtype=bool)
		neighbor_zp_mask = np.zeros_like(obstacle_boundary, dtype=bool)
		neighbor_zm_mask[:, :, 1:] = obstacle_boundary[:, :, :-1]
		neighbor_zp_mask[:, :, :-1] = obstacle_boundary[:, :, 1:]
		
		zm_modify_mask = fluid_mask & neighbor_zm_mask
		zp_modify_mask = fluid_mask & neighbor_zp_mask
		
		zm_modify_indices = np.where(zm_modify_mask.ravel())[0]
		for idx_flat in zm_modify_indices:
			i, j, k = np.unravel_index(idx_flat, grid_shape)
			bidx_k, bidx_j, bidx_i = k-1, j, i
			rhs_vec = rhs_vector_field[bidx_k, bidx_j, bidx_i, :]
			n_dot_rhs = -rhs_vec[2]  # nz = -1
			coeff = 1.0 / (dz * dz)
			modify_indices.append(idx_flat)
			modify_coeffs.append(coeff)
			modify_rhs_additions.append(-coeff * dz * n_dot_rhs)
		
		zp_modify_indices = np.where(zp_modify_mask.ravel())[0]
		for idx_flat in zp_modify_indices:
			i, j, k = np.unravel_index(idx_flat, grid_shape)
			bidx_k, bidx_j, bidx_i = k+1, j, i
			rhs_vec = rhs_vector_field[bidx_k, bidx_j, bidx_i, :]
			n_dot_rhs = rhs_vec[2]  # nz = 1
			coeff = 1.0 / (dz * dz)
			modify_indices.append(idx_flat)
			modify_coeffs.append(coeff)
			modify_rhs_additions.append(-coeff * dz * n_dot_rhs)
		
		if len(modify_indices) > 0:
			modify_indices = np.array(modify_indices)
			modify_coeffs = np.array(modify_coeffs)
			modify_rhs_additions = np.array(modify_rhs_additions)
			
			unique_indices, inverse_indices = np.unique(modify_indices, return_inverse=True)
			aggregated_coeffs = np.bincount(inverse_indices, weights=modify_coeffs)
			aggregated_rhs_additions = np.bincount(inverse_indices, weights=modify_rhs_additions)
			
			for i, idx in enumerate(unique_indices):
				current_diag = laplacian[idx, idx]
				laplacian[idx, idx] = current_diag - aggregated_coeffs[i]
			
			rhs[unique_indices] -= aggregated_rhs_additions
	
	laplacian = laplacian.tocsc()
	
	return laplacian, rhs


def save_pressure_to_vtk(data, grid_shape, lengths, filename):
	"""
	3Dpressuresave asVTK Image Data (.vti)，for ParaView
	originalpressure
	
	Args:
		data: 3D numpy，shape (Nx, Ny, Nz)
		grid_shape: shape (Nx, Ny, Nz)
		lengths:  [Lx, Ly, Lz]
		filename: output filename (.vti)
	"""
	data = np.ascontiguousarray(data, dtype=np.float32)
	
	data_abs = np.abs(data)
	
	Nx, Ny, Nz = grid_shape
	
	origin = [0.0, 0.0, 0.0]
	spacing = [lengths[0] / max(Nx - 1, 1), lengths[1] / max(Ny - 1, 1), lengths[2] / max(Nz - 1, 1)]
	
	with open(filename, 'w', encoding='utf-8') as f:
		f.write('<?xml version="1.0"?>\n')
		f.write('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
		f.write(f'  <ImageData WholeExtent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}" ')
		f.write(f'Origin="{origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}" ')
		f.write(f'Spacing="{spacing[0]:.6f} {spacing[1]:.6f} {spacing[2]:.6f}">\n')
		f.write(f'    <Piece Extent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}">\n')
		f.write(f'      <PointData Scalars="pressure">\n')
		
		f.write(f'        <DataArray type="Float32" Name="pressure" format="ascii" NumberOfComponents="1">\n')
		for k in range(Nz):
			for j in range(Ny):
				for i in range(Nx):
					f.write(f'          {data[i, j, k]:.10e}\n')
		f.write('        </DataArray>\n')
		
		f.write(f'        <DataArray type="Float32" Name="pressure_abs" format="ascii" NumberOfComponents="1">\n')
		for k in range(Nz):
			for j in range(Ny):
				for i in range(Nx):
					f.write(f'          {data_abs[i, j, k]:.10e}\n')
		f.write('        </DataArray>\n')
		
		f.write('      </PointData>\n')
		f.write('    </Piece>\n')
		f.write('  </ImageData>\n')
		f.write('</VTKFile>\n')


def save_velocity_field_to_vtk(data, grid_shape, lengths, filename):
	"""
	 3D velocity fieldsave as VTK Image Data (.vti) ， ParaView 。
	data: (Nx, Ny, Nz, 3) numpy，eachgrid points [vx, vy, vz]
	grid_shape: (Nx, Ny, Nz)
	lengths: [Lx, Ly, Lz]
	filename: output .vti path
	"""
	data = np.ascontiguousarray(data, dtype=np.float32)
	Nx, Ny, Nz = grid_shape
	origin = [0.0, 0.0, 0.0]
	spacing = [
		lengths[0] / max(Nx - 1, 1),
		lengths[1] / max(Ny - 1, 1),
		lengths[2] / max(Nz - 1, 1)
	]
	with open(filename, 'w', encoding='utf-8') as f:
		f.write('<?xml version="1.0"?>\n')
		f.write('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
		f.write(f'  <ImageData WholeExtent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}" ')
		f.write(f'Origin="{origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}" ')
		f.write(f'Spacing="{spacing[0]:.6f} {spacing[1]:.6f} {spacing[2]:.6f}">\n')
		f.write(f'    <Piece Extent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}">\n')
		f.write('      <PointData Vectors="velocity">\n')
		f.write('        <DataArray type="Float32" Name="velocity" format="ascii" NumberOfComponents="3">\n')
		for k in range(Nz):
			for j in range(Ny):
				for i in range(Nx):
					f.write(f'          {data[i, j, k, 0]:.6e} {data[i, j, k, 1]:.6e} {data[i, j, k, 2]:.6e}\n')
		f.write('        </DataArray>\n')
		f.write('      </PointData>\n')
		f.write('    </Piece>\n')
		f.write('  </ImageData>\n')
		f.write('</VTKFile>\n')
	print(f"Saved velocity field to VTK: {filename}")


def load_and_visualize_vel(args, savedir, load_path, save_velocity_vtk=False):
	"""
	frameDFRBFvisualizationvelocity field
	save_velocity_vtk: if True，per-framevelocity fieldsave as ParaView  .vti  savedir/velocity_vtk/
	"""
	device = set_device(args)

	with open(os.path.join(args.datadir, 'info.json'), 'r') as fp:
		# read render settings
		meta = json.load(fp)

		# read scene data
		voxel_tran = np.float32(meta['voxel_matrix'])
		voxel_tran = np.stack([voxel_tran[:,2],voxel_tran[:,1],voxel_tran[:,0],voxel_tran[:,3]],axis=1) # swap_zx
		voxel_scale = np.broadcast_to(meta['voxel_scale'],[3])
		
		## apply manual scaling
		scene_scale = args.scene_scale
		voxel_scale = voxel_scale.copy() * scene_scale
		voxel_tran[:3,3] *= scene_scale
		train_video = meta['train_videos'][0]
		delta_t = 1.0/train_video['frame_num']
		t_info = np.float32([0.0, 1.0, 0.5, delta_t])

	t_list = np.arange(t_info[0], t_info[1], t_info[-1])
	frame_num = t_list.shape[0]
	eval_frame_limit = getattr(args, 'eval_frame_limit', None)
	visualize_frame_num = frame_num
	if eval_frame_limit is not None:
		visualize_frame_num = min(frame_num, max(1, eval_frame_limit + 1))

	# lengths = [1.0, args.Ny/ args.Nx, args.Nz/ args.Nx]
	lengths = [args.Nx, args.Ny, args.Nz]
	lengths = np.array(lengths)
	init_centers, h = generate_kernels(lengths, args.kernel_num)
	
	# model = TiDFRBF(WendlandC4(), init_centers, h).to(device)
	
	batchsize = args.batchsize
	imgs = []
	vor_imgs = []
	gt_imgs = []
	gt_vor_imgs = []
	vel_diff_imgs = []
	vor_diff_imgs = []
	mask_imgs = []
	mses = []
	cosine_similarities = []
	vor_mses = []
	divs = []
	gt_divs = []
	
	print(f"\n=== Loading and visualizing {visualize_frame_num-1} velocity frame(s) ===")
	
	has_gt = (getattr(args, 'gt_prefix', None) is not None and 
	          getattr(args, 'gt_ext', None) is not None and 
	          getattr(args, 'gt_prefix_vel', None) is not None and 
	          getattr(args, 'gt_ext_vel', None) is not None)
	if has_gt:
		first_density_gt = f"{args.gt_prefix}0000{args.gt_ext}"
		first_velocity_gt = f"{args.gt_prefix_vel}0000{args.gt_ext_vel}"
		if not (os.path.exists(first_density_gt) and os.path.exists(first_velocity_gt)):
			print(f"GT paths are configured but files are missing; skipping GT comparison: {first_density_gt}, {first_velocity_gt}")
			has_gt = False
	
	if has_gt:
		print("GT data detected; computing GT metrics")
	else:
		print("GT data not found，SkippingGT")
	
	if save_velocity_vtk:
		velocity_vtk_dir = os.path.join(savedir, "velocity_vtk")
		os.makedirs(velocity_vtk_dir, exist_ok=True)
		print(f"per-framevelocity fieldsave as .vti : {velocity_vtk_dir}")
	
	frame_velocities_dir = os.path.join(load_path, "frame_velocities")
	is_sliding_window = os.path.exists(frame_velocities_dir)
	
	window_size = None
	if is_sliding_window:
		print(f"Detected sliding-window training output; loading velocity models from {frame_velocities_dir}")
		window_pattern = os.path.join(load_path, "window_*_*")
		window_dirs = glob.glob(window_pattern)
		if window_dirs:
			first_window_name = os.path.basename(window_dirs[0])
			match = re.match(r'window_(\d+)_(\d+)', first_window_name)
			if match:
				w_start = int(match.group(1))
				w_end = int(match.group(2))
				window_size = w_end - w_start
				print(f"Inferred window size: {window_size}")
	else:
		print(f"Using legacy training-output layout， {load_path}/ckpt Loading velocity models")
	
	for frame_idx in tqdm(range(visualize_frame_num-1), desc="Loading and visualizing frames"):
		print(f"\n=== Loading velocity field for frame {frame_idx} ===")
		
		if is_sliding_window:
			vel_model_path = os.path.join(frame_velocities_dir, f"frame_{frame_idx:03d}_velocity.pth")
			
			if not os.path.exists(vel_model_path):
				found_model = False
				
				if window_size is not None and frame_idx + 1 < frame_num:
					next_window_dir = os.path.join(load_path, f"window_{frame_idx+1}_{frame_idx+1+window_size}")
					if os.path.exists(next_window_dir):
						vel_pattern = os.path.join(next_window_dir, "ckpt", f"velrbf_frame_{frame_idx:03d}_ckpt_*.pth")
						vel_files = glob.glob(vel_pattern)
						if vel_files:
							vel_model_path = max(vel_files, key=os.path.getctime)
							found_model = True
							print(f"Loading from the next-window checkpoint（may be optimized）: {vel_model_path}")
				
				if not found_model and window_size is not None:
					current_window_dir = os.path.join(load_path, f"window_{frame_idx}_{frame_idx+window_size}")
					if os.path.exists(current_window_dir):
						vel_pattern = os.path.join(current_window_dir, "ckpt", f"velrbf_frame_{frame_idx:03d}_ckpt_*.pth")
						vel_files = glob.glob(vel_pattern)
						if vel_files:
							vel_model_path = max(vel_files, key=os.path.getctime)
							found_model = True
							print(f"Loading from the current-window checkpoint（original）: {vel_model_path}")
				
				if not found_model:
					window_pattern = os.path.join(load_path, "window_*_*")
					all_window_dirs = glob.glob(window_pattern)
					for window_dir in all_window_dirs:
						vel_pattern = os.path.join(window_dir, "ckpt", f"velrbf_frame_{frame_idx:03d}_ckpt_*.pth")
						vel_files = glob.glob(vel_pattern)
						if vel_files:
							vel_model_path = max(vel_files, key=os.path.getctime)
							found_model = True
							print(f"Loading from window {os.path.basename(window_dir)}  checkpoint : {vel_model_path}")
							break
				
				if not found_model:
					print(f"Warning: not foundframe {frame_idx} velocity model，Skippingframe")
					continue
			else:
				print(f"Loading from frame_velocities directory: {vel_model_path}")
			
			model = TiDFRBF.load(vel_model_path, device)
			model.eval()
		else:
			ckpt_dir = os.path.join(load_path, "ckpt")
			pattern = f"velrbf_frame_{frame_idx:03d}_ckpt_*.pth"
			matching_files = glob.glob(os.path.join(ckpt_dir, pattern))
			
			if not matching_files:
				print(f"Warning: not foundframe {frame_idx} frame checkpoint file，Skippingframe")
				continue
			
			ckpt_numbers = []
			for file_path in matching_files:
				match = re.search(r'_ckpt_(\d+)\.pth$', file_path)
				if match:
					ckpt_numbers.append(int(match.group(1)))
			
			if not ckpt_numbers:
				print(f"Warning: filein checkpoint number，Skippingframe {frame_idx} frame")
				continue
			
			max_ckpt_num = max(ckpt_numbers)
			frame_load_path = os.path.join(ckpt_dir, f"velrbf_frame_{frame_idx:03d}_ckpt_{max_ckpt_num:06d}.pth")
			print(f"Using checkpoint: {max_ckpt_num:06d}")
			
			model = TiDFRBF.load(frame_load_path, device)
			model.eval()
		
		grid_shape = (args.Nx, args.Ny, args.Nz)
		min_corner = np.zeros(3)
		max_corner = lengths
		points_np = utils_grid.generate_gridpoints(grid_shape, min_corner, max_corner)
		points = torch.from_numpy(points_np).float().to(device)
		
		inflow_height_ratio = 0.0
		inflow_velocity_param = None
		if not is_sliding_window:
			if args.load_path is not None:
				inflow_velocity_path = os.path.join(args.load_path, "ckpt", f"inflow_velocity_{frame_idx:03d}_{max_ckpt_num:06d}.npz")
				if os.path.exists(inflow_velocity_path):
					inflow_velocity_data = np.load(inflow_velocity_path)
					inflow_velocity_param = torch.from_numpy(inflow_velocity_data['velocity']).float().to(device)
					print(f"Loadedinflow: {inflow_velocity_path}, shape: {inflow_velocity_param.shape}")
				else:
					print(f"not foundinflowfile: {inflow_velocity_path}，using model-predicted velocity")
			else:
				default_ckpt_dir = os.path.dirname(frame_load_path)
				inflow_velocity_path = os.path.join(default_ckpt_dir, f"inflow_velocity_{frame_idx:03d}_{args.ckpt_num:06d}.npz")
				if os.path.exists(inflow_velocity_path):
					inflow_velocity_data = np.load(inflow_velocity_path)
					inflow_velocity_param = torch.from_numpy(inflow_velocity_data['velocity']).float().to(device)
					print(f"Loadedinflow: {inflow_velocity_path}, shape: {inflow_velocity_param.shape}")
		
		if inflow_velocity_param is not None:
			train_scale = getattr(args, 'scale', 1)
			if train_scale == 1 and inflow_velocity_param.shape[:3] != grid_shape:
				loaded_shape = inflow_velocity_param.shape[:3]
				if all(loaded_shape[i] <= grid_shape[i] for i in range(3)):
					scale_x = grid_shape[0] / loaded_shape[0] if loaded_shape[0] > 0 else 1
					scale_y = grid_shape[1] / loaded_shape[1] if loaded_shape[1] > 0 else 1
					scale_z = grid_shape[2] / loaded_shape[2] if loaded_shape[2] > 0 else 1
					if abs(scale_x - scale_y) < 0.1 and abs(scale_y - scale_z) < 0.1:
						train_scale = (scale_x + scale_y + scale_z) / 3
					else:
						train_scale = max(scale_x, scale_y, scale_z)
					print(f"shapescale: {train_scale:.2f} (shape: {loaded_shape}, shape: {grid_shape})")
			
			if inflow_velocity_param.shape[:3] != grid_shape:
				print(f"inflow {inflow_velocity_param.shape[:3]} interpolation {grid_shape}")
				
				inflow_velocity_5d = inflow_velocity_param.permute(3, 0, 1, 2).unsqueeze(0)
				inflow_velocity_5d = F.interpolate(inflow_velocity_5d, size=grid_shape, mode='trilinear', align_corners=False)
				inflow_velocity_param = inflow_velocity_5d.squeeze(0).permute(1, 2, 3, 0)
				print(f"interpolationshape: {inflow_velocity_param.shape}")
		
		with torch.no_grad():
			vel_pred_full = torch.zeros((len(points), 3), device=device)
			vor_pred_full = torch.zeros((len(points), 3), device=device)
			
			for v_idx in range(0, len(points), batchsize):
				batch_points = points[v_idx:min(len(points), v_idx + batchsize)]
				batch_vel_pred = model(batch_points)
				batch_vor_pred = model.vorticity(batch_points)
				vel_pred_full[v_idx:min(len(points), v_idx + batchsize)] = batch_vel_pred
				vor_pred_full[v_idx:min(len(points), v_idx + batchsize)] = batch_vor_pred
			
			if inflow_velocity_param is not None:
				pred_velocity = vel_pred_full.reshape(grid_shape + (3,))
				pred_velocity = pred_velocity.permute(2, 1, 0, 3)
				
				inflow_velocity_mask = create_inflow_mask_torch(grid_shape, inflow_height_ratio, device, pred_velocity.dtype)
				inflow_velocity_mask = inflow_velocity_mask.permute(2, 1, 0)  # (Nx, Ny, Nz) -> (Nz, Ny, Nx)
				inflow_velocity_mask_3d = inflow_velocity_mask.unsqueeze(-1).expand_as(pred_velocity)
				inflow_velocity_param_permuted = inflow_velocity_param.permute(2, 1, 0, 3)  # (Nx, Ny, Nz, 3) -> (Nz, Ny, Nx, 3)
				pred_velocity = torch.where(inflow_velocity_mask_3d, inflow_velocity_param_permuted, pred_velocity)
				
				pred_velocity = pred_velocity.permute(2, 1, 0, 3)
				vel_pred_full = pred_velocity.reshape(len(points), 3)
	
			if has_gt:
				gt_dens_path = f"{args.gt_prefix}{frame_idx:04d}{args.gt_ext}"
				dens_gt_np = np.load(gt_dens_path)
				dens_gt_np = dens_gt_np["arr_0"]
				dens_gt_np = dens_gt_np[:args.Nz, :args.Ny, :args.Nx]
				if dens_gt_np.shape[0] == 1:
					dens_gt_np = dens_gt_np.squeeze(axis=0)
				dens_mask = dens_gt_np > 0
				# gt_path = f"{args.gt_prefix_vel}{frame_idx:04d}{args.gt_ext_vel}"
				# im_gt_np = np.load(gt_path)
				# im_gt_np = im_gt_np["arr_0"]
				# im_gt_np = im_gt_np[:args.Nz, :args.Ny, :args.Nx, :3]
				# im_gt_np_norm = np.linalg.norm(im_gt_np, axis=-1, keepdims=True)
				# dens_mask = (im_gt_np_norm > 0.5)
				mask_image = den_scalar2rgb(dens_mask, scale=255, is3D=True, logv=False)
				mask_output_path = f"{savedir}/images/vel/mask_frame_{frame_idx:06d}.png"
				imageio.imwrite(mask_output_path, mask_image)
				print(f"Frame {frame_idx} mask visualization saved to: {mask_output_path}")
				mask_imgs.append(mask_image)
				dens_mask = dens_mask.ravel()
			else:
				dens_mask = np.ones(len(points), dtype=bool)
			
			im_estim = vel_pred_full.cpu().numpy()
			im_estim = np.reshape(im_estim, grid_shape + (3,))  # (x, y, z, 3)
			
			if save_velocity_vtk:
				vtk_path = os.path.join(velocity_vtk_dir, f"frame_{frame_idx:03d}_velocity.vti")
				save_velocity_field_to_vtk(im_estim, grid_shape, lengths, vtk_path)
			
			im_estim = np.swapaxes(im_estim, 0, 2)  # (z, y, x, 3)
   
			div = div_np(im_estim)
			cur_div = np.mean(np.square(div.ravel()[dens_mask]))
			divs.append(cur_div)
			
			estim_image = vel_uv2hsv(im_estim, scale=300, is3D=True, logv=False)
			
			# try:
			# 	with torch.enable_grad():
			# 		autodiff_batch = min(batchsize, 32768)
			# 		div_autodiff = compute_divergence_autodiff(model, points, batchsize=autodiff_batch)  # [N]
			# 		div_autodiff_np = div_autodiff.detach().cpu().numpy().reshape(grid_shape)
			# 		cur_div_autodiff = np.mean(np.square(div_autodiff_np.ravel()[dens_mask]))
			# 		print(f"[AutoDiff] div mse(masked): {cur_div_autodiff:.6f} | [FiniteDiff] {cur_div:.6f}")
			# except Exception as _:
			
			output_path = f"{savedir}/images/vel/vel_frame_{frame_idx:06d}_reconstructed.png"
			imageio.imwrite(output_path, estim_image)
			print(f"Frame {frame_idx} reconstructed velocity field saved to: {output_path}")
			
			imgs.append(estim_image)

			_, NETw_estim = jacobian3D_np(im_estim)
			# im_estim_vor = vor_pred_full.cpu().numpy()
			im_estim_vor = np.reshape(NETw_estim[0], grid_shape + (3,))
			# im_estim_vor = np.swapaxes(im_estim_vor, 0, 2)
			
			estim_image_vor = vel_uv2hsv(im_estim_vor, scale=1500, is3D=True, logv=False)
			output_path_vor = f"{savedir}/images/vel/vor_frame_{frame_idx:06d}_reconstructed.png"
			imageio.imwrite(output_path_vor, estim_image_vor)
			print(f"Frame {frame_idx} reconstructed vorticity field saved to: {output_path_vor}")
			vor_imgs.append(estim_image_vor)
   
			if has_gt:
				gt_path = f"{args.gt_prefix_vel}{frame_idx:04d}{args.gt_ext_vel}"
				im_gt_np = np.load(gt_path)
				im_gt_np = im_gt_np["arr_0"]
				im_gt_np = im_gt_np[:args.Nz, :args.Ny, :args.Nx]
				gt_image = vel_uv2hsv(im_gt_np, scale=300, is3D=True, logv=False)
				imageio.imwrite(f"{savedir}/images/vel/vel_frame_{frame_idx:06d}_gt.png", gt_image)
				gt_imgs.append(gt_image)

				try:
					div_gt = div_np(im_gt_np)
					cur_div_gt = np.mean(np.square(div_gt.ravel()[dens_mask]))
					print(f"[GT FiniteDiff] div mse(masked): {cur_div_gt:.6f}")
					gt_divs.append(cur_div_gt)
				except Exception as _:
					print("[GT FiniteDiff] Compute divergence，SkippingframeGT。")
   
				vel_error = im_estim - im_gt_np
				cur_mse = np.mean(np.square(vel_error.reshape(-1, 3)[dens_mask]))
				mses.append(cur_mse)
				
				vel_estim_flat = im_estim.reshape(-1, 3)[dens_mask]  # (N_mask, 3)
				vel_gt_flat = im_gt_np.reshape(-1, 3)[dens_mask]  # (N_mask, 3)
				
				# cosine_similarity = dot(a, b) / (norm(a) * norm(b))
				dot_products = np.sum(vel_estim_flat * vel_gt_flat, axis=1)  # (N_mask,)
				norms_estim = np.linalg.norm(vel_estim_flat, axis=1)  # (N_mask,)
				norms_gt = np.linalg.norm(vel_gt_flat, axis=1)  # (N_mask,)
				
				valid_mask = (norms_estim > 1e-10) & (norms_gt > 1e-10)
				if np.any(valid_mask):
					cosine_sim_per_point = dot_products[valid_mask] / (norms_estim[valid_mask] * norms_gt[valid_mask])
					cur_cosine_sim = np.mean(cosine_sim_per_point)
				else:
					cur_cosine_sim = float('nan')
				
				cosine_similarities.append(cur_cosine_sim)
				try:
					# vel_error_masked = vel_error.reshape(-1, 3)
					# vel_error_masked[~dens_mask] = 0
					# vel_error_masked = vel_error_masked.reshape(vel_error.shape)
					# vel_diff_image = vel_uv2hsv(vel_error_masked, scale=300, is3D=True, logv=False)
					vel_diff_image = vel_uv2hsv(vel_error, scale=300, is3D=True, logv=False)
					imageio.imwrite(f"{savedir}/images/vel/vel_frame_{frame_idx:06d}_diff.png", vel_diff_image)
					vel_diff_imgs.append(vel_diff_image)
				except Exception as _:
					print("[Warn] failed to visualize velocity difference。")
				
				_, NETw_gt = jacobian3D_np(im_gt_np)
				gt_vor_image = vel_uv2hsv(
					NETw_gt[0], scale=1500, is3D=True, logv=False
				)
				imageio.imwrite(f"{savedir}/images/vel/vor_frame_{frame_idx:06d}_gt.png", gt_vor_image)
				gt_vor_imgs.append(gt_vor_image)
				
				vor_error = im_estim_vor - NETw_gt[0]
				cur_vor_mse = np.mean(np.square(vor_error.reshape(-1, 3)[dens_mask]))
				vor_mses.append(cur_vor_mse)
				try:
					vor_diff_image = vel_uv2hsv(vor_error, scale=1500, is3D=True, logv=False)
					imageio.imwrite(f"{savedir}/images/vel/vor_frame_{frame_idx:06d}_diff.png", vor_diff_image)
					vor_diff_imgs.append(vor_diff_image)
				except Exception as _:
					print("[Warn] vorticityvisualization。")
				
				print(f"[{frame_idx}/{frame_num}]: velocity MSE: {cur_mse}, cosine similarity: {cur_cosine_sim:.6f}, vorticity MSE: {cur_vor_mse}, div: {cur_div}")
			else:
				print(f"[{frame_idx}/{frame_num}]: div: {cur_div}")
	
	if imgs:
		video = np.stack(imgs, axis=0)
		video_path = f"{savedir}/video/vel/vel_reconstructed_video.mp4"
		imageio.mimwrite(video_path, video, fps=25, quality=8)
		print(f"\n=== Reconstruction complete ===")
		print(f"Reconstructed images saved in: {savedir}/images/vel/")
		print(f"Reconstructed video saved to: {video_path}")
	else:
		print("Warning: Loadedframe")

	if vor_imgs:
		video_vor = np.stack(vor_imgs, axis=0)
		video_path_vor = f"{savedir}/video/vel/vor_reconstructed_video.mp4"
		imageio.mimwrite(video_path_vor, video_vor, fps=25, quality=8)
		print(f"Reconstructed vorticity video saved to: {video_path_vor}")
  
	if gt_imgs:
		video_gt = np.stack(gt_imgs, axis=0)
		video_path_gt = f"{savedir}/video/vel/vel_gt_video.mp4"
		imageio.mimwrite(video_path_gt, video_gt, fps=25, quality=8)
		print(f"GT velocity video saved to: {video_path_gt}")
  
	if gt_vor_imgs:
		video_gt_vor = np.stack(gt_vor_imgs, axis=0)
		video_path_gt_vor = f"{savedir}/video/vel/vor_gt_video.mp4"
		imageio.mimwrite(video_path_gt_vor, video_gt_vor, fps=25, quality=8)
		print(f"GT vorticity video saved to: {video_path_gt_vor}")

	if vel_diff_imgs:
		video_vel_diff = np.stack(vel_diff_imgs, axis=0)
		video_path_vel_diff = f"{savedir}/video/vel/vel_diff_video.mp4"
		imageio.mimwrite(video_path_vel_diff, video_vel_diff, fps=25, quality=8)
		print(f"Velocity error video saved to: {video_path_vel_diff}")

	if vor_diff_imgs:
		video_vor_diff = np.stack(vor_diff_imgs, axis=0)
		video_path_vor_diff = f"{savedir}/video/vel/vor_diff_video.mp4"
		imageio.mimwrite(video_path_vor_diff, video_vor_diff, fps=25, quality=8)
		print(f"Vorticity error video saved to: {video_path_vor_diff}")
	
	if mask_imgs:
		video_mask = np.stack(mask_imgs, axis=0)
		video_path_mask = f"{savedir}/video/vel/mask_video.mp4"
		imageio.mimwrite(video_path_mask, video_mask, fps=25, quality=8)
		print(f"Mask visualization video saved to: {video_path_mask}")
  
	if divs:
		mean_div = np.mean(divs)
		if has_gt:
			mean_mse = np.mean(mses) if mses else float('nan')
			mean_cosine_sim = np.nanmean(cosine_similarities) if cosine_similarities else float('nan')
			mean_vor_mse = np.mean(vor_mses) if vor_mses else float('nan')
			mean_gt_div = np.mean(gt_divs) if gt_divs else float('nan')
			print(f"mean div: {mean_div:.6f}, mean vel mse: {mean_mse:.6f}, mean cosine similarity: {mean_cosine_sim:.6f}, mean vor mse: {mean_vor_mse:.6f}, mean gt div: {mean_gt_div:.6f}")
		else:
			print(f"mean div: {mean_div:.6f} (without GT data，SkippingMSE)")
		
		metrics_path = f"{savedir}/metrics.txt"
		file_exists = os.path.exists(metrics_path)
		mode = "a" if file_exists else "w"
		
		with open(metrics_path, mode) as f:
			if file_exists:
				f.write("\n" + "="*50 + "\n")
			f.write("=== Velocity Field Evaluation Metrics ===\n")
			if has_gt:
				f.write(f"mean div: {mean_div:.6f}, mean vel mse: {mean_mse:.6f}, mean cosine similarity: {mean_cosine_sim:.6f}, mean vor mse: {mean_vor_mse:.6f}, mean gt div: {mean_gt_div:.6f}\n")
			else:
				f.write(f"mean div: {mean_div:.6f} (without GT data，SkippingMSE)\n")
		print(f"Metrics saved to: {metrics_path}")


def compute_and_visualize_pressure(args, savedir, load_path, obstacle_mask, use_gt=False, train_scale=1.0):
	"""
	velocity fieldpressurevisualization
	
	Based on inviscid Navier-Stokes without external forces:：
	ρ(∂u/∂t + u·∇u) = -∇p
	
	take divergence，obtain a Poisson equation：
	-∇²p = ∇·(ρ(∂u/∂t + u·∇u))
	
	boundary conditions：
	- obstacle boundary：Neumannboundary conditions ∂p/∂n = -n·(ρ(∂u/∂t + u·∇u))
	- simulation boundary：Dirichletboundary conditions p = 0
	
	Args:
		args: argument object（containsNx, Ny, Nz, batchsize）
		savedir: output directory
		load_path: path
		obstacle_mask: mask (Nx, Ny, Nz)，Trueposition
		use_gt: useGTvelocity field（Falseusevelocity field）
		train_scale: ， downsample train_scale 
	"""
	device = set_device(args)
	
	s = float(train_scale) if train_scale is not None else 1.0
	assert s >= 1.0, "train_scale  >=1 "
	
	grid_shape_full = (args.Nx, args.Ny, args.Nz)
	grid_shape = (max(1, int(round(args.Nx / s))), max(1, int(round(args.Ny / s))), max(1, int(round(args.Nz / s))))
	
	print(f"original: {grid_shape_full}, : {grid_shape} (scale={s})")
	
	if obstacle_mask is not None:
		obstacle_mask = np.asarray(obstacle_mask, dtype=bool)
		if obstacle_mask.shape != grid_shape_full:
			raise ValueError(f"obstacle_maskshape {grid_shape_full}， {obstacle_mask.shape}")
		
		if s > 1.0:
			obstacle_mask_torch = torch.from_numpy(obstacle_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1, 1, Nx, Ny, Nz)
			obstacle_mask_torch = F.avg_pool3d(
				obstacle_mask_torch.permute(0, 1, 4, 3, 2),  # (1, 1, Nz, Ny, Nx)
				kernel_size=int(s),
				stride=int(s)
			).permute(0, 1, 4, 3, 2)
			obstacle_mask = (obstacle_mask_torch.squeeze().numpy() > 0.5).astype(bool)
			obstacle_mask = obstacle_mask[:grid_shape[0], :grid_shape[1], :grid_shape[2]]
	else:
		obstacle_mask = np.zeros(grid_shape, dtype=bool)
		radius = grid_shape[0] / 12
		center = grid_shape[0] * np.array([0.5, 0.4, 0.5])
		xc, yc, zc = center[0], center[1], center[2]
		
		x = np.arange(grid_shape[0])
		y = np.arange(grid_shape[1])
		z = np.arange(grid_shape[2])
		X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
		
		condition = (X - xc)**2 + (Y - yc)**2 + (Z - zc)**2 <= radius**2
		obstacle_mask = condition
	
	with open(os.path.join(args.datadir, 'info.json'), 'r') as fp:
		meta = json.load(fp)
		train_video = meta['train_videos'][0]
		delta_t = 1.0 / train_video['frame_num']
		t_info = np.float32([0.0, 1.0, 0.5, delta_t])
	
	t_list = np.arange(t_info[0], t_info[1], t_info[-1])
	frame_num = t_list.shape[0]
	
	lengths = [args.Nx, args.Ny, args.Nz]
	lengths = np.array(lengths)
	
	batchsize = args.batchsize
	
	os.makedirs(f"{savedir}/images/pressure", exist_ok=True)
	os.makedirs(f"{savedir}/video/pressure", exist_ok=True)
	
	print(f"\n=== startvisualization {frame_num-1} framepressure ===")
	print(f"use{'GT' if use_gt else ''}velocity field")
	print(f": {grid_shape} (original: {grid_shape_full}, scale={s})")
	
	has_gt = (getattr(args, 'gt_prefix_vel', None) is not None and 
	          getattr(args, 'gt_ext_vel', None) is not None)
	
	if use_gt and not has_gt:
		raise ValueError("use_gt=Truenot foundGTvelocity field")
	
	frame_velocities_dir = os.path.join(load_path, "frame_velocities")
	is_sliding_window = os.path.exists(frame_velocities_dir)
	
	window_size = None
	if is_sliding_window:
		window_pattern = os.path.join(load_path, "window_*_*")
		window_dirs = glob.glob(window_pattern)
		if window_dirs:
			first_window_name = os.path.basename(window_dirs[0])
			match = re.match(r'window_(\d+)_(\d+)', first_window_name)
			if match:
				w_start = int(match.group(1))
				w_end = int(match.group(2))
				window_size = w_end - w_start
	
	min_corner = np.zeros(3)
	max_corner = lengths
	Nx, Ny, Nz = grid_shape
	x_coords = np.linspace(min_corner[0], max_corner[0], Nx, endpoint=True)
	y_coords = np.linspace(min_corner[1], max_corner[1], Ny, endpoint=True)
	z_coords = np.linspace(min_corner[2], max_corner[2], Nz, endpoint=True)
	
	points_list = []
	for i in range(Nx):
		for j in range(Ny):
			for k in range(Nz):
				points_list.append([x_coords[i], y_coords[j], z_coords[k]])
	points_np = np.array(points_list)  # (Nx*Ny*Nz, 3)
	points = torch.from_numpy(points_np).float().to(device)
	
	print("Laplacian...")
	laplacian = build_laplacian_3d(grid_shape, dx=1.0, dy=1.0, dz=1.0)
	
	pressure_imgs = []
	pressure_fields = []
	
	for frame_idx in tqdm(range(frame_num - 1), desc="Computing pressure fields"):
		print(f"\n=== frame {frame_idx} framepressure ===")
		
		velocities = []
		for target_frame_idx in [frame_idx, frame_idx + 1]:
			if use_gt:
				gt_path = f"{args.gt_prefix_vel}{target_frame_idx:04d}{args.gt_ext_vel}"
				if not os.path.exists(gt_path):
					print(f"Warning: not foundGTvelocity field {gt_path}，Skippingframe")
					break
				im_gt_np = np.load(gt_path)
				im_gt_np = im_gt_np["arr_0"]
				im_gt_np = im_gt_np[:args.Nz, :args.Ny, :args.Nx]  # (Nz, Ny, Nx, 3)
				
				if s > 1.0:
					im_gt_torch = torch.from_numpy(im_gt_np).permute(3, 2, 1, 0).unsqueeze(0)  # (1, 3, Nx, Ny, Nz)
					im_gt_torch = F.avg_pool3d(
						im_gt_torch,
						kernel_size=int(s),
						stride=int(s)
					)
					im_gt_np = im_gt_torch.squeeze(0).permute(3, 2, 1, 0).numpy()  # (Nz_down, Ny_down, Nx_down, 3)
					im_gt_np = im_gt_np[:grid_shape[2], :grid_shape[1], :grid_shape[0], :]
				
				velocities.append(im_gt_np)  # (Nz_down, Ny_down, Nx_down, 3)
			else:
				if is_sliding_window:
					vel_model_path = os.path.join(frame_velocities_dir, f"frame_{target_frame_idx:03d}_velocity.pth")
					if not os.path.exists(vel_model_path):
						found_model = False
						if window_size is not None and target_frame_idx + 1 < frame_num:
							next_window_dir = os.path.join(load_path, f"window_{target_frame_idx+1}_{target_frame_idx+1+window_size}")
							if os.path.exists(next_window_dir):
								vel_pattern = os.path.join(next_window_dir, "ckpt", f"velrbf_frame_{target_frame_idx:03d}_ckpt_*.pth")
								vel_files = glob.glob(vel_pattern)
								if vel_files:
									vel_model_path = max(vel_files, key=os.path.getctime)
									found_model = True
						
						if not found_model and window_size is not None:
							current_window_dir = os.path.join(load_path, f"window_{target_frame_idx}_{target_frame_idx+window_size}")
							if os.path.exists(current_window_dir):
								vel_pattern = os.path.join(current_window_dir, "ckpt", f"velrbf_frame_{target_frame_idx:03d}_ckpt_*.pth")
								vel_files = glob.glob(vel_pattern)
								if vel_files:
									vel_model_path = max(vel_files, key=os.path.getctime)
									found_model = True
						
						if not found_model:
							print(f"Warning: not foundframe {target_frame_idx} velocity model，Skippingframe")
							break
				else:
					ckpt_dir = os.path.join(load_path, "ckpt")
					pattern = f"velrbf_frame_{target_frame_idx:03d}_ckpt_*.pth"
					matching_files = glob.glob(os.path.join(ckpt_dir, pattern))
					if not matching_files:
						print(f"Warning: not foundframe {target_frame_idx} framecheckpointfile，Skippingframe")
						break
					
					ckpt_numbers = []
					for file_path in matching_files:
						match = re.search(r'_ckpt_(\d+)\.pth$', file_path)
						if match:
							ckpt_numbers.append(int(match.group(1)))
					
					if not ckpt_numbers:
						print(f"Warning: fileincheckpoint number，Skippingframe {target_frame_idx} frame")
						break
					
					max_ckpt_num = max(ckpt_numbers)
					vel_model_path = os.path.join(ckpt_dir, f"velrbf_frame_{target_frame_idx:03d}_ckpt_{max_ckpt_num:06d}.pth")
				
				model = TiDFRBF.load(vel_model_path, device)
				model.eval()
				
				with torch.no_grad():
					vel_pred_full = torch.zeros((len(points), 3), device=device)
					for v_idx in range(0, len(points), batchsize):
						batch_points = points[v_idx:min(len(points), v_idx + batchsize)]
						batch_vel_pred = model(batch_points)
						vel_pred_full[v_idx:min(len(points), v_idx + batchsize)] = batch_vel_pred
					
					vel_np = vel_pred_full.cpu().numpy()
					vel_np = np.reshape(vel_np, grid_shape + (3,))
					vel_np = np.swapaxes(vel_np, 0, 2)  # (Nx, Ny, Nz, 3) -> (Nz, Ny, Nx, 3)
					velocities.append(vel_np)
		
		if len(velocities) < 2:
			print(f"Warning: failed to load frame {frame_idx} or frame {frame_idx + 1}; skipping this frame")
			continue
		
		vel_t = velocities[0]
		vel_tp1 = velocities[1]
		
		du_dt = (vel_tp1 - vel_t) / delta_t  # (Nz, Ny, Nx, 3)
		
		grad_v = compute_velocity_gradient_np(vel_t)  # (Nz, Ny, Nx, 3, 3)
		
		convective = compute_convective_term_np(vel_t, grad_v)  # (Nz, Ny, Nx, 3)
		
		rhs_vector = du_dt + convective  # (Nz, Ny, Nx, 3)
		rhs_scalar = div_np(rhs_vector)
		
		if rhs_scalar.ndim == 4:
			rhs_scalar = rhs_scalar.squeeze(0)  # (Nz, Ny, Nx)
		rhs_scalar = rhs_scalar.transpose(2, 1, 0)  # (Nz, Ny, Nx) -> (Nx, Ny, Nz)
		
		rhs_flat = rhs_scalar.ravel()  # (Nx*Ny*Nz,)
		
		
		# laplacian_coo = laplacian.tocoo()
		# laplacian_output_path = f"{savedir}/images/pressure/laplacian_frame_{frame_idx:06d}.txt"
		# with open(laplacian_output_path, 'w', encoding='utf-8') as f:
		# 	for idx in range(len(laplacian_coo.data)):
		# 		f.write(f"[{laplacian_coo.row[idx]}, {laplacian_coo.col[idx]}] = {laplacian_coo.data[idx]:.10e}\n")
		
		# rhs_output_path = f"{savedir}/images/pressure/rhs_frame_{frame_idx:06d}.txt"
		# with open(rhs_output_path, 'w', encoding='utf-8') as f:
		# 	for idx in range(len(rhs_flat)):
		# 		f.write(f"[{idx}] = {rhs_flat[idx]:.10e}\n")
		
		laplacian_mod, rhs_mod = apply_boundary_conditions(
			laplacian.copy(), rhs_flat.copy(), grid_shape, obstacle_mask, rhs_vector
		)
		
		
		print("\n=== checkLaplacian ===")
		
		laplacian_mod_csr = laplacian_mod.tocsr()
		laplacian_mod_transpose = laplacian_mod_csr.transpose().tocsr()
		diff = laplacian_mod_csr - laplacian_mod_transpose
		max_diff = abs(diff.data).max() if diff.nnz > 0 else 0.0
		is_symmetric = max_diff < 1e-10
		print(f"check: {'' if is_symmetric else ''}, latest: {max_diff:.2e}")
		
		# is_positive_definite = False
		# min_eigenvalue_approx = None
		# try:
		# 	laplacian_mod_dense = laplacian_mod_csr.toarray()
		# 	cholesky(laplacian_mod_dense, check_finite=False)
		# 	is_positive_definite = True
		# except LinAlgError:
		# 	try:
		# 		from scipy.sparse.linalg import eigsh
		# 		eigenvalues, _ = eigsh(laplacian_mod_csr, k=min(5, laplacian_mod_csr.shape[0]-1), which='SA')
		# 		min_eigenvalue_approx = eigenvalues.min()
		# 		if min_eigenvalue_approx > -1e-10:
		# 		else:
		# 	except Exception as e:
		
		# check_output_path = f"{savedir}/images/pressure/matrix_check_frame_{frame_idx:06d}.txt"
		# with open(check_output_path, 'w') as f:
		# 	f.write(f"=" * 50 + "\n\n")
		# 	if min_eigenvalue_approx is not None:
		
		# laplacian_mod_coo = laplacian_mod.tocoo()
		# laplacian_mod_output_path = f"{savedir}/images/pressure/laplacian_mod_frame_{frame_idx:06d}.txt"
		# with open(laplacian_mod_output_path, 'w', encoding='utf-8') as f:
		# 	for idx in range(len(laplacian_mod_coo.data)):
		# 		f.write(f"[{laplacian_mod_coo.row[idx]}, {laplacian_mod_coo.col[idx]}] = {laplacian_mod_coo.data[idx]:.10e}\n")
		
		# rhs_mod_output_path = f"{savedir}/images/pressure/rhs_mod_frame_{frame_idx:06d}.txt"
		# with open(rhs_mod_output_path, 'w', encoding='utf-8') as f:
		# 	for idx in range(len(rhs_mod)):
		# 		f.write(f"[{idx}] = {rhs_mod[idx]:.10e}\n")
		
		print(f"（: {laplacian_mod.shape[0]}x{laplacian_mod.shape[1]}）...")
		try:
			iteration_count = [0]
			initial_residual = None
			
			def callback(xk):
				"""CG，used for"""
				iteration_count[0] += 1
				if CUPY_AVAILABLE:
					residual = float(cp.asnumpy(cp.linalg.norm(laplacian_cupy @ xk - rhs_cupy)))
				else:
					residual = float(np.linalg.norm(laplacian_mod @ xk - rhs_mod))
				
				if iteration_count[0] == 1:
					initial_residual = residual
				
				if iteration_count[0] % 100 == 0:
					print(f"   {iteration_count[0]}:  = {residual:.6e}")
			
			if CUPY_AVAILABLE:
				laplacian_cupy = cusp.csc_matrix(laplacian_mod)
				rhs_cupy = cp.asarray(rhs_mod, dtype=cp.float32)
				
				initial_residual = float(cp.asnumpy(cp.linalg.norm(rhs_cupy)))
				
				try:
					pressure_flat_cupy, info = cupy_cg(
						laplacian_cupy, 
						rhs_cupy, 
						tol=1e-6,
						maxiter=10000,
						callback=callback
					)
				except TypeError:
					pressure_flat_cupy, info = cupy_cg(
						laplacian_cupy, 
						rhs_cupy, 
						tol=1e-6,
						maxiter=10000
					)
					iteration_count[0] = 10000 if info > 0 else 0
				
				final_residual = float(cp.asnumpy(cp.linalg.norm(laplacian_cupy @ pressure_flat_cupy - rhs_cupy)))
				
				if info != 0:
					print(f"Warning: CG  (info={info})，use...")
					iteration_count[0] = 0
					initial_residual = None
					try:
						pressure_flat_cupy, info = cupy_cg(
							laplacian_cupy, 
							rhs_cupy, 
							tol=1e-4,
							maxiter=20000,
							callback=callback
						)
					except TypeError:
						pressure_flat_cupy, info = cupy_cg(
							laplacian_cupy, 
							rhs_cupy, 
							tol=1e-4,
							maxiter=20000
						)
						iteration_count[0] = 20000 if info > 0 else 0
					final_residual = float(cp.asnumpy(cp.linalg.norm(laplacian_cupy @ pressure_flat_cupy - rhs_cupy)))
				
				pressure_flat = cp.asnumpy(pressure_flat_cupy)
			else:
				initial_residual = float(np.linalg.norm(rhs_mod))
				
				pressure_flat, info = scipy_cg(
					laplacian_mod, 
					rhs_mod, 
					tol=1e-6,
					maxiter=10000,
					callback=callback
				)
				
				final_residual = float(np.linalg.norm(laplacian_mod @ pressure_flat - rhs_mod))
				
				if info != 0:
					print(f"Warning: CG  (info={info})，use...")
					iteration_count[0] = 0
					initial_residual = None
					pressure_flat, info = scipy_cg(
						laplacian_mod, 
						rhs_mod, 
						tol=1e-4,
						maxiter=20000,
						callback=callback
					)
					final_residual = float(np.linalg.norm(laplacian_mod @ pressure_flat - rhs_mod))
			
			if info == 0:
				print(f"✓ CG :")
			else:
				print(f"✗ CG  (info={info}):")
			print(f"  : {iteration_count[0]}")
			print(f"  : {final_residual:.6e}")
			if initial_residual is not None:
				reduction_factor = initial_residual / final_residual if final_residual > 0 else float('inf')
				print(f"  : {initial_residual:.6e}")
				print(f"  : {reduction_factor:.2e}")
			
			pressure_field = pressure_flat.reshape(grid_shape)  # (Nx, Ny, Nz)
			
			pressure_fields.append(pressure_field)
			
			# vtk_output_path = f"{savedir}/images/pressure/pressure_frame_{frame_idx:06d}.vti"
			# save_pressure_to_vtk(pressure_field, grid_shape, lengths, vtk_output_path)
			
			pressure_for_vis = pressure_field.transpose(2, 1, 0)[..., np.newaxis]  # (Nz, Ny, Nx, 1)
			pressure_image = den_scalar2rgb(pressure_for_vis, scale=None, is3D=True, logv=False)
			
			output_path = f"{savedir}/images/pressure/pressure_frame_{frame_idx:06d}.png"
			imageio.imwrite(output_path, pressure_image)
			print(f"Frame {frame_idx} pressuresaved to: {output_path}")
			
			pressure_imgs.append(pressure_image)
			
		except Exception as e:
			print(f"Warning: : {e}")
			continue
	
	if pressure_imgs:
		video = np.stack(pressure_imgs, axis=0)
		video_path = f"{savedir}/video/pressure/pressure_video.mp4"
		imageio.mimwrite(video_path, video, fps=25, quality=8)
		print(f"\n=== pressurecomplete ===")
		print(f"pressureimages saved in: {savedir}/images/pressure/")
		print(f"pressure: {video_path}")
	else:
		print("Warning: framepressure")
	
	if obstacle_mask is not None and len(pressure_fields) > 0:
		print(f"\n===  ===")
		
		dx = lengths[0] / max(grid_shape[0] - 1, 1)
		dy = lengths[1] / max(grid_shape[1] - 1, 1)
		dz = lengths[2] / max(grid_shape[2] - 1, 1)
		
		forces_per_frame = []
		
		for frame_idx, pressure_field in enumerate(pressure_fields):
			obstacle_boundary = np.zeros(grid_shape, dtype=bool)
			if obstacle_mask.any():
				neighbor_xm = np.zeros_like(obstacle_mask, dtype=bool)
				neighbor_xp = np.zeros_like(obstacle_mask, dtype=bool)
				neighbor_ym = np.zeros_like(obstacle_mask, dtype=bool)
				neighbor_yp = np.zeros_like(obstacle_mask, dtype=bool)
				neighbor_zm = np.zeros_like(obstacle_mask, dtype=bool)
				neighbor_zp = np.zeros_like(obstacle_mask, dtype=bool)
				
				neighbor_xm[1:, :, :] = ~obstacle_mask[:-1, :, :]
				neighbor_xp[:-1, :, :] = ~obstacle_mask[1:, :, :]
				neighbor_ym[:, 1:, :] = ~obstacle_mask[:, :-1, :]
				neighbor_yp[:, :-1, :] = ~obstacle_mask[:, 1:, :]
				neighbor_zm[:, :, 1:] = ~obstacle_mask[:, :, :-1]
				neighbor_zp[:, :, :-1] = ~obstacle_mask[:, :, 1:]
				
				obstacle_boundary = obstacle_mask & (
					neighbor_xm | neighbor_xp | neighbor_ym | neighbor_yp | neighbor_zm | neighbor_zp
				)
			
			force_total = np.zeros(3)  # [Fx, Fy, Fz]
			
			boundary_indices = np.where(obstacle_boundary)
			for idx in range(len(boundary_indices[0])):
				i, j, k = boundary_indices[0][idx], boundary_indices[1][idx], boundary_indices[2][idx]
				
				normal = np.zeros(3)
				area_elements = []
				
				if i > 0 and not obstacle_mask[i-1, j, k]:
					normal[0] -= 1.0
					area_elements.append(dy * dz)
				if i < grid_shape[0] - 1 and not obstacle_mask[i+1, j, k]:
					normal[0] += 1.0
					area_elements.append(dy * dz)
				
				if j > 0 and not obstacle_mask[i, j-1, k]:
					normal[1] -= 1.0
					area_elements.append(dx * dz)
				if j < grid_shape[1] - 1 and not obstacle_mask[i, j+1, k]:
					normal[1] += 1.0
					area_elements.append(dx * dz)
				
				if k > 0 and not obstacle_mask[i, j, k-1]:
					normal[2] -= 1.0
					area_elements.append(dx * dy)
				if k < grid_shape[2] - 1 and not obstacle_mask[i, j, k+1]:
					normal[2] += 1.0
					area_elements.append(dx * dy)
				
				normal_norm = np.linalg.norm(normal)
				if normal_norm > 1e-10:
					normal = normal / normal_norm
				else:
					continue
				
				pressure_value = pressure_field[i, j, k]
				
				total_area = sum(area_elements) if area_elements else 0.0
				
				if total_area > 0:
					force_contribution = pressure_value * normal * total_area
					force_total += force_contribution
			
			forces_per_frame.append(force_total.copy())
			print(f"Frame {frame_idx}:  F = [{force_total[0]:.6e}, {force_total[1]:.6e}, {force_total[2]:.6e}]")
			print(f"   |F| = {np.linalg.norm(force_total):.6e}")
		
		forces_output_path = f"{savedir}/images/pressure/obstacle_forces.txt"
		with open(forces_output_path, 'w', encoding='utf-8') as f:
			f.write("（ ∫p·n·dS ）\n")
			f.write("=" * 60 + "\n\n")
			f.write(f": dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}\n")
			f.write(f"shape: {grid_shape}\n")
			f.write(f"total frame count: {len(pressure_fields)}\n\n")
			f.write(f"{'Frame':<8} {'Fx':<15} {'Fy':<15} {'Fz':<15} {'|F|':<15}\n")
			f.write("-" * 60 + "\n")
			
			for frame_idx, force in enumerate(forces_per_frame):
				force_magnitude = np.linalg.norm(force)
				f.write(f"{frame_idx:<8} {force[0]:<15.6e} {force[1]:<15.6e} {force[2]:<15.6e} {force_magnitude:<15.6e}\n")
			
			if len(forces_per_frame) > 0:
				forces_array = np.array(forces_per_frame)
				mean_force = np.mean(forces_array, axis=0)
				max_force = np.max(np.linalg.norm(forces_array, axis=1))
				mean_force_magnitude = np.linalg.norm(mean_force)
				
				f.write("\n" + "-" * 60 + "\n")
				f.write(f": F_mean = [{mean_force[0]:.6e}, {mean_force[1]:.6e}, {mean_force[2]:.6e}]\n")
				f.write(f": |F_mean| = {mean_force_magnitude:.6e}\n")
				f.write(f"latest: |F_max| = {max_force:.6e}\n")
		
		print(f"\nsaved to: {forces_output_path}")
		
		if len(forces_per_frame) > 0:
			forces_array = np.array(forces_per_frame)
			mean_force = np.mean(forces_array, axis=0)
			max_force = np.max(np.linalg.norm(forces_array, axis=1))
			mean_force_magnitude = np.linalg.norm(mean_force)
			print(f": F_mean = [{mean_force[0]:.6e}, {mean_force[1]:.6e}, {mean_force[2]:.6e}]")
			print(f": |F_mean| = {mean_force_magnitude:.6e}")
			print(f"latest: |F_max| = {max_force:.6e}")
			
			try:
				import matplotlib.pyplot as plt
				
				frame_indices = np.arange(len(forces_per_frame))
				forces_array = np.array(forces_per_frame)
				fx = forces_array[:, 0]
				fy = forces_array[:, 1]
				fz = forces_array[:, 2]
				
				fig, axes = plt.subplots(2, 2, figsize=(12, 10))
				fig.suptitle('', fontsize=14, fontweight='bold')
				
				axes[0, 0].plot(frame_indices, fx, 'r-', linewidth=2, label='Fx')
				axes[0, 0].set_xlabel('frame index', fontsize=11)
				axes[0, 0].set_ylabel(' Fx', fontsize=11)
				axes[0, 0].set_title('Xdirection', fontsize=12)
				axes[0, 0].grid(True, alpha=0.3)
				axes[0, 0].legend()
				
				axes[0, 1].plot(frame_indices, fy, 'g-', linewidth=2, label='Fy')
				axes[0, 1].set_xlabel('frame index', fontsize=11)
				axes[0, 1].set_ylabel(' Fy', fontsize=11)
				axes[0, 1].set_title('Ydirection', fontsize=12)
				axes[0, 1].grid(True, alpha=0.3)
				axes[0, 1].legend()
				
				axes[1, 0].plot(frame_indices, fz, 'b-', linewidth=2, label='Fz')
				axes[1, 0].set_xlabel('frame index', fontsize=11)
				axes[1, 0].set_ylabel(' Fz', fontsize=11)
				axes[1, 0].set_title('Zdirection', fontsize=12)
				axes[1, 0].grid(True, alpha=0.3)
				axes[1, 0].legend()
				
				axes[1, 1].plot(frame_indices, fx, 'r-', linewidth=2, label='Fx', alpha=0.7)
				axes[1, 1].plot(frame_indices, fy, 'g-', linewidth=2, label='Fy', alpha=0.7)
				axes[1, 1].plot(frame_indices, fz, 'b-', linewidth=2, label='Fz', alpha=0.7)
				axes[1, 1].set_xlabel('frame index', fontsize=11)
				axes[1, 1].set_ylabel('', fontsize=11)
				axes[1, 1].set_title('direction', fontsize=12)
				axes[1, 1].grid(True, alpha=0.3)
				axes[1, 1].legend()
				
				plt.tight_layout()
				
				force_plot_path = f"{savedir}/images/pressure/obstacle_forces_plot.png"
				plt.savefig(force_plot_path, dpi=150, bbox_inches='tight')
				print(f"saved to: {force_plot_path}")
				plt.close()
				
			except ImportError:
				print("Warning: matplotlib，")
			except Exception as e:
				print(f"Warning: : {e}")
	else:
		if obstacle_mask is None:
			print("\nWarning: mask，")
		elif len(pressure_fields) == 0:
			print("\nWarning: framepressure，")
