import torch
import numpy as np


class CoordinateTransform:
	def __init__(self, voxel_tran, voxel_scale, device):
		# voxel_tran: transformation matrix [3, 4] or [4, 4]
		#   In dataset_readers.py, voxel_tran[:3, :3] is used as rotation matrix R_s2w
		#   and voxel_tran[:3, 3] is used as translation T_s2w
		# voxel_scale: scale factor [3]
		voxel_tran_full = torch.from_numpy(voxel_tran).float().to(device)
		if voxel_tran_full.shape[0] == 4:
			voxel_tran_full = voxel_tran_full[:3, :]
		self.voxel_tran = voxel_tran_full  # [3, 4]
		self.scale = torch.from_numpy(voxel_scale).float().to(device)
		
		# Extract rotation and translation for Smoke->World (matching dataset_readers.py)
		# Formula: world = (smoke * scale) @ R_s2w + T_s2w
		self.R_s2w = self.voxel_tran[:3, :3]  # [3, 3] rotation matrix
		self.T_s2w = self.voxel_tran[:3, 3]    # [3] translation vector
		
		# Precompute inverse for World->Smoke
		# Formula: smoke = ((world - T_s2w) @ R_s2w_inv) / scale
		self.R_s2w_inv = torch.inverse(self.R_s2w)  # [3, 3] inverse rotation matrix
	
	def smoke2world(self, xyz_smoke):
		"""
		Transform from smoke space to world space.
		Matches the implementation in dataset_readers.py:
		world = (smoke * scale) @ R_s2w + T_s2w
		
		Args:
			xyz_smoke: [N, 3] points in smoke space [0, 1]
		
		Returns:
			xyz_world: [N, 3] points in world space
		"""
		# Scale: smoke * scale
		pos_scaled = xyz_smoke * self.scale  # [N, 3]
		
		# Rotate: pos_scaled @ R_s2w
		pos_rot = torch.matmul(pos_scaled, self.R_s2w.T)  # [N, 3]
		
		# Translate: pos_rot + T_s2w
		pos_world = pos_rot + self.T_s2w  # [N, 3]
		
		return pos_world
	
	def world2smoke(self, xyz_world):
		"""
		Transform from world space to smoke space.
		Inverse of smoke2world:
		smoke = ((world - T_s2w) @ R_s2w_inv) / scale
		
		Args:
			xyz_world: [N, 3] points in world space
		
		Returns:
			xyz_smoke: [N, 3] points in smoke space [0, 1]
		"""
		# Translate: world - T_s2w
		pos_untrans = xyz_world - self.T_s2w  # [N, 3]
		
		# Rotate: pos_untrans @ R_s2w_inv
		pos_rot = torch.matmul(pos_untrans, self.R_s2w_inv.T)  # [N, 3]
		
		# Scale: pos_rot / scale
		pos_smoke = pos_rot / self.scale  # [N, 3]
		
		return pos_smoke

