import torch
import torch.nn as nn
from utils.general_utils import inverse_sigmoid


class GaussianOverrideWrapper:
	"""
	，used for GaussianModel  get_xyz attributes，
	attributesoriginal。
	"""
	def __init__(self, original_model, override_xyz):
		self.orig = original_model
		self.override_xyz = override_xyz
	
	@property
	def get_xyz(self):
		return self.override_xyz
	
	@property
	def get_opacity(self):
		return self.orig.get_opacity
		
	@property
	def get_features(self):
		return self.orig.get_features
		
	@property
	def get_scaling(self):
		return self.orig.get_scaling
		
	@property
	def get_rotation(self):
		return self.orig.get_rotation
		
	@property
	def active_sh_degree(self):
		return self.orig.active_sh_degree
	
	def __getattr__(self, name):
		return getattr(self.orig, name)


class ExtendedGaussianWrapper:
	"""，original Gaussian  inflow Gaussian attributes"""
	def __init__(self, orig_gaussians, merged_pos_world, inflow_gaussians, inflow_group_indices):
		"""
		Args:
			orig_gaussians: original Gaussian 
			merged_pos_world: mergedposition（original +  inflow ）
			inflow_gaussians: InflowGaussians object
			inflow_group_indices: to include inflow list（ [0, 1, 2] containsframe 0、1、2 ）
		"""
		self.orig = orig_gaussians
		self.inflow = inflow_gaussians
		self.inflow_group_indices = inflow_group_indices
		self.merged_pos = merged_pos_world
		
		self.orig_num_points = orig_gaussians.get_xyz.shape[0]
		
		merged_opacity = self._compute_merged_opacity()
		merged_scaling = self._compute_merged_scaling()
		merged_rotation = self._compute_merged_rotation()
		merged_features_dc = self._compute_merged_features_dc()
		merged_features_rest = self._compute_merged_features_rest()
		
		self._opacity = merged_opacity.detach().requires_grad_(True)
		self._scaling = merged_scaling.detach().requires_grad_(True)
		self._rotation = merged_rotation.detach().requires_grad_(True)
		self._features_dc = merged_features_dc.detach().requires_grad_(True)
		self._features_rest = merged_features_rest.detach().requires_grad_(True)
		
		self._merged_opacity_ref = merged_opacity
		self._merged_scaling_ref = merged_scaling
		self._merged_rotation_ref = merged_rotation
		self._merged_features_dc_ref = merged_features_dc
		self._merged_features_rest_ref = merged_features_rest
	
	def _compute_merged_opacity(self):
		"""merged opacity（pre-activation）"""
		orig_opacity = self.orig._opacity
		inflow_opacities = []
		for group_idx in self.inflow_group_indices:
			inflow_opacities.append(self.inflow._opacity_groups[group_idx])
		if inflow_opacities:
			inflow_opacity = torch.cat(inflow_opacities, dim=0)
			return torch.cat([orig_opacity, inflow_opacity], dim=0)
		else:
			return orig_opacity
	
	def _compute_merged_scaling(self):
		"""merged scaling（pre-activation）"""
		orig_scaling = self.orig._scaling
		inflow_scalings = []
		for group_idx in self.inflow_group_indices:
			inflow_scalings.append(self.inflow._scaling_groups[group_idx])
		if inflow_scalings:
			inflow_scaling = torch.cat(inflow_scalings, dim=0)
			return torch.cat([orig_scaling, inflow_scaling], dim=0)
		else:
			return orig_scaling
	
	def _compute_merged_rotation(self):
		"""merged rotation"""
		orig_rotation = self.orig._rotation
		inflow_rotations = []
		for group_idx in self.inflow_group_indices:
			inflow_rotations.append(self.inflow.get_group_rotation(group_idx))
		if inflow_rotations:
			inflow_rotation = torch.cat(inflow_rotations, dim=0)
			return torch.cat([orig_rotation, inflow_rotation], dim=0)
		else:
			return orig_rotation
	
	def _compute_merged_features_dc(self):
		"""merged features_dc"""
		orig_features_dc = self.orig._features_dc
		inflow_features_dc_list = []
		for group_idx in self.inflow_group_indices:
			inflow_features_dc_list.append(self.inflow.get_group_features_dc(group_idx))
		if inflow_features_dc_list:
			inflow_features_dc = torch.cat(inflow_features_dc_list, dim=0)
			return torch.cat([orig_features_dc, inflow_features_dc], dim=0)
		else:
			return orig_features_dc
	
	def _compute_merged_features_rest(self):
		"""merged features_rest"""
		orig_features_rest = self.orig._features_rest
		inflow_features_rest_list = []
		for group_idx in self.inflow_group_indices:
			inflow_features_rest_list.append(self.inflow.get_group_features_rest(group_idx))
		if inflow_features_rest_list:
			inflow_features_rest = torch.cat(inflow_features_rest_list, dim=0)
			return torch.cat([orig_features_rest, inflow_features_rest], dim=0)
		else:
			return orig_features_rest
	
	@property
	def get_xyz(self):
		return self.merged_pos
	
	@property
	def get_opacity(self):
		return self.opacity_activation(self._opacity)
	
	@property
	def get_scaling(self):
		return self.scaling_activation(self._scaling)
	
	@property
	def get_rotation(self):
		return self.rotation_activation(self._rotation)
	
	@property
	def get_features_dc(self):
		return self._features_dc
	
	@property
	def get_features_rest(self):
		return self._features_rest
	
	@property
	def get_features(self):
		"""returnmergedfull features"""
		return torch.cat([self._features_dc, self._features_rest], dim=1)
	
	@property
	def active_sh_degree(self):
		return self.orig.active_sh_degree
	
	@property
	def opacity_activation(self):
		return self.orig.opacity_activation
	
	@property
	def scaling_activation(self):
		return self.orig.scaling_activation
	
	@property
	def rotation_activation(self):
		return self.orig.rotation_activation
	
	@property
	def covariance_activation(self):
		return self.orig.covariance_activation
	
	@property
	def max_sh_degree(self):
		return self.orig.max_sh_degree
	
	def __getattr__(self, name):
		return getattr(self.orig, name)


class InflowOnlyGaussianWrapper:
	"""，contains group  inflow GS，used for"""
	def __init__(self, inflow_gaussians, group_idx):
		"""
		Args:
			inflow_gaussians: InflowGaussians object
			group_idx: to include inflow 
		"""
		self.inflow = inflow_gaussians
		self.group_idx = group_idx
		
		self.inflow_pos_world = inflow_gaussians.get_group_xyz(group_idx)  # [N, 3]
		
		num_points = self.inflow_pos_world.shape[0]
		device = self.inflow_pos_world.device
		self._deformation_table = torch.zeros(num_points, dtype=torch.bool, device=device)
		
		self._opacity = self._compute_opacity()
		self._scaling = self._compute_scaling()
		self._rotation = self._compute_rotation()
		self._features_dc = self._compute_features_dc()
		self._features_rest = self._compute_features_rest()
	
	def _compute_opacity(self):
		""" opacity（pre-activation）"""
		return self.inflow._opacity_groups[self.group_idx]
	
	def _compute_scaling(self):
		""" scaling（pre-activation）"""
		return self.inflow._scaling_groups[self.group_idx]
	
	def _compute_rotation(self):
		""" rotation"""
		return self.inflow.get_group_rotation(self.group_idx)
	
	def _compute_features_dc(self):
		""" features_dc"""
		return self.inflow.get_group_features_dc(self.group_idx)
	
	def _compute_features_rest(self):
		""" features_rest"""
		return self.inflow.get_group_features_rest(self.group_idx)
	
	@property
	def get_xyz(self):
		return self.inflow_pos_world
	
	@property
	def get_opacity(self):
		return self.opacity_activation(self._opacity)
	
	@property
	def get_scaling(self):
		return self.scaling_activation(self._scaling)
	
	@property
	def get_rotation(self):
		return self.rotation_activation(self._rotation)
	
	@property
	def get_features_dc(self):
		return self._features_dc
	
	@property
	def get_features_rest(self):
		return self._features_rest
	
	@property
	def get_features(self):
		"""returnfull features"""
		return torch.cat([self._features_dc, self._features_rest], dim=1)
	
	@property
	def active_sh_degree(self):
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
	
	@property
	def covariance_activation(self):
		def identity(x):
			return x
		return identity
	
	@property
	def max_sh_degree(self):
		return self.inflow.max_sh_degree
	
	def __getattr__(self, name):
		return getattr(self.inflow, name)

