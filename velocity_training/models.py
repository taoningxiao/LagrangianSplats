import torch
import torch.nn as nn
from utils.sh_utils import RGB2SH
from utils.general_utils import inverse_sigmoid

try:
	from simple_knn._C import distCUDA2
except ImportError:
	distCUDA2 = None


class InflowGaussians(nn.Module):
	"""manage (T-1)  inflow Gaussian """
	def __init__(self, num_groups, num_points_per_group, lengths_tensor, inflow_ratio, device, gaussians_template, coord_trans, inflow_region_min=None, inflow_region_max=None):
		super(InflowGaussians, self).__init__()
		self.num_groups = num_groups
		self.num_points_per_group = num_points_per_group
		self.lengths_tensor = lengths_tensor
		self.inflow_ratio = inflow_ratio
		self.device = device
		self.max_sh_degree = gaussians_template.max_sh_degree
		self.coord_trans = coord_trans
		
		if inflow_region_min is None:
			inflow_region_min = [0.0, 0.1, 0.0]
		if inflow_region_max is None:
			inflow_region_max = [1.0, 0.3, 1.0]
		
		if isinstance(inflow_region_min, (list, tuple)):
			inflow_region_min = torch.tensor(inflow_region_min, device=device, dtype=torch.float32)
		if isinstance(inflow_region_max, (list, tuple)):
			inflow_region_max = torch.tensor(inflow_region_max, device=device, dtype=torch.float32)
		
		self.inflow_region_min = inflow_region_min
		self.inflow_region_max = inflow_region_max
		
		self._initialized_groups = [False] * num_groups
		
		self._group_origin_frames = [None] * num_groups
		
		self._xyz_groups = nn.ParameterList()
		self._features_dc_groups = nn.ParameterList()
		self._features_rest_groups = nn.ParameterList()
		self._opacity_groups = nn.ParameterList()
		self._scaling_groups = nn.ParameterList()
		self._rotation_groups = nn.ParameterList()
		
		for _ in range(num_groups):
			new_xyz_sim = torch.rand(num_points_per_group, 3, device=device)
			for axis in range(3):
				axis_range = inflow_region_max[axis] - inflow_region_min[axis]
				axis_min = inflow_region_min[axis] * lengths_tensor[axis]
				new_xyz_sim[:, axis] = new_xyz_sim[:, axis] * axis_range * lengths_tensor[axis] + axis_min
			
			# Sim Space -> Smoke Space: xyz_sim / lengths_tensor
			new_xyz_smoke = new_xyz_sim / lengths_tensor
			# Smoke Space -> World Space: coord_trans.smoke2world
			new_xyz_world = coord_trans.smoke2world(new_xyz_smoke)
			
			self._xyz_groups.append(nn.Parameter(new_xyz_world.requires_grad_(True)))
			
			white_color = torch.ones(num_points_per_group, 3, device=device) * 0.9
			fused_color = RGB2SH(white_color)
			features = torch.zeros(num_points_per_group, 3, (self.max_sh_degree + 1) ** 2, device=device)
			features[:, :3, 0] = fused_color
			features[:, 3:, 1:] = 0.0
			
			new_features_dc = features[:, :, 0:1].transpose(1, 2).contiguous()
			new_features_rest = features[:, :, 1:].transpose(1, 2).contiguous()
			self._features_dc_groups.append(nn.Parameter(new_features_dc.requires_grad_(True)))
			self._features_rest_groups.append(nn.Parameter(new_features_rest.requires_grad_(True)))
			
			if distCUDA2 is not None:
				dist2 = torch.clamp_min(distCUDA2(new_xyz_world), 0.0000001)
				new_scaling = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
			else:
				initial_scale = torch.log(torch.sqrt(torch.tensor(0.001, device=device))).item()
				new_scaling = torch.ones(num_points_per_group, 3, device=device) * initial_scale
			self._scaling_groups.append(nn.Parameter(new_scaling.requires_grad_(True)))
			
			new_rotation = torch.zeros(num_points_per_group, 4, device=device)
			new_rotation[:, 0] = 1.0
			self._rotation_groups.append(nn.Parameter(new_rotation.requires_grad_(True)))
			
			new_opacities = inverse_sigmoid(0.05 * torch.ones(num_points_per_group, 1, dtype=torch.float, device=device))
			self._opacity_groups.append(nn.Parameter(new_opacities.requires_grad_(True)))
	
	def get_group_xyz(self, group_idx):
		"""returnframe group_idx position（World Space）"""
		return self._xyz_groups[group_idx]
	
	def get_group_xyz_sim(self, group_idx):
		"""returnframe group_idx position（Sim Space），used for"""
		xyz_world = self._xyz_groups[group_idx]
		# World Space -> Smoke Space -> Sim Space
		xyz_smoke = self.coord_trans.world2smoke(xyz_world)
		xyz_sim = xyz_smoke * self.lengths_tensor
		return xyz_sim
	
	def get_group_opacity(self, group_idx):
		"""returnframe group_idx opacity（）"""
		opacity_activation = torch.sigmoid
		return opacity_activation(self._opacity_groups[group_idx])
	
	def get_group_scaling(self, group_idx):
		"""returnframe group_idx scaling（）"""
		scaling_activation = torch.exp
		return scaling_activation(self._scaling_groups[group_idx])
	
	def get_group_rotation(self, group_idx):
		"""returnframe group_idx rotation"""
		return self._rotation_groups[group_idx]
	
	def get_group_features_dc(self, group_idx):
		"""returnframe group_idx """
		return self._features_dc_groups[group_idx]
	
	def get_group_features_rest(self, group_idx):
		"""returnframe group_idx """
		return self._features_rest_groups[group_idx]
	
	def is_group_initialized(self, group_idx):
		"""checkframe group_idx GSin"""
		return self._initialized_groups[group_idx]
	
	def mark_group_initialized(self, group_idx):
		"""markframe group_idx """
		self._initialized_groups[group_idx] = True
	
	def set_group_origin_frame(self, group_idx, origin_frame):
		"""setframe group_idx frame index"""
		self._group_origin_frames[group_idx] = origin_frame
	
	def get_group_origin_frame(self, group_idx):
		"""getframe group_idx frame index"""
		return self._group_origin_frames[group_idx]
	
	def parameters(self):
		"""return"""
		for group_idx in range(self.num_groups):
			yield self._xyz_groups[group_idx]
			yield self._features_dc_groups[group_idx]
			yield self._features_rest_groups[group_idx]
			yield self._opacity_groups[group_idx]
			yield self._scaling_groups[group_idx]
			yield self._rotation_groups[group_idx]
	
	def capture(self):
		"""captureall parameter state，used for checkpoint"""
		xyz_groups_state = [group_param.data.clone() for group_param in self._xyz_groups]
		features_dc_groups_state = [group_param.data.clone() for group_param in self._features_dc_groups]
		features_rest_groups_state = [group_param.data.clone() for group_param in self._features_rest_groups]
		opacity_groups_state = [group_param.data.clone() for group_param in self._opacity_groups]
		scaling_groups_state = [group_param.data.clone() for group_param in self._scaling_groups]
		rotation_groups_state = [group_param.data.clone() for group_param in self._rotation_groups]
		
		return {
			'num_groups': self.num_groups,
			'num_points_per_group': self.num_points_per_group,
			'lengths_tensor': self.lengths_tensor.clone() if isinstance(self.lengths_tensor, torch.Tensor) else self.lengths_tensor,
			'inflow_ratio': self.inflow_ratio,
			'max_sh_degree': self.max_sh_degree,
			'initialized_groups': self._initialized_groups.copy(),
			'group_origin_frames': self._group_origin_frames.copy(),
			'xyz_groups': xyz_groups_state,
			'features_dc_groups': features_dc_groups_state,
			'features_rest_groups': features_rest_groups_state,
			'opacity_groups': opacity_groups_state,
			'scaling_groups': scaling_groups_state,
			'rotation_groups': rotation_groups_state,
		}
	
	@classmethod
	def restore(cls, checkpoint_data, gaussians_template, coord_trans, device):
		"""
		restore from checkpoint InflowGaussians
		
		Args:
			checkpoint_data:  capture() return
			gaussians_template: GaussianModel （used forget max_sh_degree）
			coord_trans: CoordinateTransform object
			device: device
		
		Returns:
			InflowGaussians instance
		"""
		inflow_gaussians = cls(
			num_groups=checkpoint_data['num_groups'],
			num_points_per_group=checkpoint_data['num_points_per_group'],
			lengths_tensor=checkpoint_data['lengths_tensor'],
			inflow_ratio=checkpoint_data['inflow_ratio'],
			device=device,
			gaussians_template=gaussians_template,
			coord_trans=coord_trans
		).to(device)
		
		for group_idx in range(checkpoint_data['num_groups']):
			inflow_gaussians._xyz_groups[group_idx].data.copy_(checkpoint_data['xyz_groups'][group_idx])
			inflow_gaussians._features_dc_groups[group_idx].data.copy_(checkpoint_data['features_dc_groups'][group_idx])
			inflow_gaussians._features_rest_groups[group_idx].data.copy_(checkpoint_data['features_rest_groups'][group_idx])
			inflow_gaussians._opacity_groups[group_idx].data.copy_(checkpoint_data['opacity_groups'][group_idx])
			inflow_gaussians._scaling_groups[group_idx].data.copy_(checkpoint_data['scaling_groups'][group_idx])
			inflow_gaussians._rotation_groups[group_idx].data.copy_(checkpoint_data['rotation_groups'][group_idx])
		
		if 'initialized_groups' in checkpoint_data:
			inflow_gaussians._initialized_groups = checkpoint_data['initialized_groups'].copy() if isinstance(checkpoint_data['initialized_groups'], list) else list(checkpoint_data['initialized_groups'])
		else:
			inflow_gaussians._initialized_groups = [False] * checkpoint_data['num_groups']
		
		if 'group_origin_frames' in checkpoint_data:
			inflow_gaussians._group_origin_frames = checkpoint_data['group_origin_frames'].copy() if isinstance(checkpoint_data['group_origin_frames'], list) else list(checkpoint_data['group_origin_frames'])
		else:
			inflow_gaussians._group_origin_frames = [None] * checkpoint_data['num_groups']
		
		return inflow_gaussians

