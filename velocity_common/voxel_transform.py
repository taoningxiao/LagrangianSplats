import torch


class VoxelTrans:
    def __init__(self, voxel_tran, voxel_tran_inv, voxel_scale, device):
        self.s2w = voxel_tran
        self.w2s = voxel_tran_inv
        self.scale = voxel_scale
        self.device = device

    def smoke2world(self, coords):
        pts = coords.view(-1, 4)
        p_smoke = pts[:, :3]
        pos_scale = p_smoke * self.scale
        pos_rot = torch.sum(pos_scale[..., None, :] * self.s2w[:3, :3], -1)
        pos_off = self.s2w[:3, -1].expand(pos_rot.shape)
        pts[:, :3] = pos_rot + pos_off
        return pts

    def world2smoke(self, pts):
        pts_clone = pts.clone()
        p_world = pts_clone[..., :3]
        pos_rot = torch.sum(p_world[..., None, :] * self.w2s[:3, :3], dim=-1)
        pos_off = self.w2s[:3, -1].expand_as(pos_rot)
        pos_scale = (pos_rot + pos_off) / self.scale
        pts_clone[..., :3] = pos_scale
        return pts_clone.clone()

    def vel_world2smoke(self, v_world, st_factor):
        st_factor_tensor = torch.tensor(st_factor).expand((3,)).to(self.device)
        vel_rot = torch.sum(v_world.clone()[..., None, :] * self.w2s[:3, :3], -1)
        return vel_rot / self.scale * st_factor_tensor

    def vel_smoke2world(self, v_smoke, st_factor):
        st_factor_tensor = torch.tensor(st_factor).expand((3,)).to(self.device)
        vel_scale = v_smoke.clone() * self.scale / st_factor_tensor
        return torch.sum(vel_scale[..., None, :] * self.s2w[:3, :3], -1)
