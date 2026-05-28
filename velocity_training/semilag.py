import torch
from typing import Tuple, Dict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np


def _make_grid(shape: Tuple[int, int, int], device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nx, ny, nz = shape
    x = torch.arange(nx, device=device, dtype=dtype)
    y = torch.arange(ny, device=device, dtype=dtype)
    z = torch.arange(nz, device=device, dtype=dtype)
    return torch.meshgrid(x, y, z, indexing="ij")


def _clamp_coords(xp: torch.Tensor, yp: torch.Tensor, zp: torch.Tensor, nx: int, ny: int, nz: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Clamp coordinates to valid range [0, N-1] for trilinear sampling.
    As specified: directly clamp x = torch.clamp(x, 0, Nx-1), etc.
    """
    xp = xp.clamp(0.0, nx - 1.0)
    yp = yp.clamp(0.0, ny - 1.0)
    zp = zp.clamp(0.0, nz - 1.0)
    return xp, yp, zp


def trilinear_sample(rho: torch.Tensor, xp: torch.Tensor, yp: torch.Tensor, zp: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Trilinear sample a scalar field rho at (xp, yp, zp) given in index-space coordinates.

    Args:
        rho: (Nx, Ny, Nz)
        xp, yp, zp: (Nx, Ny, Nz) sampling locations in [0, Nx-1], [0, Ny-1], [0, Nz-1]

    Returns:
        values: (Nx, Ny, Nz)
        cache: dict of intermediates for analytic gradients
    """
    nx, ny, nz = rho.shape
    x0 = xp.floor().to(torch.long)
    y0 = yp.floor().to(torch.long)
    z0 = zp.floor().to(torch.long)
    fx = xp - x0.to(xp.dtype)
    fy = yp - y0.to(yp.dtype)
    fz = zp - z0.to(zp.dtype)

    x1 = (x0 + 1).clamp(max=nx - 1)
    y1 = (y0 + 1).clamp(max=ny - 1)
    z1 = (z0 + 1).clamp(max=nz - 1)

    # Gather 8 neighbors
    c000 = rho[x0, y0, z0]
    c100 = rho[x1, y0, z0]
    c010 = rho[x0, y1, z0]
    c110 = rho[x1, y1, z0]
    c001 = rho[x0, y0, z1]
    c101 = rho[x1, y0, z1]
    c011 = rho[x0, y1, z1]
    c111 = rho[x1, y1, z1]

    wx0 = 1.0 - fx
    wy0 = 1.0 - fy
    wz0 = 1.0 - fz
    wx1 = fx
    wy1 = fy
    wz1 = fz

    # Trilinear interpolation
    v00 = wx0 * c000 + wx1 * c100
    v10 = wx0 * c010 + wx1 * c110
    v01 = wx0 * c001 + wx1 * c101
    v11 = wx0 * c011 + wx1 * c111
    v0 = wy0 * v00 + wy1 * v10
    v1 = wy0 * v01 + wy1 * v11
    values = wz0 * v0 + wz1 * v1

    cache = {
        "x0": x0, "y0": y0, "z0": z0,
        "x1": x1, "y1": y1, "z1": z1,
        "fx": fx, "fy": fy, "fz": fz,
        "wx0": wx0, "wy0": wy0, "wz0": wz0,
        "wx1": wx1, "wy1": wy1, "wz1": wz1,
        "c000": c000, "c100": c100, "c010": c010, "c110": c110,
        "c001": c001, "c101": c101, "c011": c011, "c111": c111,
    }
    return values, cache


def trilinear_backward_wrt_field(upstream: torch.Tensor, cache: Dict[str, torch.Tensor], shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Compute gradient wrt input field rho by scattering weights times upstream.
    upstream: (Nx, Ny, Nz) dL/d(values)
    returns grad_rho: (Nx, Ny, Nz)
    """
    nx, ny, nz = shape
    grad = torch.zeros(shape, device=upstream.device, dtype=upstream.dtype)

    x0 = cache["x0"]; y0 = cache["y0"]; z0 = cache["z0"]
    x1 = cache["x1"]; y1 = cache["y1"]; z1 = cache["z1"]
    wx0 = cache["wx0"]; wy0 = cache["wy0"]; wz0 = cache["wz0"]
    wx1 = cache["wx1"]; wy1 = cache["wy1"]; wz1 = cache["wz1"]

    # weights for 8 corners
    w000 = (wx0 * wy0 * wz0) * upstream
    w100 = (wx1 * wy0 * wz0) * upstream
    w010 = (wx0 * wy1 * wz0) * upstream
    w110 = (wx1 * wy1 * wz0) * upstream
    w001 = (wx0 * wy0 * wz1) * upstream
    w101 = (wx1 * wy0 * wz1) * upstream
    w011 = (wx0 * wy1 * wz1) * upstream
    w111 = (wx1 * wy1 * wz1) * upstream

    grad.index_put_((x0, y0, z0), w000, accumulate=True)
    grad.index_put_((x1, y0, z0), w100, accumulate=True)
    grad.index_put_((x0, y1, z0), w010, accumulate=True)
    grad.index_put_((x1, y1, z0), w110, accumulate=True)
    grad.index_put_((x0, y0, z1), w001, accumulate=True)
    grad.index_put_((x1, y0, z1), w101, accumulate=True)
    grad.index_put_((x0, y1, z1), w011, accumulate=True)
    grad.index_put_((x1, y1, z1), w111, accumulate=True)

    return grad


def trilinear_grad_wrt_coords(cache: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute ∇_{x'} ρ' = ∂A(x')/∂x' * ρ^t according to the analytic formula.
    
    For each corner (a,b,c) with weight w^{(a,b,c)} = (α)^a(1-α)^{1-a} * (β)^b(1-β)^{1-b} * (γ)^c(1-γ)^{1-c}:
    - ∂w/∂x = s_a/Δx * ̃w^{(b,c)}, where s_a = +1 if a=1, -1 if a=0
    - ̃w^{(b,c)} = (β)^b(1-β)^{1-b} * (γ)^c(1-γ)^{1-c}
    
    Since we use index coordinates, Δx = Δy = Δz = 1.
    
    Returns:
        dv_dxp, dv_dyp, dv_dzp: gradients of interpolated values wrt sampling coordinates
    """
    wx0 = cache["wx0"]  # (1-α)
    wx1 = cache["wx1"]  # α
    wy0 = cache["wy0"]  # (1-β)
    wy1 = cache["wy1"]  # β
    wz0 = cache["wz0"]  # (1-γ)
    wz1 = cache["wz1"]  # γ

    c000 = cache["c000"]  # corner (0,0,0)
    c100 = cache["c100"]  # corner (1,0,0)
    c010 = cache["c010"]  # corner (0,1,0)
    c110 = cache["c110"]  # corner (1,1,0)
    c001 = cache["c001"]  # corner (0,0,1)
    c101 = cache["c101"]  # corner (1,0,1)
    c011 = cache["c011"]  # corner (0,1,1)
    c111 = cache["c111"]  # corner (1,1,1)

    # Corner (0,0,0): s_a=-1, ̃w = wy0*wz0
    # Corner (1,0,0): s_a=+1, ̃w = wy0*wz0
    # Corner (0,1,0): s_a=-1, ̃w = wy1*wz0
    # Corner (1,1,0): s_a=+1, ̃w = wy1*wz0
    # Corner (0,0,1): s_a=-1, ̃w = wy0*wz1
    # Corner (1,0,1): s_a=+1, ̃w = wy0*wz1
    # Corner (0,1,1): s_a=-1, ̃w = wy1*wz1
    # Corner (1,1,1): s_a=+1, ̃w = wy1*wz1
    dv_dxp = (
        (-1.0) * (wy0 * wz0) * c000 +  # (0,0,0)
        (+1.0) * (wy0 * wz0) * c100 +  # (1,0,0)
        (-1.0) * (wy1 * wz0) * c010 +  # (0,1,0)
        (+1.0) * (wy1 * wz0) * c110 +  # (1,1,0)
        (-1.0) * (wy0 * wz1) * c001 +  # (0,0,1)
        (+1.0) * (wy0 * wz1) * c101 +  # (1,0,1)
        (-1.0) * (wy1 * wz1) * c011 +  # (0,1,1)
        (+1.0) * (wy1 * wz1) * c111    # (1,1,1)
    )

    dv_dyp = (
        (wx0 * (-1.0) * wz0) * c000 +  # (0,0,0)
        (wx1 * (-1.0) * wz0) * c100 +  # (1,0,0)
        (wx0 * (+1.0) * wz0) * c010 +  # (0,1,0)
        (wx1 * (+1.0) * wz0) * c110 +  # (1,1,0)
        (wx0 * (-1.0) * wz1) * c001 +  # (0,0,1)
        (wx1 * (-1.0) * wz1) * c101 +  # (1,0,1)
        (wx0 * (+1.0) * wz1) * c011 +  # (0,1,1)
        (wx1 * (+1.0) * wz1) * c111    # (1,1,1)
    )

    dv_dzp = (
        (wx0 * wy0 * (-1.0)) * c000 +  # (0,0,0)
        (wx1 * wy0 * (-1.0)) * c100 +  # (1,0,0)
        (wx0 * wy1 * (-1.0)) * c010 +  # (0,1,0)
        (wx1 * wy1 * (-1.0)) * c110 +  # (1,1,0)
        (wx0 * wy0 * (+1.0)) * c001 +  # (0,0,1)
        (wx1 * wy0 * (+1.0)) * c101 +  # (1,0,1)
        (wx0 * wy1 * (+1.0)) * c011 +  # (0,1,1)
        (wx1 * wy1 * (+1.0)) * c111    # (1,1,1)
    )

    # Since we use index coordinates directly (fx = xp - floor(xp)), 
    # dfx/dxp = 1 (except at integer boundaries, which we avoid via clamp)
    return dv_dxp, dv_dyp, dv_dzp


def semi_lagrangian_forward(rho_t: torch.Tensor, v_t: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Forward semi-Lagrangian advection (backtrace without velocity interpolation):
    x' = x - dt * v_t[x], sample rho_t at x'.

    Args:
        rho_t: (Nx, Ny, Nz)
        v_t: (Nx, Ny, Nz, 3) velocity at cell centers aligned with integer grid indices
        dt: scalar float

    Returns:
        rho_next: (Nx, Ny, Nz)
        cache: intermediates for analytic backward
    """
    assert rho_t.ndim == 3
    assert v_t.ndim == 4 and v_t.shape[-1] == 3
    nx, ny, nz = rho_t.shape
    device = rho_t.device
    dtype = rho_t.dtype

    X, Y, Z = _make_grid((nx, ny, nz), device, dtype)
    xp_raw = X - dt * v_t[..., 0]
    yp_raw = Y - dt * v_t[..., 1]
    zp_raw = Z - dt * v_t[..., 2]
    
    # Save raw coordinates before clamping to detect clamped regions
    xp, yp, zp = _clamp_coords(xp_raw, yp_raw, zp_raw, nx, ny, nz)
    
    # Detect which positions were clamped (gradient should be zero there)
    xp_clamped = (xp_raw != xp)
    yp_clamped = (yp_raw != yp)
    zp_clamped = (zp_raw != zp)

    values, cache_tri = trilinear_sample(rho_t, xp, yp, zp)
    cache = {
        "tri": cache_tri, 
        "xp": xp, "yp": yp, "zp": zp, 
        "xp_clamped": xp_clamped, "yp_clamped": yp_clamped, "zp_clamped": zp_clamped,
        "dt": torch.as_tensor(dt, device=device, dtype=dtype)
    }
    return values, cache


def semi_lagrangian_backward(upstream: torch.Tensor, rho_t: torch.Tensor, v_t: torch.Tensor, cache: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute analytic gradients wrt rho_t and v_t.
    
    According to the derivation:
    - ρ^{t+1} = A(x') * ρ^t, where x' = X - v^t * δ_t
    - ∂ρ^{t+1}/∂v^t = ∂A(x')/∂v^t = ∂A(x')/∂x' * ∂x'/∂v^t
    - ∂x'/∂v^t = -δ_t * I (since x' = X - dt * v_t)
    - ∂A(x')/∂x' is computed via trilinear_grad_wrt_coords using the analytic formula
    
    Args:
        upstream: dL/d(ρ_{t+1}) with shape (Nx, Ny, Nz)
        rho_t: density field at time t (used for shape, not values)
        v_t: velocity field at time t (used for shape, not values)
        cache: cached intermediates from forward pass
    
    Returns:
        grad_rho_t: gradient wrt ρ^t, shape (Nx, Ny, Nz)
        grad_v_t: gradient wrt v^t, shape (Nx, Ny, Nz, 3)
    """
    nx, ny, nz = rho_t.shape
    tri_cache = cache["tri"]
    
    # Gradient wrt rho_t: ∂L/∂ρ^t = (∂L/∂ρ^{t+1})^T * A(x')
    # This is equivalent to scattering the upstream gradient with trilinear weights
    grad_rho = trilinear_backward_wrt_field(upstream, tri_cache, (nx, ny, nz))

    # Gradient wrt x': ∂A(x')/∂x' computed via analytic formula
    # This gives ∇_{x'} ρ' = [∂ρ'/∂x, ∂ρ'/∂y, ∂ρ'/∂z]
    dv_dxp, dv_dyp, dv_dzp = trilinear_grad_wrt_coords(tri_cache)
    
    # Chain rule: ∂L/∂x' = (∂L/∂ρ^{t+1}) * (∂ρ^{t+1}/∂x')
    dL_dxp = upstream * dv_dxp
    dL_dyp = upstream * dv_dyp
    dL_dzp = upstream * dv_dzp

    # Chain to velocity: ∂L/∂v^t = (∂L/∂x') * (∂x'/∂v^t)
    # Since x' = X - dt * v_t, we have ∂x'/∂v_t = -dt * I
    # BUT: if x' was clamped, then ∂x'/∂v_t = 0 (clamp cuts the gradient)
    dt = cache["dt"]
    xp_clamped = cache["xp_clamped"]
    yp_clamped = cache["yp_clamped"]
    zp_clamped = cache["zp_clamped"]
    
    # Zero out gradients where coordinates were clamped
    dL_dxp_masked = dL_dxp * (~xp_clamped).to(dL_dxp.dtype)
    dL_dyp_masked = dL_dyp * (~yp_clamped).to(dL_dyp.dtype)
    dL_dzp_masked = dL_dzp * (~zp_clamped).to(dL_dzp.dtype)
    
    gx = -dt * dL_dxp_masked
    gy = -dt * dL_dyp_masked
    gz = -dt * dL_dzp_masked
    grad_v = torch.stack([gx, gy, gz], dim=-1)
    
    return grad_rho, grad_v


def compare_with_autodiff(nx: int = 8, ny: int = 9, nz: int = 7, dt: float = 0.2, seed: int = 0, device: str = "cpu") -> None:
    torch.manual_seed(seed)
    dtype = torch.float64  # use double for tighter comparison
    rho_t = torch.randn(nx, ny, nz, dtype=dtype, device=device, requires_grad=True) * 10
    v_t = torch.randn(nx, ny, nz, 3, dtype=dtype, device=device, requires_grad=True)

    # Forward with our implementation
    rho_next, cache = semi_lagrangian_forward(rho_t, v_t, dt)
    loss = (rho_next.pow(2).mean())  # arbitrary scalar loss

    # First compute upstream gradient (dL/d(rho_next)) and retain graph
    upstream = torch.autograd.grad(loss, rho_next, retain_graph=True, create_graph=False)[0]
    
    # Then compute autodiff grads (using our differentiable ops)
    grad_rho_auto, grad_v_auto = torch.autograd.grad(loss, (rho_t, v_t), create_graph=False, retain_graph=False)

    # Analytic grads using cached intermediates
    grad_rho_ana, grad_v_ana = semi_lagrangian_backward(upstream, rho_t.detach(), v_t.detach(), cache)

    def stats(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float, float]:
        """Compute absolute and relative errors between analytical and autodiff gradients.
        
        Returns:
            max_abs_diff: maximum absolute difference
            l2_diff: L2 norm of difference
            max_rel_diff: maximum relative difference (in percentage), only for non-zero values
            l2_rel_diff: L2 relative difference (in percentage)
        """
        diff = (a - b).abs()
        max_abs_diff = float(diff.max().item())
        l2_diff = float(diff.norm().item())
        
        # Relative error: |analytical - autodiff| / max(|autodiff|, |analytical|) * 100%
        # Use the larger of the two values as denominator for robustness
        abs_auto = a.abs()
        abs_ana = b.abs()
        denom = torch.maximum(abs_auto, abs_ana)
        
        # Only compute relative error where denominator is significant
        # Use a threshold based on the magnitude of the gradients
        max_val = max(float(abs_auto.max().item()), float(abs_ana.max().item()))
        threshold = max(max_val * 1e-6, 1e-10)  # At least 1e-10
        mask = denom > threshold
        
        if mask.any():
            rel_diff = diff / (denom + 1e-12) * 100.0
            # Only consider values where denominator is significant
            max_rel_diff = float(rel_diff[mask].max().item())
        else:
            # If all values are too small, relative error is not meaningful
            max_rel_diff = 0.0
        
        # L2 relative error: ||diff||_2 / max(||autodiff||_2, ||analytical||_2) * 100%
        l2_auto = float(a.norm().item())
        l2_ana = float(b.norm().item())
        l2_denom = max(l2_auto, l2_ana)
        l2_rel_diff = (l2_diff / (l2_denom + 1e-12)) * 100.0 if l2_denom > 1e-12 else 0.0
        
        return max_abs_diff, l2_diff, max_rel_diff, l2_rel_diff

    max_rho, norm_rho, max_rel_rho, norm_rel_rho = stats(grad_rho_auto, grad_rho_ana)
    max_v, norm_v, max_rel_v, norm_rel_v = stats(grad_v_auto, grad_v_ana)

    print("Comparison (analytic vs autodiff):")
    print(f" - grad rho_t: max abs diff = {max_rho:.3e}, L2 diff = {norm_rho:.3e}")
    print(f"              max rel diff = {max_rel_rho:.3e}%, L2 rel diff = {norm_rel_rho:.3e}%")
    print(f" - grad v_t  : max abs diff = {max_v:.3e}, L2 diff = {norm_v:.3e}")
    print(f"              max rel diff = {max_rel_v:.3e}%, L2 rel diff = {norm_rel_v:.3e}%")
    
    # Visualize velocity gradient differences
    # Compute norm of difference for each spatial location (3D vector difference)
    grad_v_diff = grad_v_auto - grad_v_ana
    grad_v_diff_norm = grad_v_diff.norm(dim=-1)  # (Nx, Ny, Nz) - norm of 3D vector at each location
    
    # Also compute norm of autodiff gradient for reference
    grad_v_auto_norm = grad_v_auto.norm(dim=-1)  # (Nx, Ny, Nz)
    grad_v_ana_norm = grad_v_ana.norm(dim=-1)    # (Nx, Ny, Nz)
    
    # Convert to numpy for visualization
    grad_v_diff_norm_np = grad_v_diff_norm.detach().cpu().numpy()
    grad_v_auto_norm_np = grad_v_auto_norm.detach().cpu().numpy()
    grad_v_ana_norm_np = grad_v_ana_norm.detach().cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Select middle slices for visualization
    mid_x = nx // 2
    mid_y = ny // 2
    mid_z = nz // 2
    
    # Row 1: Difference norm
    im1 = axes[0, 0].imshow(grad_v_diff_norm_np[mid_x, :, :], aspect='auto', origin='lower')
    axes[0, 0].set_title(f'Grad diff norm (x={mid_x} slice)')
    axes[0, 0].set_xlabel('z')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(grad_v_diff_norm_np[:, mid_y, :], aspect='auto', origin='lower')
    axes[0, 1].set_title(f'Grad diff norm (y={mid_y} slice)')
    axes[0, 1].set_xlabel('z')
    axes[0, 1].set_ylabel('x')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(grad_v_diff_norm_np[:, :, mid_z], aspect='auto', origin='lower')
    axes[0, 2].set_title(f'Grad diff norm (z={mid_z} slice)')
    axes[0, 2].set_xlabel('y')
    axes[0, 2].set_ylabel('x')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Row 2: Autodiff gradient norm for reference
    im4 = axes[1, 0].imshow(grad_v_auto_norm_np[mid_x, :, :], aspect='auto', origin='lower')
    axes[1, 0].set_title(f'Autodiff grad norm (x={mid_x} slice)')
    axes[1, 0].set_xlabel('z')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(grad_v_auto_norm_np[:, mid_y, :], aspect='auto', origin='lower')
    axes[1, 1].set_title(f'Autodiff grad norm (y={mid_y} slice)')
    axes[1, 1].set_xlabel('z')
    axes[1, 1].set_ylabel('x')
    plt.colorbar(im5, ax=axes[1, 1])
    
    im6 = axes[1, 2].imshow(grad_v_auto_norm_np[:, :, mid_z], aspect='auto', origin='lower')
    axes[1, 2].set_title(f'Autodiff grad norm (z={mid_z} slice)')
    axes[1, 2].set_xlabel('y')
    axes[1, 2].set_ylabel('x')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('velocity_grad_diff_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to 'velocity_grad_diff_visualization.png'")
    print(f"Max diff norm: {grad_v_diff_norm.max().item():.6e}")
    print(f"Mean diff norm: {grad_v_diff_norm.mean().item():.6e}")
    print(f"Max autodiff grad norm: {grad_v_auto_norm.max().item():.6e}")
    print(f"Mean autodiff grad norm: {grad_v_auto_norm.mean().item():.6e}")
    print(f"Max analytic grad norm: {grad_v_ana_norm.max().item():.6e}")
    print(f"Mean analytic grad norm: {grad_v_ana_norm.mean().item():.6e}")


if __name__ == "__main__":
    # Quick self-test
    compare_with_autodiff(nx = 128, ny = 192, nz = 128, dt = 10.0, device = "cuda")


