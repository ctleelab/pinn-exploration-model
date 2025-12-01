# --- installs you may need (once) ---
# pip install numpy scipy jax

import numpy as np
from scipy.ndimage import distance_transform_edt
import jax
import jax.numpy as jnp


# ---------- 1) Build SDF from mask ----------
def sdf_from_mask(mask: np.ndarray, spacing=(1.0, 1.0, 1.0), sign_convention="outside_positive"):
    mask = (mask > 0).astype(np.uint8)
    dist_out = distance_transform_edt(1 - mask, sampling=spacing)  # background -> membrane
    dist_in  = distance_transform_edt(mask,     sampling=spacing)  # membrane -> background
    d = dist_out - dist_in
    if sign_convention == "inside_positive":
        d = -d
    return d  # shape (Z,Y,X) in physical units

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_fill_holes

def sdf_from_surface_mask(surface_mask: np.ndarray, spacing=(1.,1.,1.), outside_positive=True):
    """
    surface_mask: 3D binary array where the membrane voxels are 1 (thin shell)
    spacing: (dz,dy,dx) in physical units
    outside_positive: if True -> outside distances > 0, inside < 0
    """
    surf = (surface_mask > 0)

    # 1) Classify inside vs outside by filling the shell
    filled = binary_fill_holes(surf)            # surface + interior (True)
    inside  = filled & ~surf                    # interior volume
    outside = ~filled                           # exterior volume

    # 2) Unsigned distance to the surface everywhere
    unsigned = distance_transform_edt(~surf, sampling=spacing).astype(np.float32)

    # 3) Apply sign
    d = unsigned.copy()
    if outside_positive:
        d[inside]  *= -1.0
    else:
        d[outside] *= -1.0

    d[surf] = 0.0
    return d


# pip install numpy scipy
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

def sdf_from_open_surface_mask(surface_mask: np.ndarray,
                               spacing=(1.0,1.0,1.0),
                               smooth_sigma_vox=1.0,
                               clip=None):
    """
    Build a signed distance field from an OPEN surface mask (thin shell: 1 on surface, 0 elsewhere).
    Sign is determined by local normals so one side is positive and the other is negative.
    Args:
        surface_mask: (Z,Y,X) binary array, 1 on surface voxels (thin shell), 0 elsewhere.
        spacing: (dz,dy,dx) voxel size in physical units.
        smooth_sigma_vox: Gaussian sigma (in voxels) to estimate stable normals from the shell.
        clip: optional float to clamp |d| (useful for visualization).
    Returns:
        d: (Z,Y,X) signed distance field (outside/inside defined by local normal direction).
    """
    surf = (surface_mask > 0)
    Z, Y, X = surf.shape
    dz, dy, dx = spacing

    # 1) Unsigned distance to the shell + nearest shell indices for every voxel
    U, (Zs, Ys, Xs) = distance_transform_edt(~surf, sampling=spacing, return_indices=True)
    U = U.astype(np.float32)

    # 2) Estimate surface normals from a smoothed indicator and take its gradient
    #    (direction points from high to low; consistency is enough—we just need two sides)
    f = gaussian_filter(surf.astype(np.float32), sigma=smooth_sigma_vox)
    gz, gy, gx = np.gradient(f, dz, dy, dx, edge_order=2)

    # 3) Gather the normal at the nearest surface voxel for each voxel
    nz = gz[Zs, Ys, Xs]
    ny = gy[Zs, Ys, Xs]
    nx = gx[Zs, Ys, Xs]
    nlen = np.sqrt(nx*nx + ny*ny + nz*nz) + 1e-12
    nx /= nlen; ny /= nlen; nz /= nlen

    # 4) Vector from the nearest surface voxel center to the voxel center (world units)
    zi, yi, xi = np.indices((Z, Y, X), dtype=np.float32)
    dxw = (xi - Xs) * dx
    dyw = (yi - Ys) * dy
    dzw = (zi - Zs) * dz

    # 5) Sign by local normal (dot product). Treat zeros as +.
    s = np.sign(dxw*nx + dyw*ny + dzw*nz).astype(np.float32)
    s[s == 0] = 1.0

    d = U * s
    d[surf] = 0.0  # exact zero on the shell
    if clip is not None:
        d = np.clip(d, -clip, clip)
    return d


# ---------- 2) Trilinear sampler (continuous in x,y,z) ----------
def trilinear_sample_jax(D, xyz_world, origin, spacing):
    dz, dy, dx = spacing
    z0, y0, x0 = origin
    x = (xyz_world[...,0] - x0) / dx
    y = (xyz_world[...,1] - y0) / dy
    z = (xyz_world[...,2] - z0) / dz

    Z, Y, X = D.shape
    zf = jnp.floor(z); yf = jnp.floor(y); xf = jnp.floor(x)
    z0i = jnp.clip(zf.astype(jnp.int32), 0, Z-1); z1i = jnp.clip(z0i+1, 0, Z-1)
    y0i = jnp.clip(yf.astype(jnp.int32), 0, Y-1); y1i = jnp.clip(y0i+1, 0, Y-1)
    x0i = jnp.clip(xf.astype(jnp.int32), 0, X-1); x1i = jnp.clip(x0i+1, 0, X-1)

    wz = z - zf; wy = y - yf; wx = x - xf
    def g(zi, yi, xi): return D[zi, yi, xi]

    c000 = g(z0i, y0i, x0i); c001 = g(z0i, y0i, x1i)
    c010 = g(z0i, y1i, x0i); c011 = g(z0i, y1i, x1i)
    c100 = g(z1i, y0i, x0i); c101 = g(z1i, y0i, x1i)
    c110 = g(z1i, y1i, x0i); c111 = g(z1i, y1i, x1i)

    c00 = c000*(1-wx) + c001*wx
    c01 = c010*(1-wx) + c011*wx
    c10 = c100*(1-wx) + c101*wx
    c11 = c110*(1-wx) + c111*wx
    c0  = c00*(1-wy) + c01*wy
    c1  = c10*(1-wy) + c11*wy
    return c0*(1-wz) + c1*wz

def make_d_fn(D_np, origin, spacing):
    D = jnp.array(D_np)  # jax array
    @jax.jit
    def d_fn(xyz):  # xyz (...,3) in world units (x,y,z)
        return trilinear_sample_jax(D, xyz, origin, spacing)
    return d_fn

# ---------- 3) φ mapping and ∇d ----------
def phi_from_sdf_vals(d_vals, eps):
    return jnp.tanh(d_vals / (jnp.sqrt(2.0) * eps))

def make_grad_d_fn(d_fn):
    # gradient per point
    point_grad = jax.grad(lambda p: d_fn(p[None, :])[0])
    return jax.jit(jax.vmap(point_grad))



import numpy as np
import matplotlib.pyplot as plt

def visualize_sdf_slice_and_derivative(
    D,                      # 3D SDF array, shape (Z, Y, X), outside-positive
    origin=(0.,0.,0.),      # (z0, y0, x0) world coords of voxel (0,0,0)
    spacing=(1.,1.,1.),     # (dz, dy, dx) voxel sizes in world units
    axis='z',               # 'z' (axial), 'y' (coronal), 'x' (sagittal)
    index=None,             # slice index along chosen axis (defaults to center)
    title_prefix="SDF"
):
    Z, Y, X = D.shape
    dz, dy, dx = spacing
    z0, y0, x0 = origin

    if index is None:
        index = {'z': Z//2, 'y': Y//2, 'x': X//2}[axis]

    # 3D gradients with correct physical spacing
    gz, gy, gx = np.gradient(D, dz, dy, dx, edge_order=2)
    grad_mag = np.sqrt(gx*gx + gy*gy + gz*gz)

    if axis == 'z':  # axial slice
        d2 = D[index, :, :]
        g2 = grad_mag[index, :, :]
        extent = [x0, x0+dx*(X-1), y0, y0+dy*(Y-1)]
        xlabel, ylabel = "x", "y"
        level_text = f"z = {z0 + index*dz:.3f}"
    elif axis == 'y':  # coronal slice
        d2 = D[:, index, :]
        g2 = grad_mag[:, index, :]
        extent = [x0, x0+dx*(X-1), z0, z0+dz*(Z-1)]
        xlabel, ylabel = "x", "z"
        level_text = f"y = {y0 + index*dy:.3f}"
    else:  # axis == 'x'  # sagittal slice
        d2 = D[:, :, index]
        g2 = grad_mag[:, :, index]
        extent = [y0, y0+dy*(Y-1), z0, z0+dz*(Z-1)]
        xlabel, ylabel = "y", "z"
        level_text = f"x = {x0 + index*dx:.3f}"

    # --- plot d on the slice, with zero-contour overlay ---
    plt.figure(figsize=(7,5))
    im1 = plt.imshow(d2, origin='lower', extent=extent, aspect='equal')
    plt.contour(d2, levels=[0.0], origin='lower', extent=extent, linewidths=1.0)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"{title_prefix}: d on slice ({level_text})")
    cb1 = plt.colorbar(im1); cb1.set_label("d")
    plt.tight_layout()
    plt.show()

    # --- plot ||∇d|| on the same slice ---
    plt.figure(figsize=(7,5))
    im2 = plt.imshow(g2, origin='lower', extent=extent, aspect='equal')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"{title_prefix}: ||∇d|| on slice ({level_text})")
    cb2 = plt.colorbar(im2); cb2.set_label("||∇d||")
    plt.tight_layout()
    plt.show()


def phi_from_sdf(D, eps):
    """Pointwise map from SDF d -> phi = tanh(d / (sqrt(2)*eps))."""
    return np.tanh(D / (np.sqrt(2.0) * float(eps)))


def visualize_phi_slice(
    D,                      # 3D SDF array (Z,Y,X)
    eps,                    # interface half-width in physical units
    origin=(0.,0.,0.),      # (z0,y0,x0) world coords of voxel (0,0,0)
    spacing=(1.,1.,1.),     # (dz,dy,dx) voxel sizes in world units
    axis='z',               # 'z'|'y'|'x'
    index=None,             # slice index along chosen axis (defaults to center)
    title_prefix="phi"
):
    Z, Y, X = D.shape
    dz, dy, dx = spacing
    z0, y0, x0 = origin

    phi = phi_from_sdf(D, eps)

    if index is None:
        index = {'z': Z//2, 'y': Y//2, 'x': X//2}[axis]

    if axis == 'z':
        im2d = phi[index, :, :]
        extent = [x0, x0+dx*(X-1), y0, y0+dy*(Y-1)]
        xlabel, ylabel = "x", "y"
        level_text = f"z = {z0 + index*dz:.3f}"
    elif axis == 'y':
        im2d = phi[:, index, :]
        extent = [x0, x0+dx*(X-1), z0, z0+dz*(Z-1)]
        xlabel, ylabel = "x", "z"
        level_text = f"y = {y0 + index*dy:.3f}"
    else:  # 'x'
        im2d = phi[:, :, index]
        extent = [y0, y0+dy*(Y-1), z0, z0+dz*(Z-1)]
        xlabel, ylabel = "y", "z"
        level_text = f"x = {x0 + index*dx:.3f}"

    plt.figure(figsize=(7,5))
    img = plt.imshow(im2d, origin='lower', extent=extent, vmin=-1, vmax=1, aspect='equal')
    # phi=0 contour coincides with d=0 (the membrane surface)
    # plt.contour(im2d, levels=[0.0], origin='lower', extent=extent, linewidths=1.0)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"{title_prefix} on slice ({level_text}),  ε={eps}")
    cb = plt.colorbar(img); cb.set_label("φ")
    plt.tight_layout()
    plt.show()

# Re-run after the state reset (the notebook's state was cleared).

import numpy as np
import matplotlib.pyplot as plt

def phi_from_sdf(D, eps):
    return np.tanh(D / (np.sqrt(2.0) * float(eps)))

def grad3(u, spacing):
    dz, dy, dx = spacing
    gz, gy, gx = np.gradient(u, dz, dy, dx, edge_order=2)
    return gz, gy, gx

def laplacian(u, spacing):
    dz, dy, dx = spacing
    gz, gy, gx = grad3(u, spacing)
    dzz = np.gradient(gz, dz, axis=0, edge_order=2)
    dyy = np.gradient(gy, dy, axis=1, edge_order=2)
    dxx = np.gradient(gx, dx, axis=2, edge_order=2)
    return dzz + dyy + dxx

def energy_density(phi, spacing, eps):
    gz, gy, gx = grad3(phi, spacing)
    grad_sq = gx*gx + gy*gy + gz*gz
    well = (1.0 - phi*phi)**2
    return eps*grad_sq + 0.5*(1.0/eps)*well

def residual_squared(phi, spacing, eps):
    lap = laplacian(phi, spacing)
    nonlin = (phi*phi - 1.0)*phi
    res = eps*lap - (1.0/eps)*nonlin
    return res*res

def visualize_phase_quantities_slice(D, eps, origin=(0.,0.,0.), spacing=(1.,1.,1.),
                                     axis='z', index=None, title_prefix="Phase quantities"):
    Z, Y, X = D.shape
    dz, dy, dx = spacing
    z0, y0, x0 = origin

    phi = phi_from_sdf(D, eps)
    Edens = energy_density(phi, spacing, eps)
    R2 = residual_squared(phi, spacing, eps)

    if index is None:
        index = {'z': Z//2, 'y': Y//2, 'x': X//2}[axis]

    if axis == 'z':
        im_phi = phi[index, :, :]
        im_E   = Edens[index, :, :]
        im_R2  = R2[index, :, :]
        extent = [x0, x0+dx*(X-1), y0, y0+dy*(Y-1)]
        xlabel, ylabel = "x", "y"
        level_text = f"z = {z0 + index*dz:.3f}"
        zero_contour = D[index, :, :]
    elif axis == 'y':
        im_phi = phi[:, index, :]
        im_E   = Edens[:, index, :]
        im_R2  = R2[:, index, :]
        extent = [x0, x0+dx*(X-1), z0, z0+dz*(Z-1)]
        xlabel, ylabel = "x", "z"
        level_text = f"y = {y0 + index*dy:.3f}"
        zero_contour = D[:, index, :]
    else:
        im_phi = phi[:, :, index]
        im_E   = Edens[:, :, index]
        im_R2  = R2[:, :, index]
        extent = [y0, y0+dy*(Y-1), z0, z0+dz*(Z-1)]
        xlabel, ylabel = "y", "z"
        level_text = f"x = {x0 + index*dx:.3f}"
        zero_contour = D[:, :, index]

    # phi
    plt.figure(figsize=(7,5))
    img1 = plt.imshow(im_phi, origin='lower', extent=extent, aspect='equal', vmin=-1, vmax=1)
    plt.contour(zero_contour, levels=[0.0], origin='lower', extent=extent, linewidths=1.0)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"{title_prefix}: φ on slice ({level_text}), ε={eps}")
    cb1 = plt.colorbar(img1); cb1.set_label("φ")
    plt.tight_layout()
    plt.show()

    # energy density
    plt.figure(figsize=(7,5))
    img2 = plt.imshow(im_E, origin='lower', extent=extent, aspect='equal')
    plt.contour(zero_contour, levels=[0.0], origin='lower', extent=extent, linewidths=0.6)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"{title_prefix}: ε|∇φ|² + (1/(2ε))(1-φ²)²  ({level_text})")
    cb2 = plt.colorbar(img2); cb2.set_label("energy density")
    plt.tight_layout()
    plt.show()

    # squared residual
    plt.figure(figsize=(7,5))
    img3 = plt.imshow(im_R2, origin='lower', extent=extent, aspect='equal')
    plt.contour(zero_contour, levels=[0.0], origin='lower', extent=extent, linewidths=0.6)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"{title_prefix}: (εΔφ - (1/ε)(φ²-1)φ)²  ({level_text})")
    cb3 = plt.colorbar(img3); cb3.set_label("residual²")
    plt.tight_layout()
    plt.show()



# pip install jax jaxlib numpy

import jax
import jax.numpy as jnp

# ---------------------------
# Tricubic (Catmull–Rom) SDF sampler
# ---------------------------
def _cubic_weights_catmull_rom(t):
    # t in [0,1]
    t2 = t * t
    t3 = t2 * t
    w0 = -0.5*t + 1.0*t2 - 0.5*t3
    w1 =  1.0    - 2.5*t2 + 1.5*t3
    w2 =  0.5*t  + 2.0*t2 - 1.5*t3
    w3 = -0.0    + -0.5*t2 + 0.5*t3
    return jnp.stack([w0, w1, w2, w3], axis=-1)  # (...,4)

def _tricubic_sample_point(D, xyz_world, origin, spacing):
    """
    D: (Z,Y,X) SDF samples
    xyz_world: (3,) = (x,y,z) in world units
    origin: (z0,y0,x0)
    spacing: (dz,dy,dx)
    returns: scalar d(x,y,z)
    """
    Z, Y, X = D.shape
    dz, dy, dx = spacing
    z0, y0, x0 = origin
    x, y, z = xyz_world

    # Convert to continuous grid coords
    xg = (x - x0) / dx
    yg = (y - y0) / dy
    zg = (z - z0) / dz

    # Clamp to safe range so we can use neighbors [-1,0,1,2]
    # (avoid hitting the border)
    xg = jnp.clip(xg, 1.0, X - 3.0001)
    yg = jnp.clip(yg, 1.0, Y - 3.0001)
    zg = jnp.clip(zg, 1.0, Z - 3.0001)

    ix = jnp.floor(xg).astype(jnp.int32)
    iy = jnp.floor(yg).astype(jnp.int32)
    iz = jnp.floor(zg).astype(jnp.int32)

    tx = xg - ix
    ty = yg - iy
    tz = zg - iz

    wx = _cubic_weights_catmull_rom(tx)  # (4,)
    wy = _cubic_weights_catmull_rom(ty)  # (4,)
    wz = _cubic_weights_catmull_rom(tz)  # (4,)

    x_idx = jnp.array([ix-1, ix, ix+1, ix+2], dtype=jnp.int32)
    y_idx = jnp.array([iy-1, iy, iy+1, iy+2], dtype=jnp.int32)
    z_idx = jnp.array([iz-1, iz, iz+1, iz+2], dtype=jnp.int32)

    # 4x4x4 neighborhood block
    block = D[z_idx[:,None,None], y_idx[None,:,None], x_idx[None,None,:]]  # (4,4,4)

    # Separable cubic: sum_{a,b,c} wz[a]*wy[b]*wx[c]*block[a,b,c]
    val = jnp.einsum('a,b,c,abc->', wz, wy, wx, block)
    return val

def make_continuous_sdf(D, origin, spacing):
    """
    Returns a JIT'd function d(xyz_batch) with xyz_batch shape (N,3) in world units.
    """
    Djax = jnp.array(D)

    def d_point(p):
        return _tricubic_sample_point(Djax, p, origin, spacing)

    d_batch = jax.jit(jax.vmap(d_point))  # (N,3) -> (N,)
    return d_batch

# ---------------------------
# φ, ∇φ, Δφ, energy, residual — all continuous in (x,y,z)
# ---------------------------
def make_phase_functions(D, origin, spacing, eps):
    """
    Returns callables over arbitrary points xyz (N,3) in world units:
      phi(xyz), grad_phi(xyz) -> (N,3), laplacian_phi(xyz), energy_density(xyz), residual_sq(xyz)
    """
    eps = float(eps)
    d_fn = make_continuous_sdf(D, origin, spacing)

    # scalar φ at a single point
    def phi_point(p):
        d = _tricubic_sample_point(jnp.array(D), p, origin, spacing)
        return jnp.tanh(d / (jnp.sqrt(2.0) * eps))

    # vectorized / JIT'd versions
    phi      = jax.jit(jax.vmap(phi_point))                              # (N,3)->(N,)
    grad1    = jax.grad(lambda p: phi_point(p))                          # (3,) -> (3,)
    grad_phi = jax.jit(jax.vmap(grad1))                                  # (N,3)->(N,3)

    # Laplacian via Hessian trace
    hess1    = jax.hessian(lambda p: phi_point(p))                       # (3,3)
    def lap_point(p):
        H = hess1(p)
        return jnp.trace(H)
    laplacian_phi = jax.jit(jax.vmap(lap_point))                         # (N,3)->(N,)

    # Energy density: ε|∇φ|² + (1/(2ε))(1-φ²)²
    def energy_point(p):
        g = grad1(p)
        ph = phi_point(p)
        return eps*jnp.dot(g,g) + 0.5*(1.0/eps)*(1.0 - ph*ph)**2
    energy_density = jax.jit(jax.vmap(energy_point))                     # (N,3)->(N,)

    # Residual²: ( εΔφ - (1/ε)(φ²-1)φ )²
    def residual_sq_point(p):
        ph = phi_point(p)
        lap = lap_point(p)
        res = eps*lap - (1.0/eps)*((ph*ph - 1.0)*ph)
        return res*res
    residual_sq = jax.jit(jax.vmap(residual_sq_point))                   # (N,3)->(N,)

    return phi, grad_phi, laplacian_phi, energy_density, residual_sq



def visualize_continuous_slice(D, origin, spacing, eps, axis='z', index=None, n=None,
                               title_prefix="Continuous"):
    """
    Samples sub-voxel φ, energy density, and residual² on a slice plane.
    D: SDF (Z,Y,X), outside-positive. origin=(z0,y0,x0), spacing=(dz,dy,dx).
    axis: 'z'|'y'|'x'. index: slice index in that axis (defaults to center).
    n: resolution along in-plane axes; defaults to the array's native size.
    """
    Z, Y, X = D.shape
    dz, dy, dx = spacing
    z0, y0, x0 = origin

    phi, _, _, E, R2 = make_phase_functions(D, origin, spacing, eps)

    if axis == 'z':
        if index is None: index = Z//2
        n0, n1 = (n or Y), (n or X)
        ys = y0 + jnp.linspace(0, (Y-1)*dy, n0)
        xs = x0 + jnp.linspace(0, (X-1)*dx, n1)
        yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
        zz = jnp.full_like(xx, z0 + index*dz)
        P = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        xlabel, ylabel = "x", "y"
        extent = [xs[0], xs[-1], ys[0], ys[-1]]
        level_text = f"z = {z0 + index*dz:.3f}"
    elif axis == 'y':
        if index is None: index = Y//2
        n0, n1 = (n or Z), (n or X)
        zs = z0 + jnp.linspace(0, (Z-1)*dz, n0)
        xs = x0 + jnp.linspace(0, (X-1)*dx, n1)
        zz, xx = jnp.meshgrid(zs, xs, indexing='ij')
        yy = jnp.full_like(xx, y0 + index*dy)
        P = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        xlabel, ylabel = "x", "z"
        extent = [xs[0], xs[-1], zs[0], zs[-1]]
        level_text = f"y = {y0 + index*dy:.3f}"
    else:  # 'x'
        if index is None: index = X//2
        n0, n1 = (n or Z), (n or Y)
        zs = z0 + jnp.linspace(0, (Z-1)*dz, n0)
        ys = y0 + jnp.linspace(0, (Y-1)*dy, n1)
        zz, yy = jnp.meshgrid(zs, ys, indexing='ij')
        xx = jnp.full_like(zz, x0 + index*dx)
        P = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        xlabel, ylabel = "y", "z"
        extent = [ys[0], ys[-1], zs[0], zs[-1]]
        level_text = f"x = {x0 + index*dx:.3f}"

    # Evaluate fields continuously at all points in the plane
    phi_ = phi(P).reshape(zz.shape)
    E_   = E(P).reshape(zz.shape)
    R2_  = R2(P).reshape(zz.shape)

    # Plot
    def _panel(im, title, cbar_label, vmin=None, vmax=None):
        plt.figure(figsize=(7,5))
        h = plt.imshow(im, origin='lower', extent=extent, aspect='equal', vmin=vmin, vmax=vmax)
        # zero-level of φ matches the membrane (nice overlay)
        plt.contour(phi_, levels=[0.0], origin='lower', extent=extent, linewidths=0.8)
        plt.xlabel(xlabel); plt.ylabel(ylabel)
        plt.title(f"{title_prefix}: {title}  ({level_text})")
        cb = plt.colorbar(h); cb.set_label(cbar_label)
        plt.tight_layout(); plt.show()

    _panel(phi_, r"$\phi$", "φ", vmin=-1, vmax=1)
    _panel(E_, r"$\epsilon|\nabla\phi|^2 + \frac{1}{2\epsilon}(1-\phi^2)^2$", "energy density")
    _panel(R2_, r"$(\epsilon\Delta\phi - \frac{1}{\epsilon}(\phi^2-1)\phi)^2$", "residual²")

