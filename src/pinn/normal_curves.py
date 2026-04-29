import numpy as np
import jax
import jax.numpy as jnp
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pinn.model import PINN, grad_phi, hessian_phi
from jax import vmap
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
from pathlib import Path
from pinn.plot import compute_isosurface_mesh_from_checkpoint


# -----------------------
# Helpers: adjugate (cofactor transpose) for batched 3x3 matrices
# -----------------------
def adjugate_3x3_batched(A):
    """
    A: (N,3,3)
    returns adj(A): (N,3,3)
    """
    a00, a01, a02 = A[:, 0, 0], A[:, 0, 1], A[:, 0, 2]
    a10, a11, a12 = A[:, 1, 0], A[:, 1, 1], A[:, 1, 2]
    a20, a21, a22 = A[:, 2, 0], A[:, 2, 1], A[:, 2, 2]

    # Cofactors (not transposed yet)
    C00 = a11 * a22 - a12 * a21
    C01 = -(a10 * a22 - a12 * a20)
    C02 = a10 * a21 - a11 * a20

    C10 = -(a01 * a22 - a02 * a21)
    C11 = a00 * a22 - a02 * a20
    C12 = -(a00 * a21 - a01 * a20)

    C20 = a01 * a12 - a02 * a11
    C21 = -(a00 * a12 - a02 * a10)
    C22 = a00 * a11 - a01 * a10

    # adj(A) = cofactor(A)^T
    adjA = jnp.stack(
        [
            jnp.stack([C00, C10, C20], axis=1),
            jnp.stack([C01, C11, C21], axis=1),
            jnp.stack([C02, C12, C22], axis=1),
        ],
        axis=1,
    )
    return adjA


# -----------------------
# Core: kappa (div n) and K at a single point (x,) using your grad/hessian helpers
# -----------------------
def make_kappa_K_fns(phi_fn_pts, normal_eps=1e-8, curvature_eps=1e-8):
    """
    Returns scalar functions kappa(x), K(x), and vector n(x),
    suitable for jax.grad/jacobian.
    """
    def n_of_x(x):
        g = grad_phi(phi_fn_pts, x[None, :])[0]  # (3,)
        G = jnp.linalg.norm(g) + normal_eps
        return g / G

    def kappa_of_x(x):
        g = grad_phi(phi_fn_pts, x[None, :])[0]             # (3,)
        H = hessian_phi(phi_fn_pts, x[None, :])[0]          # (3,3)
        trH = jnp.trace(H)
        Hg = H @ g
        gHg = jnp.dot(g, Hg)
        G = jnp.linalg.norm(g)
        Gsafe = jnp.sqrt(G * G + curvature_eps**2)
        return trH / Gsafe - gHg / (Gsafe**3)               # div(n) = 2H

    def K_of_x(x):
        g = grad_phi(phi_fn_pts, x[None, :])[0]             # (3,)
        H = hessian_phi(phi_fn_pts, x[None, :])[0]          # (3,3)
        G = jnp.linalg.norm(g)
        Gsafe = jnp.sqrt(G * G + curvature_eps**2)
        n = g / (Gsafe + normal_eps)

        # K = n^T adj(H) n / |∇φ|^2  (evaluated on the level set)
        # adj(H) here is adjugate (cofactor transpose)
        # For a single 3x3, we can reuse batched helper by adding batch dim
        adjH = adjugate_3x3_batched(H[None, :, :])[0]
        return (n @ adjH @ n) / (Gsafe**2)

    return n_of_x, kappa_of_x, K_of_x


def surface_laplacian_of_scalar(f_scalar, n_of_x):
    """
    Returns function Δ_s f(x) = div( P(x) grad f(x) ), with P = I - n n^T.
    Uses autodiff: divergence = trace(Jacobian).
    """
    def v_of_x(x):
        n = n_of_x(x)                       # (3,)
        P = jnp.eye(3) - jnp.outer(n, n)     # (3,3)
        gf = jax.grad(f_scalar)(x)          # (3,)
        return P @ gf                       # (3,)

    def div_v(x):
        J = jax.jacfwd(v_of_x)(x)           # (3,3)
        return jnp.trace(J)

    return div_v


# -----------------------
# Main function
# -----------------------
def calc_norm_curv_K_force(
    checkpoint,
    grid_size=64,
    normal_eps=1e-8,
    curvature_eps=1e-8,
    curvature_kind="H",     # "kappa" (= div(n)) or "H" (= kappa/2)
    transpose=True,
    x_range=None,
    y_range=None,
    z_range=None,
    level=0.0,

    # force params (Helfrich, no spontaneous curvature by default)
    kappa_b=1.0,                # bending rigidity κ
    sigma=0.0,                  # tension σ
    compute_force=True,
    batch_size=4096,            # chunking for expensive 4th-derivative ops

    half_length=1.0,            # physical half-length per axis
    hidden_dim=128,
):
    """
    Returns:
      verts, faces,
      normals (N,3),
      curv (N,)         # kappa or H depending on curvature_kind
      gauss (N,)        # Gaussian curvature K
      force (N,3)       # Helfrich force vector (if compute_force else None)

    Notes:
      - kappa := div(n) = 2H
      - Gaussian curvature computed from ∇φ and ∇²φ
      - Force uses f_n = -κ_b(Δ_s kappa + kappa^3 - 2 kappa K) + σ kappa, then f = f_n n
    """

    # --- 1) Extract cropped isosurface mesh ---
    verts, faces = compute_isosurface_mesh_from_checkpoint(
        checkpoint=checkpoint,
        grid_size=grid_size,
        level=level,
        transpose=transpose,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        hidden_dim=hidden_dim,
    )
    faces = faces.astype(np.int32)

    # --- 2) NN φ(x) and derivatives at verts ---
    model = PINN(hidden_dim=hidden_dim)
    params = checkpoint["state"]["params"]
    phi_fn_pts = lambda pts: model.apply(params, pts)  # expects (N,3)

    # voxel_scale = jnp.asarray((half_length, half_length, half_length))
    # def phi_fn_pts(pts):
    #     pts = jnp.asarray(pts)                 # pts in physical coordinates
    #     pts_norm = pts / voxel_scale           # convert physical -> normalized
    #     return model.apply(params, pts_norm)   # expects (N,3)

    verts_j = jnp.asarray(verts)  # (N,3)

    grads = grad_phi(phi_fn_pts, verts_j)                        # (N,3)
    gnorm = jnp.linalg.norm(grads, axis=1, keepdims=True)        # (N,1)
    normals_j = grads / (gnorm + normal_eps)                     # (N,3)
    normals = - np.asarray(normals_j)

    Hphi = hessian_phi(phi_fn_pts, verts_j)                      # (N,3,3)
    trH = jnp.trace(Hphi, axis1=1, axis2=2)                      # (N,)

    Hg = jnp.einsum("nij,nj->ni", Hphi, grads)                   # (N,3)
    gHg = jnp.einsum("ni,ni->n", grads, Hg)                      # (N,)

    G = jnp.squeeze(gnorm, axis=1)                               # (N,)
    Gsafe = jnp.sqrt(G * G + curvature_eps**2)                   # (N,)

    kappa = trH / Gsafe - gHg / (Gsafe**3)                       # (N,) = div(n)=2H

    # --- 3) Gaussian curvature K ---
    adjH = adjugate_3x3_batched(Hphi)                            # (N,3,3)
    # K = n^T adj(Hphi) n / |∇φ|^2
    num = jnp.einsum("ni,nij,nj->n", normals_j, adjH, normals_j) # (N,)
    K = num / (Gsafe**2)                                         # (N,)
    gauss = np.asarray(K)

    # --- 4) Choose curvature output (kappa or H) ---
    if curvature_kind.lower() == "h":
        curv_j = -0.5 * kappa
    else:
        curv_j = -kappa
    curv = np.asarray(curv_j)

    # --- 5) Force (optional; expensive) ---
    force = None
    if compute_force:
        # Build scalar functions for kappa(x), K(x), n(x)
        n_of_x, kappa_of_x, K_of_x = make_kappa_K_fns(
            phi_fn_pts, normal_eps=normal_eps, curvature_eps=curvature_eps
        )

        # Δ_s kappa(x) via autodiff: div(P grad kappa)
        lap_s_kappa_of_x = surface_laplacian_of_scalar(kappa_of_x, n_of_x)

        # Vectorize
        v_kappa = jax.vmap(kappa_of_x)
        v_K = jax.vmap(K_of_x)
        v_n = jax.vmap(n_of_x)
        v_lap = jax.vmap(lap_s_kappa_of_x)

        # Chunk to control memory
        N = verts_j.shape[0]
        out_force = []
        for i in range(0, N, batch_size):
            x = verts_j[i : i + batch_size]          # (B,3)
            kap = v_kappa(x)                         # (B,)
            Kg = v_K(x)                              # (B,)
            nrm = v_n(x)                             # (B,3)
            lap = v_lap(x)                           # (B,)

            # Helfrich (no C0): f_n = -κ_b(Δ_s kappa + kappa^3 - 2 kappa K) + σ kappa
            # f_n = -kappa_b * (lap + kap**3 - 2.0 * kap * Kg) + sigma * kap # original
            f_n =  kappa_b * (lap - kap**3 + 2.0 * kap * Kg) + sigma * kap # test
            out_force.append(f_n[:, None] * nrm)      # (B,3)

        force = np.asarray(jnp.concatenate(out_force, axis=0))

    theta = np.arccos(normals @ (1,0,0))

    scale = half_length / 1.0
    verts = verts * scale
    curv  = curv  / scale
    gauss = gauss / (scale**2)

    return verts, faces, normals, theta, curv, gauss, force



def calc_norm_curv(
    checkpoint,
    grid_size=64,
    normal_eps=1e-8,
    curvature_eps=1e-8,
    curvature_kind="kappa",     # "kappa" (= div(n)) or "H" (= kappa/2)
    transpose=True,             # passed to compute_isosurface_mesh_from_checkpoint
    x_range=None,               # e.g. (-0.5, 0.5)
    y_range=None,               # e.g. (-1.0, 0.2)
    z_range=None,               # e.g. (-0.5, 0.5)
    level=0.0,                  # isosurface level (default 0)
):
    """
    1) Extract φ=level isosurface with marching cubes, restricting sampling to
       x_range/y_range/z_range (in normalized coordinates, typically [-1,1]).
    2) Compute:
        - normals n = ∇φ/||∇φ||
        - curvature κ = div(n)
          = tr(Hess φ)/||∇φ|| - (g^T Hess φ g)/||∇φ||^3
      (optionally H = κ/2)

    curvature_kind:
      - "kappa": κ = div(n)  (twice mean curvature)
      - "H": mean curvature H = κ/2
    """

    # --- 1) Extract cropped isosurface mesh (verts in (x,y,z) normalized coords) ---
    verts, faces = compute_isosurface_mesh_from_checkpoint(
        checkpoint=checkpoint,
        grid_size=grid_size,
        level=level,
        transpose=transpose,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
    )
    faces = faces.astype(np.int32)

    # --- 2) NN φ(x) and derivatives at verts ---
    model = PINN()
    params = checkpoint["state"]["params"]

    # model.apply expects (N,3)
    phi_fn_pts = lambda pts: model.apply(params, pts)

    verts_j = jnp.asarray(verts)  # (N,3)

    # Gradient g = ∇φ at verts
    grads = grad_phi(phi_fn_pts, verts_j)  # (N,3)
    gnorm = jnp.linalg.norm(grads, axis=1, keepdims=True)  # (N,1)
    normals = np.asarray(grads / (gnorm + normal_eps))

    # Hessian at verts: (N,3,3)
    H = hessian_phi(phi_fn_pts, verts_j)
    trH = jnp.trace(H, axis1=1, axis2=2)  # (N,)

    # g^T H g
    Hg = jnp.einsum("nij,nj->ni", H, grads)      # (N,3)
    gHg = jnp.einsum("ni,ni->n", grads, Hg)      # (N,)

    G = jnp.squeeze(gnorm, axis=1)               # (N,)
    Gsafe = jnp.sqrt(G * G + curvature_eps**2)   # (N,)

    kappa = trH / Gsafe - gHg / (Gsafe**3)       # (N,) = div(n)

    if curvature_kind.lower() == "h":
        kappa = 0.5 * kappa

    curv = np.asarray(kappa)

    return verts, faces, normals, curv



def calc_norm_curv_ori(
    checkpoint,
    grid_size=64,
    normal_eps=1e-8,
    curvature_eps=1e-8,
    curvature_kind="kappa",     # "kappa" (= div(n)) or "H" (= kappa/2)
):
    """
    Extract φ=0 isosurface with marching cubes and compute:
      - normals n = ∇φ/||∇φ||
      - curvature κ = div(n) = tr(Hess φ)/||∇φ|| - (g^T Hess φ g)/||∇φ||^3

    curvature_kind:
      - "kappa": κ = div(n)  (twice mean curvature)
      - "H": mean curvature H = κ/2
    """

    # --- 1) Build grid in [-1, 1]^3 ---
    x = jnp.linspace(-1, 1, grid_size)
    y = jnp.linspace(-1, 1, grid_size)
    z = jnp.linspace(-1, 1, grid_size)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    grid_points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    # --- 2) NN φ(x) on grid ---
    model = PINN()
    state = checkpoint["state"]
    params = state["params"]

    phi_fn = lambda pts: model.apply(params, pts)  # expects (N,3)
    phi_values = phi_fn(grid_points).reshape(grid_size, grid_size, grid_size)
    phi_values_np = np.asarray(phi_values, dtype=np.float32).copy()

    # --- 3) Marching cubes (IMPORTANT: correct spacing for linspace(-1,1,N)) ---
    spacing = (2 / (grid_size - 1),) * 3
    verts, faces, _, _ = marching_cubes(phi_values_np, level=0.0, spacing=spacing)

    # marching_cubes vertices are in [0,2] coordinate; shift to [-1,1]
    verts = verts - 1.0  # (N,3), numpy
    faces = faces.astype(np.int32)

    # --- 4) Compute normals (and curvature) from NN derivatives at verts ---
    normals = None
    curv = None

    # Define phi_fn for derivative routines (still (N,3)->...)
    phi_fn_pts = lambda pts: model.apply(params, pts)
    verts_j = jnp.asarray(verts)

    # Gradient g = ∇φ at verts
    grads = grad_phi(phi_fn_pts, verts_j)  # (N,3), JAX
    gnorm = jnp.linalg.norm(grads, axis=1, keepdims=True)  # (N,1)
    normals = np.asarray(grads / (gnorm + normal_eps))

    # Hessian at verts: (N,3,3)
    H = hessian_phi(phi_fn_pts, verts_j)
    trH = jnp.trace(H, axis1=1, axis2=2)  # (N,)

    # g^T H g
    # (N,3,3) @ (N,3,1) -> (N,3,1); then dot with g -> (N,)
    Hg = jnp.einsum("nij,nj->ni", H, grads)          # (N,3)
    gHg = jnp.einsum("ni,ni->n", grads, Hg)          # (N,)

    G = jnp.squeeze(gnorm, axis=1)                   # (N,)
    Gsafe = jnp.sqrt(G * G + curvature_eps**2)       # (N,)

    kappa = trH / Gsafe - gHg / (Gsafe**3)           # (N,)  = div(n)
    if curvature_kind.lower() == "h":
        kappa = 0.5 * kappa
    curv = np.asarray(kappa)

    # --- 8) Return ---
    return verts, faces, normals, curv



def plot_norm_curv(
    ax,
    verts, 
    faces,
    norms=None, 
    curvs=None,
    no_label=True,
    n_quiver=0,
    quiver_length=0.05,
    norm_direction=1.0,
    # ---- curvature options ----
    curvature_eps=1e-8,
    color_by_curvature=True,
    cmap="coolwarm",
    clim=None,                 # None or (vmin, vmax)
    show_colorbar=False,
    cbar_label=None,
):
    norms = norms * norm_direction
    
    # --- 5) Plot mesh (optionally color by curvature) ---
    if color_by_curvature and (curvs is not None):
        # Convert vertex curvature to face curvature by averaging the 3 vertices
        abs_curvs = np.abs(curvs)
        face_vals = abs_curvs[faces].mean(axis=1)  # (M,)

        # Normalize to [0,1] for colormap
        vmin, vmax = (np.min(face_vals), np.max(face_vals)) if clim is None else clim
        if vmin == vmax:
            vmax = vmin + 1e-12

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        face_colors = cmap_obj(norm(face_vals))  # (M,4)

        mesh = Poly3DCollection(
            verts[faces],
            alpha=0.85,                 # higher alpha so colors are visible
            edgecolor="k",
            linewidth=0.15,
            facecolors=face_colors,
        )
        ax.add_collection3d(mesh)

        if show_colorbar:
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
            mappable.set_array(face_vals)
            label = cbar_label
            if label is None:
                # label = "kappa = div(n)" if curvature_kind.lower() == "kappa" else "H (mean curvature)"
                label = r"Mean curvature ($1/\sigma$)"
            # plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02, label=label)
            cbar = plt.colorbar(
                mappable,
                ax=ax,
                orientation="horizontal",
                shrink=0.6,
                pad=0.15,     # vertical spacing when horizontal
                label=label,
            )
            # cbar.outline.set_visible(False)


    else:
        mesh = Poly3DCollection(
            verts[faces],
            alpha=0.1,
            edgecolor="k",
            linewidth=0.2,
            facecolor="cyan",
        )
        ax.add_collection3d(mesh)

    # --- 6) Optional: draw normal arrows ---
    if (norms is not None) and (n_quiver and n_quiver > 0):
        n = verts.shape[0]
        sel = np.linspace(0, n - 1, min(n_quiver, n), dtype=int)
        P = verts[sel]
        N = norms[sel]
        ax.quiver(
            P[:, 0], P[:, 1], P[:, 2],
            N[:, 0], N[:, 1], N[:, 2],
            length=quiver_length,
            normalize=True,
            color="black",
            alpha=1.0,
            linewidths=0.5,
        )

    # --- 7) Styling (yours) ---
    if no_label is True:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
        ax.xaxis.line.set_color((1, 1, 1, 0))
        ax.yaxis.line.set_color((1, 1, 1, 0))
        ax.zaxis.line.set_color((1, 1, 1, 0))

        limit_val = 0.6
        ax.set_xlim(-limit_val, limit_val)
        ax.set_ylim(-limit_val, limit_val)
        ax.set_zlim(-0.5, 0.5)
    else:
        ax.set_title(f"Step {step}", fontsize=12, y=0.9)
        ax.view_init(elev=30, azim=45)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)

    ax.grid(False)
    ax.set_facecolor("white")

    return




def calc_plot_norm_curv(
    ax,
    checkpoint,
    grid_size=64,
    no_label=True,
    step=None,
    return_normals=True,
    normal_eps=1e-8,
    n_quiver=0,
    quiver_length=0.05,
    norm_direction=1.0,

    # ---- curvature options ----
    compute_curvature=True,
    curvature_eps=1e-8,
    curvature_kind="kappa",     # "kappa" (= div(n)) or "H" (= kappa/2)
    color_by_curvature=True,
    cmap="coolwarm",
    clim=None,                 # None or (vmin, vmax)
    show_colorbar=False,
    cbar_label=None,
):
    """
    Extract φ=0 isosurface with marching cubes and compute:
      - normals n = ∇φ/||∇φ||
      - curvature κ = div(n) = tr(Hess φ)/||∇φ|| - (g^T Hess φ g)/||∇φ||^3

    curvature_kind:
      - "kappa": κ = div(n)  (twice mean curvature)
      - "H": mean curvature H = κ/2
    """

    # --- 1) Build grid in [-1, 1]^3 ---
    x = jnp.linspace(-1, 1, grid_size)
    y = jnp.linspace(-1, 1, grid_size)
    z = jnp.linspace(-1, 1, grid_size)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    grid_points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    # --- 2) NN φ(x) on grid ---
    model = PINN()
    state = checkpoint["state"]
    params = state["params"]

    phi_fn = lambda pts: model.apply(params, pts)  # expects (N,3)

    phi_values = phi_fn(grid_points).reshape(grid_size, grid_size, grid_size)

    # marching_cubes needs writable numpy
    phi_values_np = np.asarray(phi_values, dtype=np.float32).copy()

    # --- 3) Marching cubes (IMPORTANT: correct spacing for linspace(-1,1,N)) ---
    spacing = (2 / (grid_size - 1),) * 3
    verts, faces, _, _ = marching_cubes(phi_values_np, level=0.0, spacing=spacing)

    # marching_cubes vertices are in [0,2] coordinate; shift to [-1,1]
    verts = verts - 1.0  # (N,3), numpy
    faces = faces.astype(np.int32)

    # --- 4) Compute normals (and curvature) from NN derivatives at verts ---
    normals = None
    curv = None

    # Define phi_fn for derivative routines (still (N,3)->...)
    phi_fn_pts = lambda pts: model.apply(params, pts)

    verts_j = jnp.asarray(verts)

    if return_normals or compute_curvature:
        # Gradient g = ∇φ at verts
        grads = grad_phi(phi_fn_pts, verts_j)  # (N,3), JAX
        gnorm = jnp.linalg.norm(grads, axis=1, keepdims=True)  # (N,1)

        if return_normals:
            normals = np.asarray(norm_direction * grads / (gnorm + normal_eps))

        if compute_curvature:
            # Hessian at verts: (N,3,3)
            H = hessian_phi(phi_fn_pts, verts_j)

            # tr(H)
            trH = jnp.trace(H, axis1=1, axis2=2)  # (N,)

            # g^T H g
            # (N,3,3) @ (N,3,1) -> (N,3,1); then dot with g -> (N,)
            Hg = jnp.einsum("nij,nj->ni", H, grads)          # (N,3)
            gHg = jnp.einsum("ni,ni->n", grads, Hg)          # (N,)

            G = jnp.squeeze(gnorm, axis=1)                   # (N,)
            Gsafe = jnp.sqrt(G * G + curvature_eps**2)       # (N,)

            kappa = trH / Gsafe - gHg / (Gsafe**3)           # (N,)  = div(n)
            if curvature_kind.lower() == "h":
                kappa = 0.5 * kappa

            curv = np.asarray(kappa)

    # --- 5) Plot mesh (optionally color by curvature) ---
    if color_by_curvature and (curv is not None):
        # Convert vertex curvature to face curvature by averaging the 3 vertices
        abs_curv = np.abs(curv)
        face_vals = abs_curv[faces].mean(axis=1)  # (M,)

        # face_vals = curv[faces].mean(axis=1)  # (M,)

        # Normalize to [0,1] for colormap
        vmin, vmax = (np.min(face_vals), np.max(face_vals)) if clim is None else clim
        if vmin == vmax:
            vmax = vmin + 1e-12

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        face_colors = cmap_obj(norm(face_vals))  # (M,4)

        mesh = Poly3DCollection(
            verts[faces],
            alpha=0.85,                 # higher alpha so colors are visible
            edgecolor="k",
            linewidth=0.15,
            facecolors=face_colors,
        )
        ax.add_collection3d(mesh)

        if show_colorbar:
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
            mappable.set_array(face_vals)
            label = cbar_label
            if label is None:
                label = "kappa = div(n)" if curvature_kind.lower() == "kappa" else "H (mean curvature)"
            plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02, label=label)

    else:
        mesh = Poly3DCollection(
            verts[faces],
            alpha=0.1,
            edgecolor="k",
            linewidth=0.2,
            facecolor="cyan",
        )
        ax.add_collection3d(mesh)

    # --- 6) Optional: draw normal arrows ---
    if (return_normals and normals is not None) and (n_quiver and n_quiver > 0):
        n = verts.shape[0]
        sel = np.linspace(0, n - 1, min(n_quiver, n), dtype=int)
        P = verts[sel]
        N = normals[sel]
        ax.quiver(
            P[:, 0], P[:, 1], P[:, 2],
            N[:, 0], N[:, 1], N[:, 2],
            length=quiver_length,
            normalize=True,
            color="black",
            alpha=1.0,
            linewidths=0.5,
        )

    # --- 7) Styling (yours) ---
    if no_label is True:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
        ax.xaxis.line.set_color((1, 1, 1, 0))
        ax.yaxis.line.set_color((1, 1, 1, 0))
        ax.zaxis.line.set_color((1, 1, 1, 0))

        limit_val = 0.6
        ax.set_xlim(-limit_val, limit_val)
        ax.set_ylim(-limit_val, limit_val)
        ax.set_zlim(-0.5, 0.5)
    else:
        ax.set_title(f"Step {step}", fontsize=12, y=0.9)
        ax.view_init(elev=30, azim=45)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)

    ax.grid(False)
    ax.set_facecolor("white")

    # --- 8) Return ---
    if return_normals and compute_curvature:
        return verts, faces, normals, curv
    if return_normals:
        return verts, faces, normals
    if compute_curvature:
        return verts, faces, curv
    return verts, faces



def interactive_isosurface_plot(
    checkpoint,
    grid_size=64,
    level=0.0,

    # ---- normals ----
    compute_normals=False,
    n_cone=0,                  # 0 disables cones; else number of cones
    cone_length=0.06,          # affects cone size (sizeref)
    cone_opacity=0.6,
    norm_direction=1.0,
    normal_eps=1e-8,

    # ---- curvature ----
    compute_curvature=True,
    curvature_kind="kappa",    # "kappa" (=div(n)) or "H" (=kappa/2)
    abs_curvature=False,
    curvature_eps=1e-8,
    colorscale="Viridis",
    clim=None,                 # None or (cmin, cmax)
    show_colorbar=True,
    colorbar_title=None,

    # ---- mesh appearance ----
    mesh_opacity=0.8,
):
    """
    Returns:
      fig, verts, faces, normals_or_None, curv_or_None

    Notes:
      - Assumes NO transpose in phi volume.
      - Uses spacing = 2/(grid_size-1) consistent with linspace(-1,1,grid_size).
      - Curvature uses Hessian formula:
           kappa = tr(H)/||g|| - (g^T H g)/||g||^3
    """

    # --- 1) Grid in [-1,1]^3 ---
    x = jnp.linspace(-1, 1, grid_size)
    y = jnp.linspace(-1, 1, grid_size)
    z = jnp.linspace(-1, 1, grid_size)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    grid_points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)  # (N,3)

    # --- 2) NN phi on grid ---
    model = PINN()
    state = checkpoint["state"]
    params = state["params"]
    phi_fn = lambda pts: model.apply(params, pts)  # (N,3)->(N,) or (N,1)

    phi_vals = phi_fn(grid_points).reshape(grid_size, grid_size, grid_size)

    # skimage marching_cubes needs a writable numpy array
    phi_np = np.asarray(phi_vals, dtype=np.float32).copy()

    # --- 3) Marching cubes ---
    spacing = (2 / (grid_size - 1),) * 3
    verts, faces, _, _ = marching_cubes(phi_np, level=level, spacing=spacing)
    verts = verts - 1.0                 # map [0,2] -> [-1,1]
    faces = faces.astype(np.int32)

    # --- 4) Compute normals / curvature at verts (optional) ---
    verts_j = jnp.asarray(verts)

    normals = None
    curv = None

    if compute_normals or compute_curvature:
        # Gradient
        grads = grad_phi(phi_fn, verts_j)  # (Nv,3)
        gnorm = jnp.linalg.norm(grads, axis=1, keepdims=True)  # (Nv,1)
        if compute_normals:
            normals = np.asarray(norm_direction * grads / (gnorm + normal_eps))

        if compute_curvature:
            H = hessian_phi(phi_fn, verts_j)  # (Nv,3,3)
            trH = jnp.trace(H, axis1=1, axis2=2)              # (Nv,)
            Hg = jnp.einsum("nij,nj->ni", H, grads)           # (Nv,3)
            gHg = jnp.einsum("ni,ni->n", grads, Hg)           # (Nv,)

            G = jnp.squeeze(gnorm, axis=1)                    # (Nv,)
            Gsafe = jnp.sqrt(G * G + curvature_eps**2)

            kappa = trH / Gsafe - gHg / (Gsafe**3)            # (Nv,) = div(n)
            if curvature_kind.lower() == "h":
                kappa = 0.5 * kappa

            curv = np.asarray(kappa)
            if abs_curvature:
                curv = np.abs(curv)

    # --- 5) Plotly mesh ---
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    mesh_kwargs = dict(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=i, j=j, k=k,
        opacity=mesh_opacity,
        name="phi surface",
    )

    # Color by curvature (vertex intensity)
    if curv is not None:
        cmin, cmax = (None, None) if clim is None else clim
        mesh = go.Mesh3d(
            **mesh_kwargs,
            intensity=curv,                 # per-vertex values
            colorscale=colorscale,
            showscale=show_colorbar,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(title=(colorbar_title or ("|kappa|" if abs_curvature else ("kappa" if curvature_kind.lower()=="kappa" else "H")))),
        )
    else:
        mesh = go.Mesh3d(**mesh_kwargs, color="cyan", showscale=False)

    data = [mesh]

    # --- 6) Optional normals as cones (true 3D arrows) ---
    if (n_cone is not None) and (n_cone > 0) and (normals is not None):
        n = verts.shape[0]
        sel = np.linspace(0, n - 1, min(n_cone, n), dtype=int)
        P = verts[sel]
        N = normals[sel]

    # --- normals as line segments ---
    if (n_cone is not None) and (n_cone > 0) and (normals is not None):
        n = verts.shape[0]
        sel = np.linspace(0, n - 1, min(n_cone, n), dtype=int)

        P = verts[sel]
        N = normals[sel]

        L = cone_length   # reuse this as line length

        xs, ys, zs = [], [], []
        for p, nvec in zip(P, N):
            xs.extend([p[0], p[0] + L * nvec[0], None])
            ys.extend([p[1], p[1] + L * nvec[1], None])
            zs.extend([p[2], p[2] + L * nvec[2], None])

        normal_lines = go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(
                color="black",
                width=4,           # thickness
            ),
            opacity=0.4,
            name="normals",
            showlegend=True,
        )

        data.append(normal_lines)


    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(
            aspectmode="data",  # keeps geometry aspect correct
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
    )

    return fig, verts, faces, normals, curv


def face_areas(verts, faces):
    """Triangle areas for each face. verts (Nv,3), faces (Nf,3)."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

def vertex_area_weights(verts, faces):
    """
    Lumped vertex area weights: each triangle area is split equally to its 3 vertices.
    Returns w (Nv,) such that sum(w) = total surface area.
    """
    A = face_areas(verts, faces)  # (Nf,)
    w = np.zeros(len(verts), dtype=float)
    # accumulate A/3 to each vertex of the triangle
    np.add.at(w, faces[:, 0], A / 3.0)
    np.add.at(w, faces[:, 1], A / 3.0)
    np.add.at(w, faces[:, 2], A / 3.0)
    return w, A



def plot_normals_hist(
    normals,
    verts,
    faces,
    ref_dir=(0, 0, 1),
    bins=60,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    # ---- area weights ----
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    w_vert = np.zeros(len(verts))
    np.add.at(w_vert, faces[:, 0], face_areas / 3)
    np.add.at(w_vert, faces[:, 1], face_areas / 3)
    np.add.at(w_vert, faces[:, 2], face_areas / 3)

    ref = np.asarray(ref_dir, float)
    ref /= np.linalg.norm(ref)

    theta = np.arccos(np.clip(normals @ ref, -1, 1)) * 180 / np.pi

    ax.hist(theta, bins=bins, weights=w_vert, density=True)
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel("area-weighted PDF")
    ax.set_title("Normal orientation")



def plot_curvature_hist(
    curv,
    verts,
    faces,
    bins=80,
    abs_val=True,
    clip=None,
    area_weighted=True,
    title="Curvature distribution",
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    k = np.abs(curv) if abs_val else curv

    if clip is not None:
        k = np.clip(k, *clip)

    # if clip is not None:
    #     mask = (k >= clip[0]) & (k <= clip[1])
    #     k = k[mask]
    #     if weights is not None:
    #         weights = weights[mask]


    weights = None
    if area_weighted:
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        A = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

        weights = np.zeros(len(verts))
        np.add.at(weights, faces[:, 0], A / 3)
        np.add.at(weights, faces[:, 1], A / 3)
        np.add.at(weights, faces[:, 2], A / 3)

    ax.hist(k, bins=bins, weights=weights)
    ax.set_xlabel("|κ|")
    ax.set_ylabel("area")
    ax.set_title(title)



def write_vtk_polydata(filename, vertices, faces, point_vectors=None, point_scalars=None):
    """
    Write a legacy VTK POLYDATA file (ASCII) readable by ParaView.

    Parameters
    ----------
    filename : str or Path
    vertices : (N,3) float
    faces : (M,3) int  (triangles)
    point_vectors : dict[str, (N,3)] or None
    point_scalars : dict[str, (N,)] or None
    """
    filename = Path(filename)
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)

    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3), got {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"faces must be (M,3) triangles, got {F.shape}")

    # VTK legacy POLYGONS wants: for each face: "3 i j k"
    # and the POLYGONS header wants total_ints = M*(3+1)
    M = F.shape[0]
    total_ints = M * 4

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Mem3DG mesh export\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # Points
        f.write(f"POINTS {V.shape[0]} float\n")
        for x, y, z in V:
            f.write(f"{x:.9g} {y:.9g} {z:.9g}\n")

        # Polygons
        f.write(f"POLYGONS {M} {total_ints}\n")
        for i, j, k in F:
            f.write(f"3 {int(i)} {int(j)} {int(k)}\n")

        # Point data
        has_vectors = point_vectors is not None and len(point_vectors) > 0
        has_scalars = point_scalars is not None and len(point_scalars) > 0
        if has_vectors or has_scalars:
            f.write(f"\nPOINT_DATA {V.shape[0]}\n")

        if has_vectors:
            for name, vec in point_vectors.items():
                vec = np.asarray(vec, dtype=np.float64)
                if vec.shape != V.shape:
                    raise ValueError(f"vector field '{name}' must be (N,3), got {vec.shape}")
                f.write(f"VECTORS {name} float\n")
                for x, y, z in vec:
                    f.write(f"{x:.9g} {y:.9g} {z:.9g}\n")
                f.write("\n")

        if has_scalars:
            for name, s in point_scalars.items():
                s = np.asarray(s, dtype=np.float64).reshape(-1)
                if s.shape[0] != V.shape[0]:
                    raise ValueError(f"scalar field '{name}' must be (N,), got {s.shape}")
                f.write(f"SCALARS {name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for val in s:
                    # f.write(f"{np.abs(val):.9g}\n")
                    f.write(f"{val:.9g}\n")
                f.write("\n")


import numpy as np
import matplotlib.pyplot as plt

def _area_weights_per_vertex(verts, faces):
    """Area lumping: each triangle area split equally to its 3 vertices."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    w_vert = np.zeros(len(verts), dtype=float)
    np.add.at(w_vert, faces[:, 0], face_areas / 3.0)
    np.add.at(w_vert, faces[:, 1], face_areas / 3.0)
    np.add.at(w_vert, faces[:, 2], face_areas / 3.0)
    return w_vert

def _theta_deg(normals, ref_dir):
    ref = np.asarray(ref_dir, float)
    ref /= np.linalg.norm(ref)
    # normals assumed unit; if not, normalize safely
    n = np.asarray(normals, float)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, n_norm, out=np.zeros_like(n), where=(n_norm > 0))
    return np.degrees(np.arccos(np.clip(n @ ref, -1.0, 1.0)))

def plot_normals_hist_2exp_vs_gt(
    exp1_normals, exp1_verts, exp1_faces,
    exp2_normals, exp2_verts, exp2_faces,
    gt_normals,  gt_verts,  gt_faces,
    ref_dir=(0, 0, 1),
    bins=60,
    ax=None,
    labels=("exp 1", "exp 2", "ground truth"),
    hist_alpha=0.35,
    title="Normal Orientation",
    show_legend=True,
):
    """
    Plot two experiments as area-weighted histograms + ground truth as smooth curve.

    Notes:
      - Uses vertex-lumped area weights derived from faces.
      - Histograms are density=True (area-weighted PDF).
      - Ground-truth curve is computed from an area-weighted histogram and drawn
        at bin centers (simple smoothing via convolution).
    """
    if ax is None:
        fig, ax = plt.subplots()

    # ---- compute theta + weights ----
    w1 = _area_weights_per_vertex(exp1_verts, exp1_faces)
    t1 = _theta_deg(exp1_normals, ref_dir)

    w2 = _area_weights_per_vertex(exp2_verts, exp2_faces)
    t2 = _theta_deg(exp2_normals, ref_dir)

    wgt = _area_weights_per_vertex(gt_verts, gt_faces)
    tgt = _theta_deg(gt_normals, ref_dir)

    # ---- shared bin edges (for comparable PDFs) ----
    tmin = min(t1.min(initial=0), t2.min(initial=0), tgt.min(initial=0))
    tmax = max(t1.max(initial=180), t2.max(initial=180), tgt.max(initial=180))
    edges = np.linspace(tmin, tmax, int(bins) + 1)

    # ---- experiments: histograms ----
    ax.hist(t1, bins=edges, weights=w1, density=True, alpha=hist_alpha,
            label=labels[0], histtype="stepfilled")
    ax.hist(t2, bins=edges, weights=w2, density=True, alpha=hist_alpha,
            label=labels[1], histtype="stepfilled")

    # ---- ground truth: smooth curve from weighted histogram ----
    pdf_gt, _ = np.histogram(tgt, bins=edges, weights=wgt, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # light smoothing (moving average). increase window for smoother curve.
    win = max(1, int(len(pdf_gt) * 0.05))  # ~5% of bins
    if win % 2 == 0:
        win += 1
    if win > 1:
        kernel = np.ones(win) / win
        pdf_gt_smooth = np.convolve(pdf_gt, kernel, mode="same")
    else:
        pdf_gt_smooth = pdf_gt

    ax.plot(centers, pdf_gt_smooth, label=labels[2], color="black")

    xticks_deg = [0, 90, 180]
    ax.set_xticks(xticks_deg)
    # ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_xticklabels([r"$-\pi/2$", r"$0$", r"$\pi/2$"])
    ax.set_xlabel(r"$\theta$")


    ax.set_xlabel(r"$\theta$ (deg)")
    # ax.set_ylabel("PDF")
    ax.set_title(title)
    if show_legend:
        ax.legend(frameon=False)
    return ax


# def load_mesh_npz(npz_path, keys=("verts", "faces", "norms", "curvs")):
#     dat = np.load(npz_path, allow_pickle=True)
#     verts = dat[keys[0]]
#     faces = dat[keys[1]]
#     norms = dat[keys[2]]
#     curvs = dat[keys[3]]
#     return verts, faces, norms, curvs

def load_mesh_npz(npz_path, keys=("verts", "faces", "norms", "theta", "curvs", "gauss", "force")):
    dat = np.load(npz_path, allow_pickle=True)
    verts = dat[keys[0]]
    faces = dat[keys[1]]
    norms = dat[keys[2]]
    theta = dat[keys[3]]
    curvs = dat[keys[4]]
    gauss = dat[keys[5]]
    force = dat[keys[6]]
    return verts, faces, norms, theta, curvs, gauss, force



import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def match_nearest_and_plot(
    verts_A, vals_A,
    verts_B, vals_B,
    max_dist=None,          # e.g., 0.02 (same units as verts). None = keep all.
    mutual=False,           # True = keep only mutual nearest neighbors
    ax=None,
    title="A vs B (nearest-neighbor matched)",
    xlim=None, ylim=None,
):
    """
    Nearest-neighbor match A -> B (optionally mutual), then scatter/correlation plot.

    Parameters
    ----------
    verts_A : (N,3) array
    vals_A  : (N,) array
    verts_B : (M,3) array
    vals_B  : (M,) array
    max_dist : float or None
        If set, drop pairs with NN distance > max_dist.
    mutual : bool
        If True, keep only mutual NN pairs (A's NN in B, and that B point's NN in A is the same A).
    ax : matplotlib axis or None
    title : str

    Returns
    -------
    out : dict with keys
        idx_B_for_A : (N,) int indices in B for each A (before filtering)
        dist_A_to_B : (N,) float NN distances (before filtering)
        mask        : (N,) bool mask of kept A points after filtering
        A_kept      : matched A values
        B_kept      : matched B values
        corr        : Pearson correlation on kept pairs
    """
    verts_A = np.asarray(verts_A, dtype=float)
    vals_A  = np.asarray(vals_A, dtype=float).reshape(-1)
    verts_B = np.asarray(verts_B, dtype=float)
    vals_B  = np.asarray(vals_B, dtype=float).reshape(-1)

    if verts_A.ndim != 2 or verts_A.shape[1] != 3:
        raise ValueError(f"verts_A must be (N,3); got {verts_A.shape}")
    if verts_B.ndim != 2 or verts_B.shape[1] != 3:
        raise ValueError(f"verts_B must be (M,3); got {verts_B.shape}")
    if vals_A.shape[0] != verts_A.shape[0]:
        raise ValueError(f"vals_A must have length N={verts_A.shape[0]}; got {vals_A.shape[0]}")
    if vals_B.shape[0] != verts_B.shape[0]:
        raise ValueError(f"vals_B must have length M={verts_B.shape[0]}; got {vals_B.shape[0]}")

    # A -> B nearest neighbor
    tree_B = cKDTree(verts_B)
    dist_A_to_B, idx_B_for_A = tree_B.query(verts_A, k=1, workers=-1)

    mask = np.ones(len(verts_A), dtype=bool)
    if max_dist is not None:
        mask &= (dist_A_to_B <= float(max_dist))

    if mutual:
        # For each B, find its nearest A
        tree_A = cKDTree(verts_A)
        _, idx_A_for_B = tree_A.query(verts_B, k=1, workers=-1)

        # Mutual means: A_i -> B_j and B_j -> A_i
        mutual_mask = (idx_A_for_B[idx_B_for_A] == np.arange(len(verts_A)))
        mask &= mutual_mask

    A_kept = vals_A[mask]
    B_kept = vals_B[idx_B_for_A[mask]]

    if len(A_kept) < 2:
        raise RuntimeError(
            f"Not enough matched points after filtering (kept={len(A_kept)}). "
            f"Try increasing max_dist or turning off mutual."
        )

    # Pearson correlation
    corr = float(np.corrcoef(A_kept, B_kept)[0, 1])

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 4.2))

    ax.scatter(A_kept, B_kept, s=8, alpha=0.6)
    ax.set_xlabel("A values")
    ax.set_ylabel("B values (NN matched)")
    ax.set_title(f"{title}\nPearson r = {corr:.3f} (n={len(A_kept)})")

    # Optional y=x reference line based on data range
    lo = np.nanmin([A_kept.min(), B_kept.min()])
    hi = np.nanmax([A_kept.max(), B_kept.max()])
    # ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.plot(
        [lo, hi], [lo, hi],
        linestyle="--",
        color="black",      # ← force black
        linewidth=1.5,      # optional: thicker
        zorder=3            # optional: draw on top of points
    )

    ax.set_aspect("equal" if np.isfinite(lo) and np.isfinite(hi) else "auto")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_aspect("auto")

    return {
        "idx_B_for_A": idx_B_for_A,
        "dist_A_to_B": dist_A_to_B,
        "mask": mask,
        "A_kept": A_kept,
        "B_kept": B_kept,
        "corr": corr,
    }


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


import numpy as np
from scipy.spatial import cKDTree

def match_nearest_neighbor(
    verts_A, vals_A,
    verts_B, vals_B,
    *,
    max_dist=None,
    mutual=False,
    workers=-1,
):
    """
    Nearest-neighbor matching from A -> B with optional distance cutoff and mutual NN.

    Returns dict:
        A_kept, B_kept, corr, mask, dist, idx_B
    """
    tree_B = cKDTree(verts_B)
    dist, idx_B = tree_B.query(verts_A, k=1, workers=workers)

    mask = np.ones(len(verts_A), dtype=bool)
    if max_dist is not None:
        mask &= dist <= max_dist

    if mutual:
        tree_A = cKDTree(verts_A)
        _, idx_A_back = tree_A.query(verts_B, k=1, workers=workers)
        mask &= (idx_A_back[idx_B] == np.arange(len(verts_A)))

    A_kept = vals_A[mask]
    B_kept = vals_B[idx_B[mask]]

    if len(A_kept) < 2:
        raise RuntimeError("Too few matched points after filtering.")

    corr = float(np.corrcoef(A_kept, B_kept)[0, 1])

    return corr



def match_and_plot_two_data(
    verts_A, vals_A,
    verts_B1, vals_B1,
    verts_B2, vals_B2,
    max_dist=None,
    mutual=False,
    xlim=None,
    ylim=None,
    aspect="auto",
    box_aspect=None,
    labels=("B1", "B2"),
    colors=("tab:blue", "tab:orange"),
    title="A vs B (nearest-neighbor matched)",
    xlabel="A values",
    ylabel="B values (NN matched)",
    ax=None,
    figsize=(4.5, 4.5),
):
    """
    Nearest-neighbor matching for A-B1 and A-B2, plotted together.

    Returns
    -------
    out : dict with keys
        'B1', 'B2' → each contains:
            A_kept, B_kept, corr, mask, dist
    """

    def _match(verts_A, vals_A, verts_B, vals_B):
        tree_B = cKDTree(verts_B)
        dist, idx_B = tree_B.query(verts_A, k=1, workers=-1)

        mask = np.ones(len(verts_A), dtype=bool)
        if max_dist is not None:
            mask &= dist <= max_dist

        if mutual:
            tree_A = cKDTree(verts_A)
            _, idx_A_back = tree_A.query(verts_B, k=1, workers=-1)
            mask &= (idx_A_back[idx_B] == np.arange(len(verts_A)))

        A_kept = vals_A[mask]
        B_kept = vals_B[idx_B[mask]]

        if len(A_kept) < 2:
            raise RuntimeError("Too few matched points after filtering.")

        corr = float(np.corrcoef(A_kept, B_kept)[0, 1])

        return {
            "A_kept": A_kept,
            "B_kept": B_kept,
            "corr": corr,
            "mask": mask,
            "dist": dist,
        }

    # ---- Run matching ----
    out_B1 = _match(verts_A, vals_A, verts_B1, vals_B1)
    out_B2 = _match(verts_A, vals_A, verts_B2, vals_B2)

    # ---- Plot ----
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        out_B1["A_kept"], out_B1["B_kept"],
        s=2, alpha=1, color=colors[0],
        label=f"{labels[0]} (r={out_B1['corr']:.3f})",
        zorder=2, edgecolors="none",
        rasterized=True,
    )

    ax.scatter(
        out_B2["A_kept"], out_B2["B_kept"],
        s=2, alpha=1, color=colors[1],
        label=f"{labels[1]} (r={out_B2['corr']:.3f})",
        zorder=2, edgecolors="none",
        rasterized=True,
    )

    # ---- Identity line ----
    all_vals = np.concatenate([
        out_B1["A_kept"], out_B1["B_kept"],
        out_B2["B_kept"]
    ])
    lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)

    ax.plot(
        [lo, hi], [lo, hi],
        "--", color="black", #linewidth=1.5, zorder=5
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_aspect(aspect)
    if box_aspect is not None:
        ax.set_box_aspect(box_aspect)

    # ax.legend(frameon=False, handletextpad=-0.5, loc="upper left", bbox_to_anchor=(-0.1, 1.0),)

    handles, labels = ax.get_legend_handles_labels()
    # fig.legend(
    #     handles, labels,
    #     loc="lower center",
    #     bbox_to_anchor=(0.4, -0.15),  # move up/down
    #     ncol=2,
    #     frameon=False,
    #     handletextpad=-0.5,
    #     columnspacing=-0.5,
    # )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.45, -0.26),  # x=center, y=below axes
        frameon=False,
        ncol=1,                      # often nice below plots
        handletextpad=-0.5,
        columnspacing=0.0,
        labelspacing=0.0,
    )
    # fig.subplots_adjust(bottom=0.25)   # try 0.25–0.35
    # fig.subplots_adjust(bottom=0.3)   # try 0.25–0.35

    return fig 



import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def match_and_plot_multi_data(
    verts_A,
    vals_A,
    B_datasets,
    max_dist=None,
    mutual=False,
    xlim=None,
    ylim=None,
    aspect="auto",
    box_aspect=None,
    labels=None,
    colors=None,
    title=None,
    xlabel="A values",
    ylabel="B values (NN matched)",
    ax=None,
    figsize=(4.5, 4.5),
    s=2,
    alpha=1,
):
    """
    Nearest-neighbor matching for A against multiple B datasets, plotted together.

    Parameters
    ----------
    verts_A : (N, d) array
        Coordinates for dataset A.
    vals_A : (N,) array
        Values for dataset A.
    B_datasets : list of tuple
        Each element should be (verts_B, vals_B).
        Example:
            [
                (verts_B1, vals_B1),
                (verts_B2, vals_B2),
                (verts_B3, vals_B3),
            ]
    max_dist : float or None
        Keep only matches with distance <= max_dist.
    mutual : bool
        If True, enforce mutual nearest-neighbor matching.
    labels : list of str or None
        Labels for each B dataset.
    colors : list of color specs or None
        Colors for each B dataset.

    Returns
    -------
    fig : matplotlib.figure.Figure
    out : dict
        Keys are labels, values are dicts containing:
            A_kept, B_kept, corr, mask, dist
    """

    def _match(verts_A, vals_A, verts_B, vals_B):
        tree_B = cKDTree(verts_B)
        dist, idx_B = tree_B.query(verts_A, k=1, workers=-1)

        mask = np.ones(len(verts_A), dtype=bool)
        if max_dist is not None:
            mask &= dist <= max_dist

        if mutual:
            tree_A = cKDTree(verts_A)
            _, idx_A_back = tree_A.query(verts_B, k=1, workers=-1)
            mask &= (idx_A_back[idx_B] == np.arange(len(verts_A)))

        A_kept = vals_A[mask]
        B_kept = vals_B[idx_B[mask]]

        if len(A_kept) < 2:
            raise RuntimeError("Too few matched points after filtering.")

        corr = float(np.corrcoef(A_kept, B_kept)[0, 1])

        return {
            "A_kept": A_kept,
            "B_kept": B_kept,
            "corr": corr,
            "mask": mask,
            "dist": dist,
        }

    nB = len(B_datasets)

    if labels is None:
        labels = [f"B{i+1}" for i in range(nB)]
    if colors is None:
        colors = [None] * nB

    if len(labels) != nB:
        raise ValueError("labels must have the same length as B_datasets")
    if len(colors) != nB:
        raise ValueError("colors must have the same length as B_datasets")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    out = {}

    all_vals = []

    for (verts_B, vals_B), label, color in zip(B_datasets, labels, colors):
        result = _match(verts_A, vals_A, verts_B, vals_B)
        out[label] = result

        ax.scatter(
            result["A_kept"],
            result["B_kept"],
            s=s,
            alpha=alpha,
            color=color,
            label=f"{label} (r={result['corr']:.3f})",
            zorder=2,
            edgecolors="none",
            rasterized=True,
        )

        all_vals.append(result["A_kept"])
        all_vals.append(result["B_kept"])

    # ---- Identity line ----
    all_vals = np.concatenate(all_vals)
    lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)

    ax.plot(
        [lo, hi], [lo, hi],
        "--", color="black"
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_aspect(aspect)
    if box_aspect is not None:
        ax.set_box_aspect(box_aspect)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.45, -0.26),
        frameon=False,
        ncol=1,
        handletextpad=-0.5,
        columnspacing=0.0,
        labelspacing=0.0,
    )

    return fig, out




def plot_ablation(
    metric_dict,
    shape_list,
    phase_list,
    x_values,
    ylabel="Metric",
    xlabel="x_values",
    title=None,
    figsize=(6, 4),
    marker="o",
    # linewidth=1.8,
    markersize=2,
    show_legend=True,
    save_path=None,
    ylim = None,
    gap_after_first = False,
    xtick_labels=None,
):
    # equally spaced x positions
    x = np.arange(len(x_values))

    # colors for shapes
    # cmap = plt.get_cmap("tab10")
    # shape_colors = {shape: cmap(i) for i, shape in enumerate(shape_list)}

    shape_colors = {
        "biconcave": "tab:red",
        "bud_04": "tab:blue",
    }

    # line styles for phases
    linestyle_map = {
        0: ":",
        1: "--",
        2: "-",
    }

    fig, ax = plt.subplots(figsize=figsize)

    if gap_after_first:
        ax.axvline(0.5, linestyle="--", color="black", alpha=1.0, linewidth=0.5)

    for shape in shape_list:
        for phase in phase_list:
            # y = [metric_dict[shape][phase][x_val] for x_val in x_values]
            y = [metric_dict[shape][phase].get(x_val, np.nan) for x_val in x_values]


            ax.plot(
                x,
                y,
                color=shape_colors[shape],
                linestyle=linestyle_map.get(phase, "-"),
                marker=marker,
                # linewidth=linewidth,
                markersize=markersize,
                label=f"{shape}, phase {phase}",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in x_values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xtick_labels is not None:
        ax.set_xticklabels(xtick_labels)


    if ylim is not None:
        ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)

    if show_legend:
        # ax.legend(frameon=False, fontsize=9)
        ax.legend(frameon=False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

