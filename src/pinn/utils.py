import numpy as np
from pinn.model import loss_data, loss_physics, total_loss
import jax
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt

def initial_loss_ori(state, x_train, cryoET_data):
# def initial_loss(state, x_train, cryoET_data, membrane_indices):
    """
    Compute the actual initial loss values before training starts.
    Returns a structured dictionary compatible with checkpoint storage.
    """

    # Define the function using current model parameters
    params = state.params
    phi_fn = lambda x: state.apply_fn(params, x.reshape(-1, 3))

    # Compute the losses
    loss_data_val = loss_data(phi_fn, cryoET_data)
    # loss_data_val = loss_data(phi_fn, cryoET_data, membrane_indices)
    loss_physics_val = loss_physics(phi_fn, x_train)
    total_loss_val = total_loss(phi_fn, x_train, cryoET_data, state.lambda_1, state.lambda_2)
    # total_loss_val = total_loss(phi_fn, x_train, cryoET_data, state.lambda_1, state.lambda_2, membrane_indices)

    # Convert to structured format (single-step array)
    loss_log = {
        "step": np.array([0]),  # Ensure consistency with later steps
        "total_loss": np.array([total_loss_val]),
        "data_loss": np.array([loss_data_val]),
        "physics_loss": np.array([loss_physics_val])
    }

    return loss_log


from pinn.model import loss_sign


def initial_loss(state, x_train, data_edge, data_sign):
    """
    Compute the actual initial loss values before training starts.
    Returns a structured dictionary compatible with checkpoint storage.
    """

    # Define the function using current model parameters
    params = state.params
    phi_fn = lambda x: state.apply_fn(params, x.reshape(-1, 3))

    # Compute the losses
    loss_data_val = loss_data(phi_fn, data_edge)
    loss_physics_val = loss_physics(phi_fn, x_train)
    loss_sign_val = loss_sign(phi_fn, data_sign)
    total_loss_val = state.lambda_1 * loss_data_val + state.lambda_2 * loss_physics_val + state.lambda_3 * loss_sign_val

    # Convert to structured format (single-step array)
    loss_log = {
        "step": np.array([0]),  # Ensure consistency with later steps
        "total_loss": np.array([total_loss_val]),
        "data_loss": np.array([loss_data_val]),
        "physics_loss": np.array([loss_physics_val]),
        "sing_loss": np.array([loss_sign_val])
    }

    return loss_log



def initial_loss_ori2(state, x_train, cryoET_data, data_sign):
# def initial_loss(state, x_train, cryoET_data, membrane_indices):
    """
    Compute the actual initial loss values before training starts.
    Returns a structured dictionary compatible with checkpoint storage.
    """

    # Define the function using current model parameters
    params = state.params
    phi_fn = lambda x: state.apply_fn(params, x.reshape(-1, 3))

    # Compute the losses
    loss_data_val = loss_data(phi_fn, cryoET_data)
    loss_physics_val = loss_physics(phi_fn, x_train)
    loss_sign_val = loss_sign(phi_fn, data_sign)
    total_loss_val = state.lambda_1 * loss_data_val + state.lambda_2 * loss_physics_val + state.lambda_3 * loss_sign_val

    # Convert to structured format (single-step array)
    loss_log = {
        "step": np.array([0]),  # Ensure consistency with later steps
        "total_loss": np.array([total_loss_val]),
        "data_loss": np.array([loss_data_val]),
        "physics_loss": np.array([loss_physics_val]),
        "sing_loss": np.array([loss_sign_val])
    }

    return loss_log


def assemble_loss_history(checkpoint_data):
    """
    Assemble loss history from all available checkpoints.

    Args:
        checkpoint_data (dict): Dictionary containing loss values from multiple checkpoints.

    Returns:
        dict: Aggregated loss history with 'step', 'total_loss', 'data_loss', and 'physics_loss'.
    """
    # assembled_loss = {"step": [], "total_loss": [], "data_loss": [], "physics_loss": []}
    assembled_loss = {"step": [], "total_loss": [], "data_loss": [], "physics_loss": [], "sign_loss": []}

    for step, checkpoint in checkpoint_data.items():
        if 'loss' in checkpoint:
            loss_data = checkpoint['loss']
            assembled_loss["step"].extend(loss_data["step"].tolist())
            assembled_loss["total_loss"].extend(loss_data["total_loss"].tolist())
            assembled_loss["data_loss"].extend(loss_data["data_loss"].tolist())
            assembled_loss["physics_loss"].extend(loss_data["physics_loss"].tolist())
            assembled_loss["sign_loss"].extend(loss_data["sign_loss"].tolist())

    # Convert lists to NumPy arrays for easier handling
    for key in assembled_loss:
        assembled_loss[key] = np.array(assembled_loss[key])

    return assembled_loss


class _PINN(nn.Module):
    hidden_dim: int = 16

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)  # Ensure input is at least 2D
        if x.shape[-1] != 3:
            raise ValueError(f"Expected input shape (*,3), but got {x.shape}")

        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)

        x = nn.Dense(1)(x)  # Output single scalar φ(x,y,z)
        x = nn.tanh(x)

        return x.squeeze()


def _phi_importance_weights(phi, mode="exp", beta=6.0, power=2.0, eps=1e-8):
    """
    phi: (M,) in [-1, 1]
    returns positive weights (M,)
    """
    a = jnp.clip(jnp.abs(phi), 0.0, 1.0)

    if mode == "exp":
        # bigger beta => more concentrated near phi=0
        w = jnp.exp(-beta * a)
    elif mode == "poly":
        # bigger power => more concentrated near phi=0
        w = jnp.power(1.0 - a + eps, power)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'exp' or 'poly'.")

    return jnp.maximum(w, eps)


def sample_collocation_points(
    key,
    checkpoint,
    n_points: int,
    *,
    bounds=(-1.0, 1.0),
    oversample: int = 20,
    mode: str = "exp",
    beta: float = 3.0,
    power: float = 2.0,
    uniform_frac: float = 0.1,
    perm = (2,1,0),
):
    """
    Samples collocation points in [-1,1]^3 (or custom bounds) with preference near phi≈0.

    Strategy:
      1) draw M = oversample * n_points candidates uniformly
      2) compute weights from phi candidates (higher near phi=0)
      3) sample (1-uniform_frac)*n_points by weighted choice + uniform_frac*n_points uniformly

    Returns:
      points: (n_points, 3)
      info: dict with candidate phi/weights (useful for debugging/plotting)
    """

    state = checkpoint["state"]
    params = state["params"]
    model = _PINN()
    phi_fn = lambda x: model.apply(params, x)

    lo, hi = bounds
    key_cand, key_pick, key_uni = jax.random.split(key, 3)

    M = int(oversample * n_points)
    cand = jax.random.uniform(key_cand, (M, 3), minval=lo, maxval=hi)

    # Evaluate phi on candidates
    phi_cand = phi_fn(cand)  # (M,)
    phi_cand = jnp.asarray(phi_cand).reshape(-1)

    # Importance weights (bigger near phi=0)
    w = _phi_importance_weights(phi_cand, mode=mode, beta=beta, power=power)
    p = w / jnp.sum(w)

    n_uni = int(jnp.round(uniform_frac * n_points))
    n_imp = n_points - n_uni

    # Weighted sample WITHOUT replacement (good default)
    idx = jax.random.choice(key_pick, M, shape=(n_imp,), replace=False, p=p)
    pts_imp = cand[idx]

    # Add a small fraction of uniform samples for coverage (optional)
    if n_uni > 0:
        pts_uni = jax.random.uniform(key_uni, (n_uni, 3), minval=lo, maxval=hi)
        points = jnp.concatenate([pts_imp, pts_uni], axis=0)
    else:
        points = pts_imp

    # Shuffle final points
    key_shuf = jax.random.fold_in(key, 12345)
    pts = jax.random.permutation(key_shuf, points, axis=0)

    pts = pts[:, perm] # Since phi_fn is (Z,Y,X), saving data in (x,y,z) style

    return {
        "points": pts
    }



def sample_sign_points(
    n_sample=1000,
    threshold=0.2,
    checkpoint=None,
    max_iter=20,
    key=jax.random.PRNGKey(1234),
    perm=(2, 1, 0),
):

    state = checkpoint["state"]
    params = state["params"]
    model = _PINN()
    phi_fn = lambda x: model.apply(params, x)

    pts_list = []
    sign_list = []

    remaining = n_sample
    k = key

    for _ in range(max_iter):
        if remaining <= 0:
            break

        k, subk = jax.random.split(k)
        x_try = jax.random.uniform(
            subk, (remaining * 2, 3), minval=-1.0, maxval=1.0
        )

        phi_val = phi_fn(x_try)

        pos_mask = phi_val > threshold
        neg_mask = phi_val < -threshold
        accept_mask = pos_mask | neg_mask

        x_acc = x_try[accept_mask]
        phi_acc = phi_val[accept_mask]

        if x_acc.shape[0] == 0:
            continue

        sign_acc = jnp.where(phi_acc > 0, 1.0, -1.0)

        pts_list.append(x_acc)
        sign_list.append(sign_acc)

        remaining -= x_acc.shape[0]

    if remaining > 0:
        print(f"[Warning] Only collected {n_sample - remaining}/{n_sample} sign anchors")

    pts = jnp.concatenate(pts_list, axis=0)[:n_sample]
    sign = jnp.concatenate(sign_list, axis=0)[:n_sample]

    pts = pts[:, perm] # Since phi_fn is (Z,Y,X), saving data in (x,y,z) style

    return {
        "points": pts,
        "label" : sign,
    }

def save_pts_data(data, path, meta=None):
    """
    Required:
      data["points"] : (N, 3)

    Optional:
      data["label"]  : (N,)   # sign, edge, etc
      meta           : dict   # parameters etc
    """
    out = {
        "points": np.asarray(data["points"]),
    }

    if "label" in data:
        out["label"] = np.asarray(data["label"])

    if meta is not None:
        # store metadata as a pickled object
        out["meta"] = np.array(meta, dtype=object)

    np.savez(path, **out)


def load_pts_data(path, perm=(0, 1, 2)):
    """
    Returns:
      data : dict with keys
        - "points" : (N, 3)
        - "label"  : (N,) or None
        - "meta"   : dict or None
    """
    npz = np.load(path, allow_pickle=True)

    pts = npz["points"]
    pts = pts[:, perm]

    data = {
        "points": pts,
        "label": None,
        "meta": None,
    }

    if "label" in npz.files:
        data["label"] = npz["label"]

    if "meta" in npz.files:
        data["meta"] = npz["meta"].item()

    return data



def load_sign_data(path, perm=(0, 1, 2), flip=(1, 1, 1)):

    data = np.load(path)

    pts = np.asarray(data["points"], dtype=np.float32)
    sgn = np.asarray(data["sign"], dtype=np.float32)

    # permute axes
    pts = pts[:, perm]

    # flip axes if needed
    pts = pts * np.asarray(flip, dtype=np.float32)

    return {
        "points": jnp.asarray(pts),
        "sign":   jnp.asarray(sgn),
    }

    # data = np.load(path)
    # return {
    #     "points": jnp.asarray(data["points"]),
    #     "sign": jnp.asarray(data["sign"]),
    # }

def load_edge_data(path, perm=(0, 1, 2), flip=(1, 1, 1)):

    data = np.load(path)

    pts = np.asarray(data["points"], dtype=np.float32)
    edg = np.asarray(data["edge"], dtype=np.float32)

    # permute axes
    pts = pts[:, perm]

    # flip axes if needed
    pts = pts * np.asarray(flip, dtype=np.float32)

    return {
        "points": jnp.asarray(pts),
        "edge":   jnp.asarray(edg),
    }


import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def visualize_sign_sampling(
    grid_size=64,
    slice_index=None,
    slice_thickness=None,
    checkpoint=None,
    signs=None,
    vmin=-1.0,
    vmax=1.0,
    s=12,
    alpha_pts=0.9,
    figsize=(15, 5),
    title=None,
):
    """
    Show x-, y-, z-slices of phi side-by-side, with sampled sign anchors overlaid.

    IMPORTANT CONVENTION:
      - phi_fn expects input points in (z, y, x) order.
      - signs["points"] are assumed to be in (x, y, z) order in [-1,1]^3.

    Slices:
      - x-slice: yz plane at x = x[slice_index]
      - y-slice: xz plane at y = y[slice_index]
      - z-slice: xy plane at z = z[slice_index]
    """
    assert checkpoint is not None, "Please provide checkpoint"
    assert signs is not None and "points" in signs and "sign" in signs, \
        "Please provide signs dict with 'points' and 'sign'"

    # --- build phi_fn (expects z,y,x) ---
    state = checkpoint["state"]
    params = state["params"]
    model = _PINN()

    def phi_fn(p_zyx):
        out = model.apply(params, p_zyx)
        # robust flatten: handles (N,), (N,1), (N,...) -> (N,)
        return jnp.reshape(out, (out.shape[0],))

    # --- grid coordinates in xyz for plotting semantics ---
    x = jnp.linspace(-1.0, 1.0, grid_size)
    y = jnp.linspace(-1.0, 1.0, grid_size)
    z = jnp.linspace(-1.0, 1.0, grid_size)

    # Build grid in xyz, then feed model as zyx
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")  # (grid,grid,grid) with axes (x,y,z)
    grid_xyz = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)  # (N,3) in (x,y,z)
    grid_zyx = grid_xyz[:, (2, 1, 0)]                                # (N,3) in (z,y,x)

    # Evaluate phi on grid (model-native order)
    phi_vals = phi_fn(grid_zyx)  # (N,)
    phi_3d = phi_vals.reshape((grid_size, grid_size, grid_size))  # indexed as (x_idx,y_idx,z_idx)

    # --- slice settings ---
    if slice_index is None:
        slice_index = grid_size // 2
    slice_index = int(slice_index)

    if slice_thickness is None:
        slice_thickness = 2.0 / (grid_size - 1)  # ~one voxel in coordinate units

    pts_xyz = np.asarray(signs["points"], dtype=np.float32)
    sgn = np.asarray(signs["sign"], dtype=np.float32)

    # --- helper to plot one panel ---
    def _panel(ax, img, xlabel, ylabel, px, py, ps, panel_title):
        im = ax.imshow(
            img.T,
            origin="lower",
            extent=[-1, 1, -1, 1],
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            cmap="coolwarm",
        )

        pos = ps > 0
        neg = ps < 0
        if np.any(pos):
            ax.scatter(px[pos], py[pos], s=s, marker="o", alpha=alpha_pts, label="+1")
        if np.any(neg):
            ax.scatter(px[neg], py[neg], s=s, marker="x", alpha=alpha_pts, label="-1")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        return im

    # --- prepare slices + anchors (xyz semantics) ---
    x_coord = float(x[slice_index])
    y_coord = float(y[slice_index])
    z_coord = float(z[slice_index])

    # X-slice: yz at fixed x -> img is (Y,Z)
    img_x = np.asarray(phi_3d[slice_index, :, :])
    mask_x = np.abs(pts_xyz[:, 0] - x_coord) <= slice_thickness
    px_x, py_x, ps_x = pts_xyz[mask_x, 1], pts_xyz[mask_x, 2], sgn[mask_x]

    # Y-slice: xz at fixed y -> img is (X,Z)
    img_y = np.asarray(phi_3d[:, slice_index, :])
    mask_y = np.abs(pts_xyz[:, 1] - y_coord) <= slice_thickness
    px_y, py_y, ps_y = pts_xyz[mask_y, 0], pts_xyz[mask_y, 2], sgn[mask_y]

    # Z-slice: xy at fixed z -> img is (X,Y)
    img_z = np.asarray(phi_3d[:, :, slice_index])
    mask_z = np.abs(pts_xyz[:, 2] - z_coord) <= slice_thickness
    px_z, py_z, ps_z = pts_xyz[mask_z, 0], pts_xyz[mask_z, 1], sgn[mask_z]

    # --- plot ---
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    _panel(
        axes[0], img_x, xlabel="y", ylabel="z",
        px=px_x, py=py_x, ps=ps_x,
        panel_title=f"x-slice (x={x_coord:+.3f}) | n={int(mask_x.sum())}"
    )
    _panel(
        axes[1], img_y, xlabel="x", ylabel="z",
        px=px_y, py=py_y, ps=ps_y,
        panel_title=f"y-slice (y={y_coord:+.3f}) | n={int(mask_y.sum())}"
    )
    _panel(
        axes[2], img_z, xlabel="x", ylabel="y",
        px=px_z, py=py_z, ps=ps_z,
        panel_title=f"z-slice (z={z_coord:+.3f}) | n={int(mask_z.sum())}"
    )

    # one legend
    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=True)

    if title is None:
        title = f"phi slices at slice_index={slice_index} (anchors within ±{slice_thickness:.3f}) | model input=(z,y,x)"
    fig.suptitle(title, y=1.02)

    plt.tight_layout()
    plt.show()

    return {
        "phi_3d": phi_3d,
        "slice_index": slice_index,
        "slice_coords": {"x": x_coord, "y": y_coord, "z": z_coord},
        "n_points_in_slices": {"x": int(mask_x.sum()), "y": int(mask_y.sum()), "z": int(mask_z.sum())},
    }



def visualize_sign_sampling_mrc(
    grid_size=64,
    slice_index=32,
    cryoET_data=None,          # 3D volume (Z,Y,X) recommended
    signs=None,                # {"points": (N,3) in [-1,1]^3 (x,y,z), "sign": (N,) in {-1,+1}}
    slice_thickness_vox=0.75,  # points within this distance (in voxels) are shown on each slice
    vmin=None,
    vmax=None,
    use_percentile=True,
    p_lo=2,
    p_hi=98,
    figsize=(15, 5),
    title=None,
    s=18,
    alpha_pts=0.9,
):
    """
    Visualize manual sign anchors overlaid on the *cryoET intensity* volume, showing
    x/y/z slices side-by-side at the same slice_index.

    Assumes cryoET_data is shaped (Z, Y, X) and signs["points"] are normalized (x,y,z) in [-1,1]^3.
    """
    assert cryoET_data is not None, "Please pass cryoET_data (3D volume)"
    assert signs is not None and "points" in signs and "sign" in signs, "Please pass signs dict"

    vol = np.asarray(cryoET_data)
    assert vol.ndim == 3, f"cryoET_data must be 3D, got shape {vol.shape}"
    Z, Y, X = vol.shape

    # Optional: sanity check vs grid_size
    if grid_size is not None and (Z != grid_size or Y != grid_size or X != grid_size):
        print(f"[Warning] cryoET_data shape {vol.shape} != (grid_size,grid_size,grid_size)=({grid_size},{grid_size},{grid_size})")

    # Contrast limits
    if use_percentile and (vmin is None or vmax is None):
        vmin = np.percentile(vol, p_lo) if vmin is None else vmin
        vmax = np.percentile(vol, p_hi) if vmax is None else vmax

    # Convert signs points [-1,1] -> voxel indices
    pts = np.asarray(signs["points"], dtype=np.float32)  # (N,3) in (x,y,z)
    sgn = np.asarray(signs["sign"], dtype=np.float32)

    ix = (pts[:, 0] + 1.0) * 0.5 * (X - 1)
    iy = (pts[:, 1] + 1.0) * 0.5 * (Y - 1)
    iz = (pts[:, 2] + 1.0) * 0.5 * (Z - 1)

    # Masks for each slice
    slice_index = int(slice_index)
    mz = np.abs(iz - slice_index) <= slice_thickness_vox
    my = np.abs(iy - slice_index) <= slice_thickness_vox
    mx = np.abs(ix - slice_index) <= slice_thickness_vox

    def _scatter(ax, px, py, ps):
        pos = ps > 0
        neg = ps < 0
        if np.any(pos):
            ax.scatter(px[pos], py[pos], s=s, marker="o", alpha=alpha_pts, label="+1")
        if np.any(neg):
            ax.scatter(px[neg], py[neg], s=s, marker="x", alpha=alpha_pts, label="-1")

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", frameon=True)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- Z slice: (Y,X), overlay (x,y) = (ix,iy)
    img_z = vol[slice_index, :, :]           # (Y,X)
    axes[2].imshow(img_z, origin="lower", vmin=vmin, vmax=vmax, cmap="gray_r")
    axes[2].set_title(f"z-slice (z={slice_index}) | n={int(mz.sum())}")
    axes[2].set_xlabel("x (voxel)")
    axes[2].set_ylabel("y (voxel)")
    _scatter(axes[2], ix[mz], iy[mz], sgn[mz])

    # --- Y slice: (Z,X), overlay (x,z) = (ix,iz)
    img_y = vol[:, slice_index, :]           # (Z,X)
    axes[1].imshow(img_y, origin="lower", vmin=vmin, vmax=vmax, cmap="gray_r")
    axes[1].set_title(f"y-slice (y={slice_index}) | n={int(my.sum())}")
    axes[1].set_xlabel("x (voxel)")
    axes[1].set_ylabel("z (voxel)")
    _scatter(axes[1], ix[my], iz[my], sgn[my])

    # --- X slice: (Z,Y), overlay (y,z) = (iy,iz)
    img_x = vol[:, :, slice_index]           # (Z,Y)
    axes[0].imshow(img_x, origin="lower", vmin=vmin, vmax=vmax, cmap="gray_r")
    axes[0].set_title(f"x-slice (x={slice_index}) | n={int(mx.sum())}")
    axes[0].set_xlabel("y (voxel)")
    axes[0].set_ylabel("z (voxel)")
    _scatter(axes[0], iy[mx], iz[mx], sgn[mx])

    # Shared title + layout
    if title is None:
        title = f"Manual sign anchors over cryoET intensity | slice_index={slice_index} | thickness={slice_thickness_vox} vox"
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

    return {
        "slice_index": slice_index,
        "n_points_in_slices": {"x": int(mx.sum()), "y": int(my.sum()), "z": int(mz.sum())},
        "contrast_limits": {"vmin": float(vmin), "vmax": float(vmax)},
    }


