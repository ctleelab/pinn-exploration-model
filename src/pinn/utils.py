import numpy as np
from pinn.model import loss_data, loss_sign, loss_phys, total_loss
import jax
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt
from flax.training import checkpoints
import os


def initial_loss(state, data_edge, data_sign, data_phys):
    """
    Compute the actual initial loss values before training starts.
    Returns a structured dictionary compatible with checkpoint storage.
    """

    # Define the function using current model parameters
    params = state.params
    phi_fn = lambda x: state.apply_fn(params, x.reshape(-1, 3))

    # Compute the losses
    loss_data_val = loss_data(phi_fn, data_edge)
    loss_phys_val = loss_phys(phi_fn, data_phys)
    loss_sign_val = loss_sign(phi_fn, data_sign)
    total_loss_val = state.lambda_1 * loss_data_val + state.lambda_2 * loss_phys_val + state.lambda_3 * loss_sign_val

    # Convert to structured format (single-step array)
    loss_log = {
        "step": np.array([0]),  # Ensure consistency with later steps
        "total_loss": np.array([total_loss_val]),
        "data_loss": np.array([loss_data_val]),
        "phys_loss": np.array([loss_phys_val]),
        "sign_loss": np.array([loss_sign_val])
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
    assembled_loss = {"step": [], "total_loss": [], "data_loss": [], "phys_loss": [], "sign_loss": []}

    for step, checkpoint in checkpoint_data.items():
        if 'loss' in checkpoint:
            loss_data = checkpoint['loss']
            assembled_loss["step"].extend(loss_data["step"].tolist())
            assembled_loss["total_loss"].extend(loss_data["total_loss"].tolist())
            assembled_loss["data_loss"].extend(loss_data["data_loss"].tolist())
            assembled_loss["phys_loss"].extend(loss_data["phys_loss"].tolist())
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


def sample_surface_points(
    key,
    checkpoint,
    n_points: int,
    *,
    bounds=(-1.0, 1.0),
    oversample: int = 50,
    phi_band: float = 0.02,
    max_rounds: int = 10,
    perm=(2, 1, 0),
):
    """
    Sample points only near the surface phi = 0 by keeping candidates
    satisfying |phi| < phi_band.

    Parameters
    ----------
    key : jax.random.PRNGKey
    checkpoint : dict
        Must contain checkpoint["state"]["params"].
    n_points : int
        Number of surface-near points to return.
    bounds : tuple[float, float]
        Sampling range for each coordinate.
    oversample : int
        Number of candidate points per desired point in each round.
    phi_band : float
        Threshold for near-surface points: keep points with |phi| < phi_band.
    max_rounds : int
        Maximum number of resampling rounds if not enough points are found.
    perm : tuple[int, int, int]
        Axis permutation applied at the end.

    Returns
    -------
    dict
        {"points": pts} where pts has shape (n_points, 3)
    """

    state = checkpoint["state"]
    params = state["params"]
    model = _PINN()
    phi_fn = lambda x: model.apply(params, x).reshape(-1)

    lo, hi = bounds

    collected = []
    n_collected = 0

    for i in range(max_rounds):
        key, key_cand = jax.random.split(key)

        M = int(oversample * max(n_points - n_collected, 1))
        cand = jax.random.uniform(key_cand, (M, 3), minval=lo, maxval=hi)

        phi_cand = phi_fn(cand)  # shape (M,)
        mask = jnp.abs(phi_cand) < phi_band

        pts_keep = cand[mask]
        collected.append(pts_keep)
        n_collected += pts_keep.shape[0]

        if n_collected >= n_points:
            break

    if len(collected) == 0:
        raise ValueError("No candidate points were collected.")

    points = jnp.concatenate(collected, axis=0)

    if points.shape[0] < n_points:
        raise ValueError(
            f"Could not collect enough near-surface points. "
            f"Got {points.shape[0]} points with |phi| < {phi_band}. "
            f"Try increasing oversample, phi_band, or max_rounds."
        )

    # Randomly choose exactly n_points from the collected near-surface points
    key, key_pick = jax.random.split(key)
    idx = jax.random.choice(key_pick, points.shape[0], shape=(n_points,), replace=False)
    pts = points[idx]

    # Shuffle final points
    key, key_shuf = jax.random.split(key)
    pts = jax.random.permutation(key_shuf, pts, axis=0)

    pts = pts[:, perm]  # phi_fn uses (Z,Y,X), save as (x,y,z)

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


def strip_meta(data):
    return {
        "points": data["points"],
        "label": data["label"],
    }



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


def to_numpy(tree):
    return jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x)) if isinstance(x, (jnp.ndarray, np.ndarray)) else x,
        tree,
    )

def as_f32_scalar(x):
    # Make sure to extract as a scalar (float32) even if it's an array/DeviceArray
    return np.asarray(x, dtype=np.float32).reshape(()).item()


def pick(d, *keys, default=0.0):
    """Return first existing key in dict d."""
    for k in keys:
        if k in d:
            return d[k]
    return default

def loss_dict_to_batched(loss_list):
    """loss_list: list[dict(step=int, total_loss=..., ...)] -> dict of np arrays"""
    return {
        "step":       np.asarray([e["step"] for e in loss_list], dtype=np.int64),
        "total_loss": np.asarray([as_f32_scalar(e["total_loss"]) for e in loss_list], dtype=np.float32),
        "data_loss":  np.asarray([as_f32_scalar(e["data_loss"])  for e in loss_list], dtype=np.float32),
        "phys_loss":  np.asarray([as_f32_scalar(e["phys_loss"])  for e in loss_list], dtype=np.float32),
        "sign_loss":  np.asarray([as_f32_scalar(e["sign_loss"])  for e in loss_list], dtype=np.float32),
    }

def save_ckpt(checkpoint_dir, step, state, loss_batch, keep):
    payload = {"state": to_numpy(state), "loss": loss_batch}
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=payload,
        step=step,
        overwrite=False,
        keep=keep,
    )



from skimage.measure import marching_cubes

def sample_sign_points_marching_cubes(
    n_sample=1000,
    grid_size=96,
    iso_level=0.0,
    balance=True,              # True: try 50/50 inside/outside
    oversample_factor=6,       # how many candidate points to draw per needed point
    checkpoint=None,
    key=jax.random.PRNGKey(1234),
    perm=(0,1,2),            # save as (x,y,z) if model takes (z,y,x)
):
    """
    Build a mesh from phi via marching cubes, then label points by inside/outside of the mesh.

    Returns:
      dict with keys:
        - "points": (N,3) in saved order (after perm)
        - "label" : (N,)  +1 outside, -1 inside  (or swap if you prefer)
    """
    if checkpoint is None:
        raise ValueError("checkpoint is required")

    # --- model phi(x) ---
    state = checkpoint["state"]
    params = state["params"]
    model = _PINN()
    phi_fn = lambda x: model.apply(params, x)  # expects (N,3) in model-native coordinate order

    # --- 1) evaluate phi on grid in [-1,1]^3 ---
    lin = jnp.linspace(-1.0, 1.0, grid_size)

    # IMPORTANT:
    # Your model order is (Z,Y,X). We'll generate grid points in (Z,Y,X) order for phi_fn,
    # then later interpret marching cubes vertices accordingly.
    Z, Y, X = jnp.meshgrid(lin, lin, lin, indexing="ij")  # (z,y,x)
    grid_zyx = jnp.stack([Z.ravel(), Y.ravel(), X.ravel()], axis=-1)  # (G^3,3)

    # Evaluate phi on the grid (on GPU if available)
    phi_vals = phi_fn(grid_zyx).reshape((grid_size, grid_size, grid_size))
    phi_np = np.array(np.asarray(phi_vals), dtype=np.float32, copy=True, order="C")
    phi_np.setflags(write=True)


    # --- 2) marching cubes ---
    # marching_cubes returns vertices in index coordinates (k,j,i) ~ (z,y,x) index space
    verts_zyx_idx, faces, normals, _ = marching_cubes(
        volume=phi_np,
        level=iso_level,
        spacing=(1.0, 1.0, 1.0),
    )

    # Convert vertices from index space [0, G-1] to physical coords [-1,1]
    # idx -> coord: coord = -1 + 2 * idx/(G-1)
    verts_zyx = -1.0 + 2.0 * (verts_zyx_idx / (grid_size - 1.0))

    # Build mesh in (x,y,z) order for trimesh (it doesn't care, but we must be consistent)
    # verts_zyx = (z,y,x) -> verts_xyz
    verts_xyz = verts_zyx[:, [2, 1, 0]]

    # --- 3) point-in-mesh using trimesh ---
    try:
        import trimesh
    except ImportError as e:
        raise ImportError(
            "This function requires trimesh for robust inside/outside queries.\n"
            "Install with: pip install trimesh"
        ) from e

    mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=False)

    # If you want extra robustness (sometimes helps if the mesh is not perfectly watertight):
    # mesh = mesh.process(validate=True)
    # mesh.remove_degenerate_faces()
    # mesh.remove_duplicate_faces()

    n_inside_target = n_sample // 2 if balance else None
    n_outside_target = n_sample - (n_sample // 2) if balance else None

    pts_inside = []
    pts_outside = []

    # candidate sampling loop
    remaining_total = n_sample
    k = key

    # safety loop: stop if we can’t fill for some reason
    max_iter = 50
    for _ in range(max_iter):
        if remaining_total <= 0:
            break

        # decide how many candidates to draw this round
        n_need = remaining_total if not balance else max(
            (n_inside_target - sum(p.shape[0] for p in pts_inside)),
            (n_outside_target - sum(p.shape[0] for p in pts_outside)),
            0,
        )
        if n_need <= 0:
            break

        k, subk = jax.random.split(k)
        n_cand = int(max(oversample_factor * n_need, 1024))
        cand = jax.random.uniform(subk, (n_cand, 3), minval=-1.0, maxval=1.0)
        cand_xyz = np.asarray(cand)  # trimesh works on numpy

        inside = mesh.contains(cand_xyz)  # (n_cand,) bool
        inside_pts = cand_xyz[inside]
        outside_pts = cand_xyz[~inside]

        if balance:
            # take what we still need
            need_in = n_inside_target - sum(p.shape[0] for p in pts_inside)
            need_out = n_outside_target - sum(p.shape[0] for p in pts_outside)

            if need_in > 0 and inside_pts.shape[0] > 0:
                take = inside_pts[:need_in]
                pts_inside.append(take)

            if need_out > 0 and outside_pts.shape[0] > 0:
                take = outside_pts[:need_out]
                pts_outside.append(take)

            remaining_total = (
                (n_inside_target - sum(p.shape[0] for p in pts_inside)) +
                (n_outside_target - sum(p.shape[0] for p in pts_outside))
            )
        else:
            # just collect outside+inside indiscriminately until n_sample
            # (still labels correctly)
            take_in = inside_pts
            take_out = outside_pts
            pts_inside.append(take_in)
            pts_outside.append(take_out)
            remaining_total = n_sample - (sum(p.shape[0] for p in pts_inside) + sum(p.shape[0] for p in pts_outside))

    if balance:
        inside_all = np.concatenate(pts_inside, axis=0) if pts_inside else np.zeros((0, 3), dtype=np.float32)
        outside_all = np.concatenate(pts_outside, axis=0) if pts_outside else np.zeros((0, 3), dtype=np.float32)

        if inside_all.shape[0] < n_inside_target or outside_all.shape[0] < n_outside_target:
            print(f"[Warning] Collected inside={inside_all.shape[0]}/{n_inside_target}, outside={outside_all.shape[0]}/{n_outside_target}")

        inside_all = inside_all[:n_inside_target]
        outside_all = outside_all[:n_outside_target]

        pts_xyz = np.concatenate([inside_all, outside_all], axis=0)
        # Convention: inside = -1, outside = +1
        sign = np.concatenate([
            -np.ones((inside_all.shape[0],), dtype=np.float32),
            +np.ones((outside_all.shape[0],), dtype=np.float32),
        ], axis=0)
    else:
        inside_all = np.concatenate(pts_inside, axis=0) if pts_inside else np.zeros((0, 3), dtype=np.float32)
        outside_all = np.concatenate(pts_outside, axis=0) if pts_outside else np.zeros((0, 3), dtype=np.float32)

        pts_xyz = np.concatenate([inside_all, outside_all], axis=0)[:n_sample]
        sign = np.concatenate([
            -np.ones((inside_all.shape[0],), dtype=np.float32),
            +np.ones((outside_all.shape[0],), dtype=np.float32),
        ], axis=0)[:n_sample]

    # shuffle to avoid ordered blocks
    rng = np.random.default_rng(int(jax.random.randint(k, (), 0, 2**31 - 1)))
    idx = np.arange(pts_xyz.shape[0])
    rng.shuffle(idx)
    pts_xyz = pts_xyz[idx]
    sign = sign[idx]

    # Convert to jnp and apply your "save order" permutation.
    # Currently pts_xyz is (x,y,z). You want to save in (x,y,z) style,
    # but your old code did pts = pts[:, perm] where perm=(2,1,0).
    # Keep the same behavior here for compatibility.
    pts = jnp.asarray(pts_xyz)
    pts = pts[:, perm]

    return {
        "points": pts,
        "label": jnp.asarray(sign),
    }




LOSS_KEYS = ("step", "total_loss", "data_loss", "phys_loss", "sign_loss")

def _as_1d_np(x):
    x = np.asarray(x)
    return x.reshape(-1)

def load_loss_from_checkpoint(ckpt_path):
    """Load ONLY 'loss' from one checkpoint path."""
    # restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target={"loss": None})
    restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None)
    loss = restored["loss"]
    # normalize to 1D numpy arrays
    loss_np = {k: _as_1d_np(loss[k]) for k in LOSS_KEYS}
    return loss_np

def load_loss_history_dir(ckpt_dir, steps_actual, step_offset=0):
    """
    Load loss histories from a directory.

    Parameters
    ----------
    steps_actual : list[int]
        Which checkpoint_<step_actual> to read from disk.
    step_offset : int
        Add this to the 'step' array inside each checkpoint so it becomes global.

    Returns
    -------
    hist : dict[str, np.ndarray]
        Concatenated history arrays.
    """
    out = {k: [] for k in LOSS_KEYS}

    for step_actual in steps_actual:
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{step_actual}")
        loss = load_loss_from_checkpoint(ckpt_path)

        # offset local steps -> global steps
        steps = loss["step"] + step_offset

        out["step"].append(steps)
        for k in ("total_loss", "data_loss", "phys_loss", "sign_loss"):
            out[k].append(loss[k])

    # concatenate lists of arrays
    hist = {k: np.concatenate(out[k]) if len(out[k]) > 0 else np.array([]) for k in LOSS_KEYS}

    # sort by step
    order = np.argsort(hist["step"])
    hist = {k: hist[k][order] for k in LOSS_KEYS}

    # drop duplicate steps (keep last) just in case base+cont overlaps at 10000
    if hist["step"].size > 0:
        steps = hist["step"]
        rev = steps[::-1]
        _, idx_rev = np.unique(rev, return_index=True)   # unique on reversed keeps last
        keep = (steps.size - 1 - idx_rev)
        keep.sort()
        hist = {k: hist[k][keep] for k in LOSS_KEYS}

    return hist

def concat_histories(h1, h2):
    """Concatenate two history dicts and keep last on duplicate steps."""
    if h1["step"].size == 0:
        return h2
    if h2["step"].size == 0:
        return h1

    hist = {k: np.concatenate([h1[k], h2[k]]) for k in LOSS_KEYS}
    order = np.argsort(hist["step"])
    hist = {k: hist[k][order] for k in LOSS_KEYS}

    # drop duplicates, keep last
    steps = hist["step"]
    rev = steps[::-1]
    _, idx_rev = np.unique(rev, return_index=True)
    keep = (steps.size - 1 - idx_rev)
    keep.sort()
    hist = {k: hist[k][keep] for k in LOSS_KEYS}
    return hist



