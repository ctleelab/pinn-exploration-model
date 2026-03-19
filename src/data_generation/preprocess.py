import skimage.measure
from skimage.transform import resize
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import cv2

import matplotlib.pyplot as plt
plt.style.core.USER_LIBRARY_PATHS.append('/gscratch/matsulab/utils/ctleelab-mpl-utilities/ctleelab_plothelper')
plt.style.reload_library()

def clahe_volume_slicewise_fast(volume, axis=0, clipLimit=2.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    vol = np.asarray(volume)
    volA = np.moveaxis(vol, axis, 0)  # shape (N, H, W)

    outA = np.empty(volA.shape, dtype=np.uint8)
    for i in range(volA.shape[0]):
        sl_u8 = cv2.normalize(volA[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        outA[i] = clahe.apply(sl_u8)

    return np.moveaxis(outA, 0, axis)

def downsample_ori(
    cryoet_data, 
    x_start, x_end, 
    y_start, y_end, 
    z_start, z_end, 
    k=1, 
    GRID_SIZE = 64, 
    intensity_max = 1.0, 
    k_xy=None,
):
    
    # --- Crop ---
    cropped = cryoet_data[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # --- Downsample ---
    factor = 1
    downsampled = skimage.measure.block_reduce(
        cropped, block_size=(factor, factor, factor), func=np.mean
    ).astype(np.float32)
    
    # downsampled = -downsampled
    
    # --- Pad Z to match max(Y, X) ---
    Z, Y, X = downsampled.shape
    target_size = max(Y, X)
    
    pad_total = target_size - Z
    pad_before = pad_total // 2
    pad_after  = pad_total - pad_before
    
    padded = np.pad(
        downsampled,
        pad_width=((pad_before, pad_after), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    
    # Mask: 1 where original data exists, 0 where padding is
    mask = np.ones_like(downsampled, dtype=np.uint8)
    mask_padded = np.pad(
        mask,
        pad_width=((pad_before, pad_after), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    print("Downsampled shape:", downsampled.shape)
    print("Padded shape:  ", padded.shape)
    
    # IMPORTANT: use constant mode + cval=0 so resize doesn't reflect data into padding
    resized = resize(
        padded, (GRID_SIZE, GRID_SIZE, GRID_SIZE),
        mode="constant", cval=0,
        anti_aliasing=True,
        preserve_range=True,
    )
    
    # Resize mask with nearest-neighbor (order=0) so it stays binary-ish
    mask_resized = resize(
        mask_padded.astype(np.float32), (GRID_SIZE, GRID_SIZE, GRID_SIZE),
        mode="constant", cval=0,
        anti_aliasing=False,
        order=0,
        preserve_range=True,
    ) > 0.5

    # --- Enforce zero padding on x/y boundaries ---
    if k_xy is not None and k_xy > 0:
        mask_resized[:, :, :k_xy] = False        # x-min
        mask_resized[:, :, -k_xy:] = False       # x-max
        mask_resized[:, :k_xy, :] = False        # y-min
        mask_resized[:, -k_xy:, :] = False       # y-max


    print("Resized shape:  ", resized.shape)
    
    # Hard-zero padded region after resize
    resized[~mask_resized] = 0.0
    
    # --- Normalize (option: normalize using only real voxels) ---
    intensity_max = intensity_max
    real_vals = resized[mask_resized]
    global_min = real_vals.min()
    global_max = real_vals.max()
    
    normalized = (resized - global_min) / (global_max - global_min + 1e-12)
    normalized *= intensity_max
    normalized = np.clip(normalized, 0.0, 1.0)
    
    # --- Shrink valid region by 1 z-layer on both ends ---
    valid_z = np.where(mask_resized.any(axis=(1, 2)))[0]
    if valid_z.size > 0:
        zmin, zmax = valid_z[0], valid_z[-1]
        mask_resized[zmin:zmin+k, :, :] = False
        mask_resized[zmax-k+1:zmax+1, :, :] = False
    
    # Enforce zero using the updated mask (do this at the very end)
    normalized[~mask_resized] = 0.0

    print("Normalized shape:", normalized.shape)
    return normalized


def downsample(
    cryoet_data, 
    x_start, x_end, 
    y_start, y_end, 
    z_start, z_end, 
    k=1, 
    GRID_SIZE=64, 
    intensity_max=1.0, 
    k_xy=None,
):
    # --- Crop ---
    cropped = cryoet_data[z_start:z_end, y_start:y_end, x_start:x_end]

    # --- Downsample ---
    factor = 1
    downsampled = skimage.measure.block_reduce(
        cropped, block_size=(factor, factor, factor), func=np.mean
    ).astype(np.float32)

    # --- Pad to cube (pad whichever axes are smaller than the largest axis) ---
    Z, Y, X = downsampled.shape
    target_size = max(Z, Y, X)

    def _pad_to(n, target):
        pad_total = max(0, target - n)
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return pad_before, pad_after

    z0, z1 = _pad_to(Z, target_size)
    y0, y1 = _pad_to(Y, target_size)
    x0, x1 = _pad_to(X, target_size)

    padded = np.pad(
        downsampled,
        pad_width=((z0, z1), (y0, y1), (x0, x1)),
        mode="constant",
        constant_values=0,
    )

    # Mask: 1 where original data exists, 0 where padding is
    mask = np.ones_like(downsampled, dtype=np.uint8)
    mask_padded = np.pad(
        mask,
        pad_width=((z0, z1), (y0, y1), (x0, x1)),
        mode="constant",
        constant_values=0,
    )

    print("Downsampled shape:", downsampled.shape)
    print("Padded shape:    ", padded.shape)

    # IMPORTANT: use constant mode + cval=0 so resize doesn't reflect data into padding
    resized = resize(
        padded, (GRID_SIZE, GRID_SIZE, GRID_SIZE),
        mode="constant", cval=0,
        anti_aliasing=True,
        preserve_range=True,
    )

    # Resize mask with nearest-neighbor (order=0) so it stays binary-ish
    mask_resized = resize(
        mask_padded.astype(np.float32), (GRID_SIZE, GRID_SIZE, GRID_SIZE),
        mode="constant", cval=0,
        anti_aliasing=False,
        order=0,
        preserve_range=True,
    ) > 0.5

    # --- Enforce zero padding on x/y boundaries ---
    if k_xy is not None and k_xy > 0:
        mask_resized[:, :, :k_xy] = False        # x-min
        mask_resized[:, :, -k_xy:] = False       # x-max
        mask_resized[:, :k_xy, :] = False        # y-min
        mask_resized[:, -k_xy:, :] = False       # y-max

    print("Resized shape:   ", resized.shape)

    # Hard-zero padded region after resize
    resized[~mask_resized] = 0.0

    # --- Normalize (option: normalize using only real voxels) ---
    real_vals = resized[mask_resized]
    global_min = real_vals.min()
    global_max = real_vals.max()

    normalized = (resized - global_min) / (global_max - global_min + 1e-12)
    normalized *= intensity_max
    normalized = np.clip(normalized, 0.0, 1.0)

    # --- Shrink valid region by 1 z-layer on both ends ---
    valid_z = np.where(mask_resized.any(axis=(1, 2)))[0]
    if valid_z.size > 0:
        zmin, zmax = valid_z[0], valid_z[-1]
        mask_resized[zmin:zmin+k, :, :] = False
        mask_resized[zmax-k+1:zmax+1, :, :] = False

    # Enforce zero using the updated mask (do this at the very end)
    normalized[~mask_resized] = 0.0

    print("Normalized shape:", normalized.shape)
    return normalized



def plot_edge_and_region(
    cryoet,
    edge,
    outside,
    inside,
    axis="z",
    slice_index=None,
    show_title=True,
):
    axis_to_dim = {"x": 0, "y": 1, "z": 2}
    dim = axis_to_dim[axis]
    dim_size = cryoet.shape[dim]

    if slice_index is None:
        slice_index = cryoet.shape[dim] // 2

    def get_slice(arr):
        return np.take(arr, slice_index, axis=dim)

    cryo_slice = get_slice(cryoet)
    edge_slice = get_slice(edge)
    outside_slice = get_slice(outside)
    inside_slice = get_slice(inside)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3), squeeze=False)
    axes = axes[0]

    custom_gray = LinearSegmentedColormap.from_list(
        "custom_gray", ["#f0f0f0", "#111111"]
    )

    # 1. Original cryo-ET slice
    axes[0].imshow(cryo_slice, cmap=custom_gray)
    if show_title:
        axes[0].set_title(f"Original ({axis}: {slice_index}/{dim_size})")

    # 2–4. Masks
    masks = [edge_slice, outside_slice, inside_slice]
    titles = [
        f"Edge ({axis}: {slice_index}/{dim_size})", 
        f"Outside ({axis}: {slice_index}/{dim_size})", 
        f"Inside ({axis}: {slice_index}/{dim_size})"
    ]

    for ax, mask, title in zip(axes[1:], masks, titles):
        ax.imshow(mask.astype(float), cmap="gray_r", vmin=0, vmax=1)
        if show_title:
            ax.set_title(title)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_edge(
    cryoet,
    edge,
    bulk,
    axis="z",
    slice_index=None,
    show_title=True,
):
    axis_to_dim = {"x": 0, "y": 1, "z": 2}
    dim = axis_to_dim[axis]
    dim_size = cryoet.shape[dim]

    if slice_index is None:
        slice_index = cryoet.shape[dim] // 2

    def get_slice(arr):
        arr = np.asarray(arr)
        return np.take(arr, slice_index, axis=dim)

    cryo_slice = get_slice(cryoet)
    edge_slice = get_slice(edge)
    bulk_slice = get_slice(bulk)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3), squeeze=False)
    axes = axes[0]

    custom_gray = LinearSegmentedColormap.from_list(
        "custom_gray", ["#f0f0f0", "#111111"]
    )

    # 1. Original cryo-ET slice
    axes[0].imshow(cryo_slice, cmap=custom_gray)
    if show_title:
        axes[0].set_title(f"Original ({axis}: {slice_index}/{dim_size})")

    # 2–4. Masks
    masks = [edge_slice, bulk_slice]
    titles = [
        f"Edge ({axis}: {slice_index}/{dim_size})", 
        f"Bulk ({axis}: {slice_index}/{dim_size})"
    ]

    for ax, mask, title in zip(axes[1:], masks, titles):
        ax.imshow(mask.astype(float), cmap="gray_r", vmin=0, vmax=1)
        if show_title:
            ax.set_title(title)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()



def overlay_masks_on_cryoet(
    cryoet,
    edge,
    bulk,
    outside,
    inside,
    axis="z",
    slice_index=None,
    alpha=0.35,
    show_title=True,
):
    axis_to_dim = {"x": 0, "y": 1, "z": 2}
    dim = axis_to_dim[axis]
    dim_size = cryoet.shape[dim]

    if slice_index is None:
        slice_index = cryoet.shape[dim] // 2

    def get_slice(arr):
        return np.take(arr, slice_index, axis=dim)

    cryo = get_slice(cryoet)
    e = get_slice(edge).astype(np.float32)
    b = get_slice(bulk).astype(np.float32)
    o = get_slice(outside).astype(np.float32)
    i = get_slice(inside).astype(np.float32)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    custom_gray = LinearSegmentedColormap.from_list(
        "custom_gray", ["#f0f0f0", "#111111"]
    )
    
    # Background: grayscale cryoET
    ax.imshow(cryo, cmap=custom_gray)

    # Overlays: distinct colormaps, masked by zeros
    ax.imshow(np.ma.masked_where(e == 0, e), cmap="Greens",   vmin=0, vmax=1, alpha=alpha)
    ax.imshow(np.ma.masked_where(b == 0, b), cmap="Oranges",  vmin=0, vmax=1, alpha=alpha)
    ax.imshow(np.ma.masked_where(o == 0, o), cmap="Reds",  vmin=0, vmax=1, alpha=alpha)
    ax.imshow(np.ma.masked_where(i == 0, i), cmap="Blues", vmin=0, vmax=1, alpha=alpha)

    if show_title:
        ax.set_title(f"Overlay ({axis}: {slice_index}/{dim_size})")

    ax.axis("off")

    # -------- Legend --------
    legend_elements = [
        Patch(facecolor=plt.cm.Greens(0.8),   edgecolor="none", label="Edge"),
        Patch(facecolor=plt.cm.Oranges(0.8),   edgecolor="none", label="Bulk"),
        Patch(facecolor=plt.cm.Reds(0.8),  edgecolor="none", label="Outside"),
        Patch(facecolor=plt.cm.Blues(0.8), edgecolor="none", label="Inside"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
        fontsize=9,
    )

    plt.tight_layout()
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

def plot_edge_region_with_bulk_2rows(
    cryoet,
    edge,
    bulk,
    outside,
    inside,
    axis="z",
    slice_index=None,
    alpha=0.35,
    show_title=True,
):
    """
    2x3 plot.
      Row 1: Original | Edge | Bulk
      Row 2: Overlay  | Outside | Inside
    """
    axis_to_dim = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_to_dim:
        raise ValueError("axis must be 'x', 'y', or 'z'.")
    dim = axis_to_dim[axis]
    dim_size = cryoet.shape[dim]

    if slice_index is None:
        slice_index = dim_size // 2

    def get_slice(arr):
        return np.take(arr, slice_index, axis=dim)

    cryo_slice = get_slice(cryoet)
    edge_slice = get_slice(edge).astype(np.float32)
    bulk_slice = get_slice(bulk).astype(np.float32)
    outside_slice = get_slice(outside).astype(np.float32)
    inside_slice = get_slice(inside).astype(np.float32)

    custom_gray = LinearSegmentedColormap.from_list("custom_gray", ["#f0f0f0", "#111111"])

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7), squeeze=False)

    # ---------------- Row 1 ----------------
    # (1,1) Original
    axes[0, 0].imshow(cryo_slice, cmap=custom_gray)
    if show_title:
        axes[0, 0].set_title(f"Original ({axis}: {slice_index}/{dim_size})")

    # (1,2) Edge
    axes[0, 1].imshow(edge_slice, cmap="gray_r", vmin=0, vmax=1)
    if show_title:
        axes[0, 1].set_title(f"Edge ({axis}: {slice_index}/{dim_size})")

    # (1,3) Bulk
    axes[0, 2].imshow(bulk_slice, cmap="gray_r", vmin=0, vmax=1)
    if show_title:
        axes[0, 2].set_title(f"Bulk ({axis}: {slice_index}/{dim_size})")

    # ---------------- Row 2 ----------------
    # (2,1) Overlay
    ax = axes[1, 0]
    ax.imshow(cryo_slice, cmap=custom_gray)

    e = edge_slice
    b = bulk_slice
    o = outside_slice
    i = inside_slice

    ax.imshow(np.ma.masked_where(e == 0, e), cmap="Greens",  vmin=0, vmax=1, alpha=alpha)
    ax.imshow(np.ma.masked_where(b == 0, b), cmap="Purples", vmin=0, vmax=1, alpha=alpha)
    ax.imshow(np.ma.masked_where(o == 0, o), cmap="Reds",    vmin=0, vmax=1, alpha=alpha)
    ax.imshow(np.ma.masked_where(i == 0, i), cmap="Blues",   vmin=0, vmax=1, alpha=alpha)

    if show_title:
        ax.set_title(f"Overlay ({axis}: {slice_index}/{dim_size})")

    legend_elements = [
        Patch(facecolor=plt.cm.Greens(0.8),  edgecolor="none", label="Edge"),
        Patch(facecolor=plt.cm.Purples(0.8), edgecolor="none", label="Bulk"),
        Patch(facecolor=plt.cm.Reds(0.8),    edgecolor="none", label="Outside"),
        Patch(facecolor=plt.cm.Blues(0.8),   edgecolor="none", label="Inside"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=9)

    # (2,2) Outside
    axes[1, 1].imshow(outside_slice, cmap="gray_r", vmin=0, vmax=1)
    if show_title:
        axes[1, 1].set_title(f"Outside ({axis}: {slice_index}/{dim_size})")

    # (2,3) Inside
    axes[1, 2].imshow(inside_slice, cmap="gray_r", vmin=0, vmax=1)
    if show_title:
        axes[1, 2].set_title(f"Inside ({axis}: {slice_index}/{dim_size})")

    # Styling
    for r in range(2):
        for c in range(3):
            axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()



def downsample_binary_mask_to_fraction(mask01, target_frac, seed=None):
    """
    Randomly flips 1->0 in a binary mask until the overall fraction of 1s
    (among ALL voxels) reaches target_frac.

    Parameters
    ----------
    mask01 : np.ndarray
        Binary mask (float32 0/1 or bool).
    target_frac : float
        Desired fraction of 1s among all voxels. Must be in [0, 1].
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Float32 mask with reduced number of 1s.
    """
    if not (0.0 <= target_frac <= 1.0):
        raise ValueError("target_frac must be between 0 and 1.")

    rng = np.random.default_rng(seed)

    mask_bool = mask01.astype(bool, copy=False)
    total = mask_bool.size
    current_ones = int(mask_bool.sum())
    desired_ones = int(round(target_frac * total))

    # If already at or below target, return as-is (float32)
    if current_ones <= desired_ones:
        return mask_bool.astype(np.float32)

    # Choose which 1s to flip to 0
    n_remove = current_ones - desired_ones
    one_idx = np.flatnonzero(mask_bool)              # 1D indices where mask==1
    remove_idx = rng.choice(one_idx, size=n_remove, replace=False)

    out = mask_bool.copy().ravel()
    out[remove_idx] = False
    return out.reshape(mask_bool.shape).astype(np.float32)


import numpy as np
import jax.numpy as jnp

def build_data_sign_from_masks(
    outside,
    inside,
    volume_shape,
    axis_perm=(2, 1, 0),  # (z,y,x) → (x,y,z)
    subsample=None,
    seed=0,
):
    rng = np.random.default_rng(seed)

    outside = outside.astype(bool)
    inside = inside.astype(bool)

    # --- voxel indices in array order (e.g. z,y,x) ---
    idx_out = np.argwhere(outside)
    idx_in  = np.argwhere(inside)

    # --- optional subsampling ---
    if subsample is not None:
        if len(idx_out) > subsample:
            idx_out = idx_out[rng.choice(len(idx_out), subsample, replace=False)]
        if len(idx_in) > subsample:
            idx_in = idx_in[rng.choice(len(idx_in), subsample, replace=False)]

    # --- permute axes to PINN order (x,y,z) ---
    idx_out = idx_out[:, axis_perm]
    idx_in  = idx_in[:, axis_perm]

    # --- corresponding shape in PINN order ---
    shape_xyz = np.array(volume_shape)[list(axis_perm)]

    # --- normalize to [-1, 1]^3 ---
    def normalize(idx, shape):
        coords = np.empty_like(idx, dtype=np.float32)
        for d in range(3):
            coords[:, d] = 2.0 * idx[:, d] / (shape[d] - 1) - 1.0
        return coords

    x_out = normalize(idx_out, shape_xyz)
    x_in  = normalize(idx_in,  shape_xyz)

    points = np.concatenate([x_out, x_in], axis=0)
    signs = np.concatenate([
        np.ones(len(x_out), dtype=np.float32),
        -np.ones(len(x_in), dtype=np.float32),
    ])

    return {
        "points": jnp.asarray(points),
        "label":  jnp.asarray(signs),
    }


def build_data_edge_from_masks(
    edge,
    bulk,
    volume_shape,
    axis_perm=(2, 1, 0),  # (z,y,x) → (x,y,z)
):
    edge = edge.astype(bool)
    bulk = bulk.astype(bool)

    # --- voxel indices in array order (e.g. z,y,x) ---
    idx_edge = np.argwhere(edge)
    idx_bulk = np.argwhere(bulk)

    # --- permute axes to PINN order (x,y,z) ---
    idx_edge = idx_edge[:, axis_perm]
    idx_bulk = idx_bulk[:, axis_perm]

    # --- corresponding shape in PINN order ---
    shape_xyz = np.array(volume_shape)[list(axis_perm)]

    # --- normalize to [-1, 1]^3 ---
    def normalize(idx, shape):
        coords = np.empty_like(idx, dtype=np.float32)
        for d in range(3):
            coords[:, d] = 2.0 * idx[:, d] / (shape[d] - 1) - 1.0
        return coords

    x_edge = normalize(idx_edge, shape_xyz)
    x_bulk = normalize(idx_bulk, shape_xyz)

    points = np.concatenate([x_edge, x_bulk], axis=0)
    labels = np.concatenate([
        np.ones(len(x_edge), dtype=np.float32),
        np.zeros(len(x_bulk), dtype=np.float32),
    ], axis=0)    

    return {
        "points": jnp.asarray(points),
        "label":  jnp.asarray(labels),
    }


def overlay_masks_on_cryoet_slices(
    cryoet,
    edge,
    bulk,
    outside,
    inside,
    axis="z",
    slice_indices=(0,),
    alpha=0.35,
    show_title=True,
    figsize_per_panel=3.0,
):
    axis_to_dim = {"x": 0, "y": 1, "z": 2}
    dim = axis_to_dim[axis]

    # ensure list-like
    slice_indices = list(slice_indices)
    dim_size = cryoet.shape[dim]

    # basic bounds check
    for s in slice_indices:
        if not (0 <= s < dim_size):
            raise ValueError(f"slice index {s} out of range for axis '{axis}' (0..{dim_size-1})")

    def get_slice(arr, s):
        return np.take(arr, s, axis=dim)

    n = len(slice_indices)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel * n, figsize_per_panel), squeeze=False)
    axes = axes[0]

    custom_gray = LinearSegmentedColormap.from_list(
        "custom_gray", ["#f0f0f0", "#111111"]
    )

    for ax, s in zip(axes, slice_indices):
        cryo = get_slice(cryoet, s)
        e = get_slice(edge, s).astype(np.float32)
        b = get_slice(bulk, s).astype(np.float32)
        o = get_slice(outside, s).astype(np.float32)
        i = get_slice(inside, s).astype(np.float32)

        # background
        ax.imshow(cryo, cmap=custom_gray)

        # overlays (zeros transparent)
        ax.imshow(np.ma.masked_where(e == 0, e), cmap="Greens",   vmin=0, vmax=1, alpha=alpha)
        ax.imshow(np.ma.masked_where(b == 0, b), cmap="Oranges",   vmin=0, vmax=1, alpha=alpha)
        ax.imshow(np.ma.masked_where(o == 0, o), cmap="Reds",  vmin=0, vmax=1, alpha=alpha)
        ax.imshow(np.ma.masked_where(i == 0, i), cmap="Blues", vmin=0, vmax=1, alpha=alpha)

        if show_title:
            ax.set_title(f"{axis}={s}/{dim_size-1}", fontsize=10)

        ax.axis("off")

    # one shared legend (right side)
    legend_elements = [
        Patch(facecolor=plt.cm.Greens(0.8),   edgecolor="none", label="Edge"),
        Patch(facecolor=plt.cm.Oranges(0.8),   edgecolor="none", label="Bulk"),
        Patch(facecolor=plt.cm.Reds(0.8),  edgecolor="none", label="Outside"),
        Patch(facecolor=plt.cm.Blues(0.8), edgecolor="none", label="Inside"),
    ]
    fig.legend(handles=legend_elements, loc="center right", frameon=True, fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.92, 1])  # leave room for legend

    plt.show()




def plot_sampling_points(
    cryoet,
    edges,   # {"points": jnp/np (N,3), "label": jnp/np (N,)}
    signs,   # {"points": jnp/np (M,3), "label": jnp/np (M,)}
    phys,    # {"points": jnp/np (K,3)}
    axis="z",
    slice_index=None,
    s=8,                 # point size
    alpha_pts=1.0,       # point alpha
    show_titles=True,
    show_colorbars=False,
    alpha_cryo_overlay = 0.8,
):
    """
    Produces 4 side-by-side panels:
      (1) cryo slice
      (2) edge points on that slice
      (3) sign points on that slice
      (4) physics (collocation) points on that slice

    Legend requirements:
      - Panel 2: "1: edge" and "0: bulk"
      - Panel 3: "+1" and "-1"
      - Panel 4: "collocation point"
    """

    COL_EDGE = "tab:olive"
    COL_BULK = "tab:purple"
    COL_POS  = "tab:red"
    COL_NEG  = "tab:blue"
    COL_PHYS = "tab:green"


    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    vol = np.asarray(cryoet)
    assert vol.ndim == 3, f"cryoet must be 3D, got {vol.ndim}D"
    Z, Y, X = vol.shape

    axis_to_dim = {"z": 0, "y": 1, "x": 2}
    if axis not in axis_to_dim:
        raise ValueError(f"axis must be one of {list(axis_to_dim.keys())}, got {axis}")
    dim = axis_to_dim[axis]

    if slice_index is None:
        slice_index = vol.shape[dim] // 2

    # ----- helpers -----
    def _to_numpy(a):
        try:
            return np.asarray(a)
        except Exception:
            return np.array(a)

    def points_to_voxel_indices(pts_xyz):
        """
        Map points to integer voxel indices (z,y,x).
        Supports either normalized [-1,1] or already-in-voxel coordinates.
        Assumes points are provided as (x, y, z).
        """
        pts = _to_numpy(pts_xyz).astype(np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must have shape (N,3), got {pts.shape}")

        in_norm = np.mean((pts >= -1.05) & (pts <= 1.05))
        if in_norm > 0.95:
            x_n, y_n, z_n = pts[:, 0], pts[:, 1], pts[:, 2]
            x_i = np.rint((x_n + 1) * 0.5 * (X - 1)).astype(int)
            y_i = np.rint((y_n + 1) * 0.5 * (Y - 1)).astype(int)
            z_i = np.rint((z_n + 1) * 0.5 * (Z - 1)).astype(int)
        else:
            x_i = np.rint(pts[:, 0]).astype(int)
            y_i = np.rint(pts[:, 1]).astype(int)
            z_i = np.rint(pts[:, 2]).astype(int)

        x_i = np.clip(x_i, 0, X - 1)
        y_i = np.clip(y_i, 0, Y - 1)
        z_i = np.clip(z_i, 0, Z - 1)
        return np.stack([z_i, y_i, x_i], axis=1)

    def get_slice(arr3d):
        return np.take(arr3d, slice_index, axis=dim)

    def scatter_coords_for_plane(zyx_idx):
        z_i, y_i, x_i = zyx_idx[:, 0], zyx_idx[:, 1], zyx_idx[:, 2]
        if dim == 0:      # slicing z -> image is (Y, X)
            row, col = y_i, x_i
        elif dim == 1:    # slicing y -> image is (Z, X)
            row, col = z_i, x_i
        else:             # slicing x -> image is (Z, Y)
            row, col = z_i, y_i
        return row, col

    def filter_points_on_slice(zyx_idx):
        return zyx_idx[zyx_idx[:, dim] == slice_index]

    def _legend_handle_for_scatter(sc, label, marker="x"):
        # For "x" markers, use line color + markeredgewidth (facecolor is ignored)
        col = sc.get_edgecolors()
        if col is None or len(col) == 0:
            col = sc.get_facecolors()
        color = "k" if col is None or len(col) == 0 else col[0]

        lw = sc.get_linewidths()
        mew = 1.5 if lw is None or len(lw) == 0 else float(lw[0])

        return Line2D(
            [0], [0],
            marker=marker,
            linestyle="None",
            color=color,              # THIS is what colors an "x"
            markeredgewidth=mew,      # thickness of the "x"
            markersize=max(4, np.sqrt(s)),  # roughly match scatter size
            label=label,
        )

    # ----- slice -----
    cryo_slice = get_slice(vol)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    # fig, axes = plt.subplots(4, 1, figsize=(4, 16), constrained_layout=True)


    # Panel 1: cryo
    axes[0].imshow(cryo_slice, cmap="gray_r")
    axes[0].set_axis_off()
    if show_titles:
        axes[0].set_title("Original Image")

    # Panel 2: edges (legend "1: edge" / "0: bulk")
    axes[1].imshow(cryo_slice, cmap="gray_r", alpha=alpha_cryo_overlay)
    if edges is not None:
        e_pts = points_to_voxel_indices(edges["points"])
        e_val = _to_numpy(edges["label"]).reshape(-1)
        if e_val.shape[0] != e_pts.shape[0]:
            raise ValueError(f"edges['label'] length {e_val.shape[0]} != points {e_pts.shape[0]}")
        mask = (e_pts[:, dim] == slice_index)
        e_pts_s = e_pts[mask]
        e_val_s = e_val[mask]

        if e_pts_s.shape[0] > 0:
            r, c = scatter_coords_for_plane(e_pts_s)

            # Split into edge (==1) and bulk (==0) so the legend can show desired labels
            m1 = (e_val_s == 1) | (e_val_s == 1.0)
            m0 = (e_val_s == 0) | (e_val_s == 0.0)

            handles = []
            if np.any(m1):
                sc1 = axes[1].scatter(c[m1], r[m1], s=s, alpha=alpha_pts, color=COL_EDGE, marker="x")
                handles.append(_legend_handle_for_scatter(sc1, "edge"))
            if np.any(m0):
                sc0 = axes[1].scatter(c[m0], r[m0], s=s, alpha=alpha_pts, color=COL_BULK, marker="x")
                handles.append(_legend_handle_for_scatter(sc0, "bulk"))

            if show_colorbars:
                # If you really want a colorbar, use the original scalar scatter:
                sc = axes[1].scatter(c, r, s=s, alpha=alpha_pts, c=e_val_s)
                plt.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.02)

            if handles:
                axes[1].legend(handles=handles, loc="lower right", frameon=True)

    axes[1].set_axis_off()
    if show_titles:
        axes[1].set_title("Edge Signals")

    # Panel 3: signs (legend "+1" / "-1")
    axes[2].imshow(cryo_slice, cmap="gray_r", alpha=alpha_cryo_overlay)
    if signs is not None:
        s_pts = points_to_voxel_indices(signs["points"])
        s_val = _to_numpy(signs["label"]).reshape(-1)
        if s_val.shape[0] != s_pts.shape[0]:
            raise ValueError(f"signs['label'] length {s_val.shape[0]} != points {s_pts.shape[0]}")
        mask = (s_pts[:, dim] == slice_index)
        s_pts_s = s_pts[mask]
        s_val_s = s_val[mask]

        if s_pts_s.shape[0] > 0:
            r, c = scatter_coords_for_plane(s_pts_s)

            mp = (s_val_s > 0)  # treat >0 as +1
            mn = (s_val_s < 0)  # treat <0 as -1

            handles = []
            if np.any(mp):
                scp = axes[2].scatter(c[mp], r[mp], s=s, alpha=alpha_pts, color=COL_POS, marker="x")
                handles.append(_legend_handle_for_scatter(scp, "+1"))
            if np.any(mn):
                scn = axes[2].scatter(c[mn], r[mn], s=s, alpha=alpha_pts, color=COL_NEG, marker="x")
                handles.append(_legend_handle_for_scatter(scn, "-1"))

            if show_colorbars:
                sc = axes[2].scatter(c, r, s=s, alpha=alpha_pts, c=s_val_s)
                plt.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.02)

            if handles:
                axes[2].legend(handles=handles, loc="lower right", frameon=True)

    axes[2].set_axis_off()
    if show_titles:
        axes[2].set_title("Sign Signals")

    # Panel 4: physics points (legend "collocation point")
    axes[3].imshow(cryo_slice, cmap="gray_r", alpha=alpha_cryo_overlay)
    if phys is not None:
        p_pts = points_to_voxel_indices(phys["points"])
        p_pts_s = filter_points_on_slice(p_pts)
        if p_pts_s.shape[0] > 0:
            r, c = scatter_coords_for_plane(p_pts_s)
            scp = axes[3].scatter(c, r, s=s, alpha=alpha_pts, color=COL_PHYS, marker="x")
            h = _legend_handle_for_scatter(scp, "collocation point")
            axes[3].legend(handles=[h], loc="lower right", frameon=True)

    axes[3].set_axis_off()
    if show_titles:
        axes[3].set_title("Collocation Points")

    return fig, axes




def plot_sampling_points_vertical(
    cryoet,
    edges,   # {"points": jnp/np (N,3), "label": jnp/np (N,)}
    signs,   # {"points": jnp/np (M,3), "label": jnp/np (M,)}
    phys,    # {"points": jnp/np (K,3)}
    axis="z",
    slice_index=None,
    s=8,                 # point size
    alpha_pts=1.0,       # point alpha
    show_titles=False,
    show_colorbars=False,
    alpha_cryo_overlay=0.8,
):
    """
    Produces 4 side-by-side panels:
      (1) cryo slice
      (2) edge points on that slice
      (3) sign points on that slice
      (4) physics (collocation) points on that slice

    Legend requirements:
      - Panel 2: "1: edge" and "0: bulk"
      - Panel 3: "+1" and "-1"
      - Panel 4: "collocation point"
    """

    COL_EDGE = "tab:olive"
    COL_BULK = "tab:purple"
    COL_POS  = "tab:red"
    COL_NEG  = "tab:blue"
    COL_PHYS = "tab:green"


    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    vol = np.asarray(cryoet)
    assert vol.ndim == 3, f"cryoet must be 3D, got {vol.ndim}D"
    Z, Y, X = vol.shape

    axis_to_dim = {"z": 0, "y": 1, "x": 2}
    if axis not in axis_to_dim:
        raise ValueError(f"axis must be one of {list(axis_to_dim.keys())}, got {axis}")
    dim = axis_to_dim[axis]

    if slice_index is None:
        slice_index = vol.shape[dim] // 2

    # ----- helpers -----
    def _to_numpy(a):
        try:
            return np.asarray(a)
        except Exception:
            return np.array(a)

    def points_to_voxel_indices(pts_xyz):
        """
        Map points to integer voxel indices (z,y,x).
        Supports either normalized [-1,1] or already-in-voxel coordinates.
        Assumes points are provided as (x, y, z).
        """
        pts = _to_numpy(pts_xyz).astype(np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must have shape (N,3), got {pts.shape}")

        in_norm = np.mean((pts >= -1.05) & (pts <= 1.05))
        if in_norm > 0.95:
            x_n, y_n, z_n = pts[:, 0], pts[:, 1], pts[:, 2]
            x_i = np.rint((x_n + 1) * 0.5 * (X - 1)).astype(int)
            y_i = np.rint((y_n + 1) * 0.5 * (Y - 1)).astype(int)
            z_i = np.rint((z_n + 1) * 0.5 * (Z - 1)).astype(int)
        else:
            x_i = np.rint(pts[:, 0]).astype(int)
            y_i = np.rint(pts[:, 1]).astype(int)
            z_i = np.rint(pts[:, 2]).astype(int)

        x_i = np.clip(x_i, 0, X - 1)
        y_i = np.clip(y_i, 0, Y - 1)
        z_i = np.clip(z_i, 0, Z - 1)
        return np.stack([z_i, y_i, x_i], axis=1)

    def get_slice(arr3d):
        return np.take(arr3d, slice_index, axis=dim)

    def scatter_coords_for_plane(zyx_idx):
        z_i, y_i, x_i = zyx_idx[:, 0], zyx_idx[:, 1], zyx_idx[:, 2]
        if dim == 0:      # slicing z -> image is (Y, X)
            row, col = y_i, x_i
        elif dim == 1:    # slicing y -> image is (Z, X)
            row, col = z_i, x_i
        else:             # slicing x -> image is (Z, Y)
            row, col = z_i, y_i
        return row, col

    def filter_points_on_slice(zyx_idx):
        return zyx_idx[zyx_idx[:, dim] == slice_index]

    def _legend_handle_for_scatter(sc, label, marker="x"):
        # For "x" markers, use line color + markeredgewidth (facecolor is ignored)
        col = sc.get_edgecolors()
        if col is None or len(col) == 0:
            col = sc.get_facecolors()
        color = "k" if col is None or len(col) == 0 else col[0]

        lw = sc.get_linewidths()
        mew = 1.5 if lw is None or len(lw) == 0 else float(lw[0])

        return Line2D(
            [0], [0],
            marker=marker,
            linestyle="None",
            color=color,              # THIS is what colors an "x"
            markeredgewidth=mew,      # thickness of the "x"
            markersize=max(4, np.sqrt(s)),  # roughly match scatter size
            label=label,
        )

    # ----- slice -----
    cryo_slice = get_slice(vol)

    plt.style.reload_library()
    with plt.style.context(["base", "transparent"]):
        fig, axes = plt.subplots(4, 1, figsize=(2, 8), constrained_layout=True)


        # Panel 1: cryo
        axes[0].imshow(cryo_slice, cmap="gray_r")
        axes[0].set_axis_off()
        if show_titles:
            axes[0].set_title("Original Image")

        # Panel 2: edges (legend "1: edge" / "0: bulk")
        axes[1].imshow(cryo_slice, cmap="gray_r", alpha=alpha_cryo_overlay)
        if edges is not None:
            e_pts = points_to_voxel_indices(edges["points"])
            e_val = _to_numpy(edges["label"]).reshape(-1)
            if e_val.shape[0] != e_pts.shape[0]:
                raise ValueError(f"edges['label'] length {e_val.shape[0]} != points {e_pts.shape[0]}")
            mask = (e_pts[:, dim] == slice_index)
            e_pts_s = e_pts[mask]
            e_val_s = e_val[mask]

            if e_pts_s.shape[0] > 0:
                r, c = scatter_coords_for_plane(e_pts_s)

                # Split into edge (==1) and bulk (==0) so the legend can show desired labels
                m1 = (e_val_s == 1) | (e_val_s == 1.0)
                m0 = (e_val_s == 0) | (e_val_s == 0.0)

                handles = []
                if np.any(m1):
                    sc1 = axes[1].scatter(c[m1], r[m1], s=s, alpha=alpha_pts, color=COL_EDGE, marker="x")
                    handles.append(_legend_handle_for_scatter(sc1, "edge"))
                if np.any(m0):
                    sc0 = axes[1].scatter(c[m0], r[m0], s=s, alpha=alpha_pts, color=COL_BULK, marker="x")
                    handles.append(_legend_handle_for_scatter(sc0, "bulk"))

                if show_colorbars:
                    # If you really want a colorbar, use the original scalar scatter:
                    sc = axes[1].scatter(c, r, s=s, alpha=alpha_pts, c=e_val_s)
                    plt.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.02)

                if handles:
                    axes[1].legend(handles=handles, loc="lower right", frameon=True)

        axes[1].set_axis_off()
        if show_titles:
            axes[1].set_title("Edge Signals")

        # Panel 3: signs (legend "+1" / "-1")
        axes[2].imshow(cryo_slice, cmap="gray_r", alpha=alpha_cryo_overlay)
        if signs is not None:
            s_pts = points_to_voxel_indices(signs["points"])
            s_val = _to_numpy(signs["label"]).reshape(-1)
            if s_val.shape[0] != s_pts.shape[0]:
                raise ValueError(f"signs['label'] length {s_val.shape[0]} != points {s_pts.shape[0]}")
            mask = (s_pts[:, dim] == slice_index)
            s_pts_s = s_pts[mask]
            s_val_s = s_val[mask]

            if s_pts_s.shape[0] > 0:
                r, c = scatter_coords_for_plane(s_pts_s)

                mp = (s_val_s > 0)  # treat >0 as +1
                mn = (s_val_s < 0)  # treat <0 as -1

                handles = []
                if np.any(mp):
                    scp = axes[2].scatter(c[mp], r[mp], s=s, alpha=alpha_pts, color=COL_POS, marker="x")
                    handles.append(_legend_handle_for_scatter(scp, "+1"))
                if np.any(mn):
                    scn = axes[2].scatter(c[mn], r[mn], s=s, alpha=alpha_pts, color=COL_NEG, marker="x")
                    handles.append(_legend_handle_for_scatter(scn, "-1"))

                if show_colorbars:
                    sc = axes[2].scatter(c, r, s=s, alpha=alpha_pts, c=s_val_s)
                    plt.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.02)

                if handles:
                    axes[2].legend(handles=handles, loc="lower right", frameon=True)

        axes[2].set_axis_off()
        if show_titles:
            axes[2].set_title("Sign Signals")

        # Panel 4: physics points (legend "collocation point")
        axes[3].imshow(cryo_slice, cmap="gray_r", alpha=alpha_cryo_overlay)
        if phys is not None:
            p_pts = points_to_voxel_indices(phys["points"])
            p_pts_s = filter_points_on_slice(p_pts)
            if p_pts_s.shape[0] > 0:
                r, c = scatter_coords_for_plane(p_pts_s)
                scp = axes[3].scatter(c, r, s=s, alpha=alpha_pts, color=COL_PHYS, marker="x")
                h = _legend_handle_for_scatter(scp, "collocation point")
                axes[3].legend(handles=[h], loc="lower right", frameon=True)

        axes[3].set_axis_off()
        if show_titles:
            axes[3].set_title("Collocation Points")

        for ax in axes:
            ax.invert_yaxis()

    return fig, axes


def plot_sampling_points_2x2_binary(
    vol_shape,        # (Z, Y, X) or cryoet.shape
    edges,            # {"points": (N,3), "label": (N,)}  label: 1=edge, 0=bulk
    signs,            # {"points": (M,3), "label": (M,)}  label: +1 / -1
    axis="z",
    slice_index=None,
    s=12,             # marker size
    show_title=False,
):
    """
    2x2 figure:
      [ edge | bulk ]
      [ +1   | -1   ]

    - white background
    - black square markers
    - no cryoET image
    """

    import numpy as np
    import matplotlib.pyplot as plt

    def _set_plane_limits(ax):
        # For the chosen slicing axis, define the displayed plane size.
        if dim == 0:      # z-slice -> (Y, X) shown as x in [0, X-1], y in [0, Y-1]
            ax.set_xlim(-0.5, X - 0.5)
            ax.set_ylim(-0.5, Y - 0.5)
        elif dim == 1:    # y-slice -> (Z, X)
            ax.set_xlim(-0.5, X - 0.5)
            ax.set_ylim(-0.5, Z - 0.5)
        else:             # x-slice -> (Z, Y)
            ax.set_xlim(-0.5, Y - 0.5)
            ax.set_ylim(-0.5, Z - 0.5)

        ax.set_aspect("equal", adjustable="box")


    # ---------------- basic setup ----------------
    Z, Y, X = vol_shape
    axis_to_dim = {"z": 0, "y": 1, "x": 2}
    dim = axis_to_dim[axis]

    if slice_index is None:
        slice_index = vol_shape[dim] // 2

    def _to_numpy(a):
        return np.asarray(a)

    def points_to_voxel_indices(pts_xyz):
        """(x,y,z) -> (z,y,x), supports [-1,1] or voxel coords"""
        pts = _to_numpy(pts_xyz).astype(float)

        in_norm = np.mean((pts >= -1.05) & (pts <= 1.05))
        if in_norm > 0.95:
            x = np.rint((pts[:, 0] + 1) * 0.5 * (X - 1))
            y = np.rint((pts[:, 1] + 1) * 0.5 * (Y - 1))
            z = np.rint((pts[:, 2] + 1) * 0.5 * (Z - 1))
        else:
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        x = np.clip(x.astype(int), 0, X - 1)
        y = np.clip(y.astype(int), 0, Y - 1)
        z = np.clip(z.astype(int), 0, Z - 1)

        return np.stack([z, y, x], axis=1)

    def scatter_coords(zyx):
        z, y, x = zyx[:, 0], zyx[:, 1], zyx[:, 2]
        if dim == 0:
            return x, y
        elif dim == 1:
            return x, z
        else:
            return y, z

    # ---------------- figure ----------------
    plt.style.reload_library()
    with plt.style.context(["base", "transparent"]):    
        # fig, axes = plt.subplots(2, 2, figsize=(1, 1), constrained_layout=True)
        fig, axes = plt.subplots(2, 2, figsize=(2, 2))
        axes = axes.ravel()

        # fig.set_constrained_layout_pads(
        #     w_pad=0.01,   # horizontal space between columns (inches)
        #     h_pad=0.01,   # vertical space between rows (inches)
        #     wspace=-0.0,
        #     hspace=-1,
        # )

        fig.subplots_adjust(
            wspace=0.1,   # width space (fraction of axes)
            hspace=0.1,   # height space
        )


        titles = ["Edge", "Bulk", "Inside", "Outside"]

        for ax, title in zip(axes, titles):
            ax.set_facecolor("white")
            if show_title:
                ax.set_title(title)
            ax.set_axis_off()

        # ---------------- edge / bulk ----------------
        e_pts = points_to_voxel_indices(edges["points"])
        e_val = _to_numpy(edges["label"]).reshape(-1)

        mask = e_pts[:, dim] == slice_index
        e_pts_s = e_pts[mask]
        e_val_s = e_val[mask]

        if e_pts_s.size:
            x, y = scatter_coords(e_pts_s)

            m_edge = e_val_s == 1
            m_bulk = e_val_s == 0

            axes[0].scatter(x[m_edge], y[m_edge], s=s, c="black", marker="s")
            axes[1].scatter(x[m_bulk], y[m_bulk], s=s, c="black", marker="s")

        # ---------------- sign (+ / -) ----------------
        s_pts = points_to_voxel_indices(signs["points"])
        s_val = _to_numpy(signs["label"]).reshape(-1)

        mask = s_pts[:, dim] == slice_index
        s_pts_s = s_pts[mask]
        s_val_s = s_val[mask]

        if s_pts_s.size:
            x, y = scatter_coords(s_pts_s)

            axes[2].scatter(x[s_val_s > 0], y[s_val_s > 0], s=s, c="black", marker="s")
            axes[3].scatter(x[s_val_s < 0], y[s_val_s < 0], s=s, c="black", marker="s")

        for ax in axes:
            _set_plane_limits(ax)
            # ax.invert_yaxis()

    return fig, axes


def plot_mrc(cryoet, axis="z", slice_index=None, show_title=False):
    import numpy as np
    import matplotlib.pyplot as plt

    vol = np.asarray(cryoet)
    assert vol.ndim == 3, f"cryoet must be 3D, got {vol.ndim}D"
    Z, Y, X = vol.shape

    axis_to_dim = {"z": 0, "y": 1, "x": 2}
    if axis not in axis_to_dim:
        raise ValueError(f"axis must be one of {list(axis_to_dim.keys())}, got {axis}")
    dim = axis_to_dim[axis]

    if slice_index is None:
        slice_index = vol.shape[dim] // 2

    cryo_slice = np.take(vol, slice_index, axis=dim)

    plt.style.reload_library()
    with plt.style.context(["base", "transparent"]):
        fig, ax = plt.subplots(1, 1, figsize=(0.8, 0.8), constrained_layout=True)

        ax.imshow(cryo_slice, cmap="gray_r")
        ax.set_axis_off()

        if show_title:
            ax.set_title("Original Image")

        ax.invert_yaxis()

    return fig, ax



import numpy as np
from scipy import ndimage as ndi

def keep_if_enough_neighbors(I, k=6, radius=1, connectivity="26", include_center=False):
    """
    Keep I==1 voxels if they have at least k neighboring ones.

    radius=1 -> 3x3x3 neighborhood
    connectivity:
      "6"  : face neighbors only
      "26" : full 3x3x3 cube neighbors
    include_center: whether to count the voxel itself in the neighbor count
    """
    I = (I > 0).astype(np.uint8)

    if connectivity == "26":
        se = np.ones((2*radius+1,)*3, dtype=np.uint8)
    elif connectivity == "6":
        se = ndi.generate_binary_structure(3, 1).astype(np.uint8)  # center + 6 neighbors
        if radius != 1:
            raise ValueError("connectivity='6' currently supports radius=1 only.")
    else:
        raise ValueError("connectivity must be '6' or '26'")

    if not include_center:
        se = se.copy()
        se[radius, radius, radius] = 0  # don't count self

    neighbor_count = ndi.convolve(I, se, mode="constant", cval=0)
    out = (I == 1) & (neighbor_count >= k)
    return out.astype(np.uint8)







import numpy as np
import matplotlib.pyplot as plt

def _norm_xyz_to_zyx_vox(points_xyz, shape_zyx):
    """
    Convert normalized xyz in [-1,1] to voxel coordinates (z,y,x) in [0..Z-1],[0..Y-1],[0..X-1].
    """
    pts = np.asarray(points_xyz, dtype=np.float32)
    Z, Y, X = shape_zyx

    x = pts[:, 0]; y = pts[:, 1]; z = pts[:, 2]
    ix = np.clip(((x + 1.0) * 0.5 * (X - 1)), 0, X - 1)
    iy = np.clip(((y + 1.0) * 0.5 * (Y - 1)), 0, Y - 1)
    iz = np.clip(((z + 1.0) * 0.5 * (Z - 1)), 0, Z - 1)

    # return (N,3) as float voxel coords
    return np.stack([iz, iy, ix], axis=1)

def overlay_cryoet_with_points(
    cryoet,
    edge=None,
    sign=None,
    axis="z",                 # 'z','y','x'
    slice_indices=None,       # list/array; if None -> 6 evenly spaced
    edge_label=1,             # keep edge points with this label
    sign_pos_label=1,         # positive sign label
    sign_neg_label=-1,        # negative sign label
    tol=0.5,                  # slice thickness in voxels for selecting points
    s_edge=2,                 # scatter size
    s_sign=10,
    alpha_edge=0.6,
    alpha_sign=0.9,
    vmin=None, vmax=None,
    figsize_per_panel=(4, 4),
    show_title=True,
):
    """
    Overlay a 3D cryo-ET volume (Z,Y,X) with point clouds (normalized xyz in [-1,1]) on selected slices.

    edge: dict with keys 'points' (N,3 xyz), 'label' (N,)
    sign: dict with keys 'points' (M,3 xyz), 'label' (M,) where +/-1 indicate inside/outside (or your convention)

    Points are projected to the displayed plane, and selected if they are within `tol` voxels of the slice index.
    """
    vol = np.asarray(cryoet)
    if vol.ndim != 3:
        raise ValueError(f"cryoet must be 3D (Z,Y,X), got shape {vol.shape}")

    Z, Y, X = vol.shape
    axis = axis.lower()
    if axis not in ("z", "y", "x"):
        raise ValueError("axis must be one of 'z','y','x'")

    # choose default slices
    if slice_indices is None:
        n = 6
        if axis == "z":
            slice_indices = np.linspace(0, Z - 1, n).round().astype(int)
        elif axis == "y":
            slice_indices = np.linspace(0, Y - 1, n).round().astype(int)
        else:
            slice_indices = np.linspace(0, X - 1, n).round().astype(int)
    else:
        slice_indices = np.asarray(slice_indices, dtype=int)

    # prep edge points in voxel coords
    edge_zyx = None
    if edge is not None:
        ep = np.asarray(edge["points"])
        el = np.asarray(edge.get("label", np.ones(len(ep))))
        ep = ep[el == edge_label]
        if len(ep) > 0:
            edge_zyx = _norm_xyz_to_zyx_vox(ep, vol.shape)

    # prep sign points in voxel coords (pos/neg separated)
    sign_pos_zyx = sign_neg_zyx = None
    if sign is not None:
        sp = np.asarray(sign["points"])
        sl = np.asarray(sign.get("label", np.ones(len(sp))))
        sp_pos = sp[sl == sign_pos_label]
        sp_neg = sp[sl == sign_neg_label]
        if len(sp_pos) > 0:
            sign_pos_zyx = _norm_xyz_to_zyx_vox(sp_pos, vol.shape)
        if len(sp_neg) > 0:
            sign_neg_zyx = _norm_xyz_to_zyx_vox(sp_neg, vol.shape)

    # figure layout
    n = len(slice_indices)
    fig_w = figsize_per_panel[0] * n
    fig_h = figsize_per_panel[1]
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]

    def _slice(vol, axis, idx):
        if axis == "z":
            return vol[idx, :, :]
        if axis == "y":
            return vol[:, idx, :]
        return vol[:, :, idx]  # 'x'

    # helper: select points near slice + project to 2D coords
    def _select_and_project(pts_zyx, axis, idx, tol):
        if pts_zyx is None or len(pts_zyx) == 0:
            return None
        if axis == "z":
            m = np.abs(pts_zyx[:, 0] - idx) <= tol
            # image axes: y (vertical), x (horizontal)
            return pts_zyx[m][:, [2, 1]]  # -> (x, y)
        if axis == "y":
            m = np.abs(pts_zyx[:, 1] - idx) <= tol
            # image axes: z (vertical), x (horizontal)
            return pts_zyx[m][:, [2, 0]]  # -> (x, z)
        # axis == 'x'
        m = np.abs(pts_zyx[:, 2] - idx) <= tol
        # image axes: z (vertical), y (horizontal)
        return pts_zyx[m][:, [1, 0]]      # -> (y, z) but for scatter we want (x_horizontal, y_vertical)
        # NOTE: for x-slices, horizontal is y and vertical is z.

    for ax, idx in zip(axes, slice_indices):
        img = _slice(vol, axis, idx)
        ax.imshow(img, cmap="gray_r", vmin=vmin, vmax=vmax, interpolation="nearest", alpha=0.8)

        # edge overlay
        e2d = _select_and_project(edge_zyx, axis, idx, tol)
        if e2d is not None and len(e2d) > 0:
            ax.scatter(e2d[:, 0], e2d[:, 1], s=s_edge, c="yellow", marker="o", alpha=alpha_edge)

        # sign overlays
        p2d = _select_and_project(sign_pos_zyx, axis, idx, tol)
        if p2d is not None and len(p2d) > 0:
            ax.scatter(p2d[:, 0], p2d[:, 1], s=s_sign, c="red", marker="o", alpha=alpha_sign)

        n2d = _select_and_project(sign_neg_zyx, axis, idx, tol)
        if n2d is not None and len(n2d) > 0:
            ax.scatter(n2d[:, 0], n2d[:, 1], s=s_sign, c="blue", marker="o", alpha=alpha_sign)

        ax.set_axis_off()
        if show_title:
            ax.set_title(f"{axis}={idx}")

    plt.tight_layout()
    return fig, axes



def overlay_cryoet_with_points_ax(
    ax,
    cryoet,
    slice_index,
    axis="z",                 # 'z','y','x'
    edge=None,
    sign=None,
    edge_label=1,             # kept for backward compatibility (edge==1)
    sign_pos_label=1,
    sign_neg_label=-1,
    tol=0.5,
    s_edge=1,
    s_sign=1,
    alpha_edge=0.6,
    alpha_sign=0.9,
    vmin=None, vmax=None,
    figsize=(4, 4),
    show_title=True,
    title=None,
):
    """
    Plot ONE slice of a 3D cryo-ET volume and optionally overlay edge/sign points.

    Edge points with label==1 and label==0 are BOTH plotted.
    """

    vol = np.asarray(cryoet)
    if vol.ndim != 3:
        raise ValueError(f"cryoet must be 3D (Z,Y,X), got shape {vol.shape}")

    Z, Y, X = vol.shape
    axis = axis.lower()
    if axis not in ("z", "y", "x"):
        raise ValueError("axis must be one of 'z','y','x'")

    max_idx = {"z": Z - 1, "y": Y - 1, "x": X - 1}[axis]
    if not (0 <= int(slice_index) <= max_idx):
        raise ValueError(f"slice_index out of bounds for axis='{axis}'")
    idx = int(slice_index)

    # --- helpers ---
    def _slice(vol_, axis_, i_):
        if axis_ == "z":
            return vol_[i_, :, :]
        if axis_ == "y":
            return vol_[:, i_, :]
        return vol_[:, :, i_]

    def _get_labels(d, n):
        if "label" in d and d["label"] is not None:
            return np.asarray(d["label"])
        if "sign" in d and d["sign"] is not None:
            return np.asarray(d["sign"])
        return np.ones(n)

    def _select_and_project(pts_zyx, axis_, i_, tol_):
        if pts_zyx is None or len(pts_zyx) == 0:
            return None

        if axis_ == "z":
            m = np.abs(pts_zyx[:, 0] - i_) <= tol_
            return pts_zyx[m][:, [2, 1]]  # (x, y)

        if axis_ == "y":
            m = np.abs(pts_zyx[:, 1] - i_) <= tol_
            return pts_zyx[m][:, [2, 0]]  # (x, z)

        m = np.abs(pts_zyx[:, 2] - i_) <= tol_
        return pts_zyx[m][:, [1, 0]]     # (y, z)

    # --- prepare edge points (BOTH labels) ---
    edge1_zyx = edge0_zyx = None
    if edge is not None:
        ep = np.asarray(edge["points"])
        el = _get_labels(edge, len(ep))
        ep_zyx = _norm_xyz_to_zyx_vox(ep, vol.shape)

        if np.any(el == 1):
            edge1_zyx = ep_zyx[el == 1]
        if np.any(el == 0):
            edge0_zyx = ep_zyx[el == 0]

    # --- prepare sign points ---
    sign_pos_zyx = sign_neg_zyx = None
    if sign is not None:
        sp = np.asarray(sign["points"])
        sl = _get_labels(sign, len(sp))
        sp_zyx = _norm_xyz_to_zyx_vox(sp, vol.shape)

        if np.any(sl == sign_pos_label):
            sign_pos_zyx = sp_zyx[sl == sign_pos_label]
        if np.any(sl == sign_neg_label):
            sign_neg_zyx = sp_zyx[sl == sign_neg_label]

    # --- plot image ---
    img = _slice(vol, axis, idx)
    custom_gray = LinearSegmentedColormap.from_list(
        "custom_gray", ["#f0f0f0", "#777777"]
    )
    ax.imshow(img, cmap=custom_gray, vmin=vmin, vmax=vmax,
              interpolation="nearest", alpha=1.0)

    # --- edge overlays ---
    if edge1_zyx is not None:
        e2d = _select_and_project(edge1_zyx, axis, idx, tol)
        if e2d is not None and len(e2d) > 0:
            ax.scatter(e2d[:, 0], e2d[:, 1],
                       s=s_edge, c="red", alpha=alpha_edge,
                       marker="o", label="edge=1")

    if edge0_zyx is not None:
        e2d = _select_and_project(edge0_zyx, axis, idx, tol)
        if e2d is not None and len(e2d) > 0:
            ax.scatter(e2d[:, 0], e2d[:, 1],
                       s=s_edge, c="blue", alpha=alpha_edge,
                       marker="o", label="edge=0")

    # --- sign overlays ---
    if sign_pos_zyx is not None:
        p2d = _select_and_project(sign_pos_zyx, axis, idx, tol)
        if p2d is not None and len(p2d) > 0:
            ax.scatter(p2d[:, 0], p2d[:, 1],
                       s=s_sign, c="red", alpha=alpha_sign,
                       marker="o", label="sign +")

    if sign_neg_zyx is not None:
        n2d = _select_and_project(sign_neg_zyx, axis, idx, tol)
        if n2d is not None and len(n2d) > 0:
            ax.scatter(n2d[:, 0], n2d[:, 1],
                       s=s_sign, c="blue", alpha=alpha_sign,
                       marker="o", label="sign -")

    ax.set_axis_off()
    if show_title:
        ax.set_title(title if title is not None else f"{axis}={idx}")



