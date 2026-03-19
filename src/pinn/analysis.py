import numpy as np
import jax
import jax.numpy as jnp
from pinn.model import PINN
from pinn.model import phase_volume, phase_surface, phase_bend
import mrcfile
from skimage import measure
import trimesh
from pinn.cryoet_io import load_mrc_data
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import LogLocator


def calc_area_seg(ckpt_dat, epsilon, sampling, grid_size=64, num_point=10000):
	
	if sampling == "grid":
		x = jnp.linspace(-1, 1, grid_size)
		y = jnp.linspace(-1, 1, grid_size)
		z = jnp.linspace(-1, 1, grid_size)
		sample_points = jnp.stack(jnp.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
	elif sampling == "random":
		key = jax.random.PRNGKey(0)
		sample_points = jax.random.uniform(key, (num_point, 3), minval=-1, maxval=1)

	state  = ckpt_dat["state"]
	params = state["params"]
	model  = PINN()
	phi_fn = lambda x: model.apply(params, x)

	value = phase_surface(phi_fn, sample_points, epsilon)

	return value


def calc_bend_mem(mrc_path, iso_value=None, rescale=True, verbose=True):
## IT DOES NOT LOOK CORRECT ##

    """
    Calculate bending energy (∫ H^2 dS) of a membrane surface 
    extracted from an MRC volume.

    Args:
        mrc_path (str): Path to the .mrc file
        iso_value (float): Threshold for marching cubes (default: mean of volume)
        rescale (bool): If True, rescales box to [-1,1]^3
        verbose (bool): If True, prints diagnostics

    Returns:
        float: Estimated bending energy
    """
    # --- Load MRC ---
    with mrcfile.open(mrc_path) as mrc:
        volume = mrc.data.astype(np.float32)
    nx, ny, nz = volume.shape

    if iso_value is None:
        iso_value = float(np.mean(volume))

    # --- Extract surface with marching cubes ---
    verts, faces, _, _ = measure.marching_cubes(volume, level=iso_value)

    # Rescale vertices to [-1,1]^3
    if rescale:
        verts[:, 0] = 2.0 * verts[:, 0] / nx - 1.0
        verts[:, 1] = 2.0 * verts[:, 1] / ny - 1.0
        verts[:, 2] = 2.0 * verts[:, 2] / nz - 1.0

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # --- Compute bending energy ---
    V = mesh.vertices
    F = mesh.faces
    face_areas = mesh.area_faces

    n_vertices = len(V)
    W = np.zeros((n_vertices, n_vertices))
    A_mixed = np.zeros(n_vertices)

    for tri, area in zip(F, face_areas):
        i, j, k = tri
        vi, vj, vk = V[i], V[j], V[k]

        # edge vectors
        e_ij = vi - vj
        e_jk = vj - vk
        e_ki = vk - vi

        # avoid division by zero
        def cot(a, b):
            cross = np.cross(a, b)
            return np.dot(a, b) / (np.linalg.norm(cross) + 1e-12)

        cot_alpha = cot(vj - vi, vk - vi)
        cot_beta  = cot(vi - vj, vk - vj)
        cot_gamma = cot(vi - vk, vj - vk)

        # symmetric weights
        W[i, j] += cot_gamma; W[j, i] += cot_gamma
        W[j, k] += cot_alpha; W[k, j] += cot_alpha
        W[k, i] += cot_beta;  W[i, k] += cot_beta

        # distribute face area
        for v in tri:
            A_mixed[v] += area / 3.0

    # Laplace-Beltrami
    H_vec = np.zeros((n_vertices, 3))
    for i in range(n_vertices):
        neighbors = np.nonzero(W[i])[0]
        for j in neighbors:
            H_vec[i] += W[i, j] * (V[i] - V[j])

    H = np.linalg.norm(H_vec, axis=1) / (2 * A_mixed + 1e-12)

    # ∫ H^2 dS
    energy = np.sum((H**2) * A_mixed)

    if verbose:
        print(f"Bending energy: {energy:.6f}")
        print(f"Surface area: {mesh.area:.6f}, Watertight: {mesh.is_watertight}")

    return energy


def calc_area_mem(mrc_path, iso_value=None):
    """
	Calculate the membrane surface area from an MRC volume using marching cubes.
	Because marching cubes produce a doubled membrane, the resulting area is divided by two.
	Before applying this correction, you should verify that the extracted surface is reasonable
	i.e., that the membrane is sufficiently thin so that the two surfaces truly represent a doubled membrane.
	This validation can be performed by visualizing the surface in the notebook analysis_synthetic.ipynb
    """

    # Load volume
    with mrcfile.open(mrc_path) as mrc:
        volume = mrc.data.astype(np.float32)
    nx, ny, nz = volume.shape

    # Choose iso-value
    if iso_value is None:
        iso_value = float(np.mean(volume))

    # Marching cubes in voxel space
    verts, faces, normals, values = measure.marching_cubes(volume, level=iso_value)

    # Rescale vertices to [-1,1]^3
    verts[:, 0] = 2.0 * verts[:, 0] / nx - 1.0
    verts[:, 1] = 2.0 * verts[:, 1] / ny - 1.0
    verts[:, 2] = 2.0 * verts[:, 2] / nz - 1.0

    # Mesh & surface area
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    
    return mesh.area / 2.0


def calc_bend_seg(ckpt_dat, epsilon, kappa, sampling, grid_size=64, num_point=10000):
	
	if sampling == "grid":
		x = jnp.linspace(-1, 1, grid_size)
		y = jnp.linspace(-1, 1, grid_size)
		z = jnp.linspace(-1, 1, grid_size)
		sample_points = jnp.stack(jnp.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
	elif sampling == "random":
		key = jax.random.PRNGKey(0)
		sample_points = jax.random.uniform(key, (num_point, 3), minval=-1, maxval=1)

	state  = ckpt_dat["state"]
	params = state["params"]
	model  = PINN()
	phi_fn = lambda x: model.apply(params, x)

	value = phase_bend(phi_fn, sample_points, epsilon, kappa)

	return value


def calc_vol_seg(ckpt_dat, sampling, grid_size=64, num_point=10000):
    
    if sampling == "grid":
        x = jnp.linspace(-1, 1, grid_size)
        y = jnp.linspace(-1, 1, grid_size)
        z = jnp.linspace(-1, 1, grid_size)
        sample_points = jnp.stack(jnp.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
    elif sampling == "random":
        key = jax.random.PRNGKey(0)
        sample_points = jax.random.uniform(key, (num_point, 3), minval=-1, maxval=1)

    state  = ckpt_dat["state"]
    params = state["params"]
    model  = PINN()
    phi_fn = lambda x: model.apply(params, x)

    value = phase_volume(phi_fn, sample_points)

    return value


def dice_loss(mrc_path, ckpt_dat, grid_size=64, threshold=0.8, band_thickness=0.05):
    """
    Compute Dice loss between ground-truth membrane (from MRC)
    and predicted membrane (phi=0 contour).
    
    Args:
        mrc_path (str): path to ground-truth .mrc file
        ckpt_dat: checkpoint dict containing segmentation model state
        grid_size (int): sampling resolution
        band_thickness (float): thickness around phi=0 for prediction mask

    Returns:
        float: Dice loss
    """
    # --- Ground truth from MRC ---
    gt_volume = load_mrc_data(mrc_path, grid_size=grid_size)
    gt_mask = (gt_volume > threshold).astype(np.uint8)

    # --- Prediction from phi ---
    state  = ckpt_dat["state"]
    params = state["params"]
    model  = PINN()
    phi_fn = lambda x: model.apply(params, x)

    # Sample voxel grid
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    z = np.linspace(-1, 1, grid_size)
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)

    phi_vals = phi_fn(grid).reshape(grid_size, grid_size, grid_size)

    # # Prediction membrane mask: band around phi=0
    # pred_mask = (np.abs(phi_vals) < band_thickness).astype(np.uint8)

    # Prediction membrane mask: using marching cube
    pitch = 2.0 / grid_size
    phi_vals = np.array(phi_vals)
    phi_vals = np.pad(phi_vals, 1, mode="edge")
    verts, faces, _, _ = measure.marching_cubes(
        phi_vals, level=0.0, spacing=(pitch, pitch, pitch)
    )
    verts = verts - 1.0 - pitch/2
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)  

    pred_mask = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    idx = ((verts + 1) / pitch).astype(int)  # map [-1,1] → [0, grid_size)
    for v in idx:
        if np.all((v >= 0) & (v < grid_size)):
            pred_mask[tuple(v)] = 1


    # --- Dice calculation ---
    intersection = np.sum(gt_mask * pred_mask)
    volume_sum = np.sum(gt_mask) + np.sum(pred_mask)
    dice = (2. * intersection) / (volume_sum + 1e-8)

    # print("intersection: ", intersection)
    # print("gt: ", np.sum(gt_mask))
    # print("pd: ", np.sum(pred_mask))
    # print("volume_sum: ", volume_sum)

    return 1 - dice   # Dice loss


def surface_dice(
    mrc_path, ckpt_dat, grid_size=64, thre_sklt = 0.8, 
    threshold=0.8, band_thickness=0.05):
    """
    Compute surface Dice between ground-truth membrane (from MRC)
    and predicted membrane (phi=0 contour).
    
    Args:
        mrc_path (str): path to ground-truth .mrc file
        ckpt_dat: checkpoint dict containing segmentation model state
        grid_size (int): sampling resolution
        thre_sklt (float): threshold for skeletnization
        threshold (float): threshold for membrane mask
        band_thickness (float): thickness around phi=0 for prediction mask

    Returns:
        float: surface Dice loss (1 - Dice) <- this is None if marching cube failed
    """
    # --- Ground truth from MRC ---
    gt_volume = load_mrc_data(mrc_path, grid_size=grid_size)
    gt_sklt = (gt_volume > thre_sklt).astype(np.uint8)
    gt_mask = (gt_volume > threshold).astype(np.uint8)

    # --- Prediction from phi ---
    state  = ckpt_dat["state"]
    params = state["params"]
    model  = PINN()
    phi_fn = lambda x: model.apply(params, x)

    # Sample voxel grid
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    z = np.linspace(-1, 1, grid_size)
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)

    phi_vals = phi_fn(grid).reshape(grid_size, grid_size, grid_size)

    # Prediction membrane mask: band around phi=0
    pred_mask = (np.abs(phi_vals) < band_thickness).astype(np.uint8)

    # Prediction membrane mask: using marching cube
    pitch = 2.0 / grid_size
    phi_vals = np.array(phi_vals)
    phi_vals = np.pad(phi_vals, 1, mode="edge")
    # verts, faces, _, _ = measure.marching_cubes(
    #     phi_vals, level=0.0, spacing=(pitch, pitch, pitch)
    # )
    try:
        verts, faces, _, _ = measure.marching_cubes(
            phi_vals,
            level=0.0,
            spacing=(pitch, pitch, pitch),
        )
    except (ValueError, RuntimeError) as e:
        # marching cubes failed
        return None

    verts = verts - 1.0 - pitch/2
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)  

    pred_sklt = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    idx = ((verts + 1) / pitch).astype(int)  # map [-1,1] → [0, grid_size)
    for v in idx:
        if np.all((v >= 0) & (v < grid_size)):
            pred_sklt[tuple(v)] = 1

    # --- Dice calculation ---
    precision = np.sum(pred_sklt * gt_mask) / np.sum(pred_sklt)
    recall = np.sum(gt_sklt * pred_mask) / np.sum(gt_sklt)
    dice = 2 * (precision * recall) / (precision + recall)

    # print("gt_sklt: ", np.sum(gt_sklt))
    # print("gt_mask: ", np.sum(gt_mask))
    # print("pred_sklt: ", np.sum(pred_sklt))
    # print("pred_mask: ", np.sum(pred_mask))

    return 1 - dice


def get_masks(mrc_path, dat_seg, grid_size=64, thre_sklt=0.8, threshold=0.8, band_thickness=0.05):
    """
    Generate ground-truth and predicted masks for Dice evaluation.

    Args:
        mrc_path (str): Path to ground-truth MRC file
        dat_seg: checkpoint dict containing segmentation model state
        grid_size (int): Sampling resolution for prediction
        band_thickness (float): Thickness around φ=0 used to define membrane band

    Returns:
        (gt_mask, pred_mask): two 3D numpy arrays of shape (grid_size, grid_size, grid_size)
    """
    # --- Ground truth from MRC (resized to grid_size) ---
    gt_volume = load_mrc_data(mrc_path, grid_size=grid_size)
    gt_sklt = (gt_volume > thre_sklt).astype(np.uint8)
    gt_mask = (gt_volume > threshold).astype(np.uint8)

    # --- Prediction from φ ---
    state  = dat_seg["state"]
    params = state["params"]
    model  = PINN()
    phi_fn = lambda x: model.apply(params, x)

    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    z = np.linspace(-1, 1, grid_size)
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)

    phi_vals = phi_fn(grid).reshape(grid_size, grid_size, grid_size)

    # mask using band_thickness
    pred_mask = (np.abs(phi_vals) < band_thickness).astype(np.uint8)

    # mask using marching cube
    pitch = 2.0 / grid_size
    phi_vals = np.array(phi_vals)
    phi_vals = np.pad(phi_vals, 1, mode="edge")
    verts, faces, _, _ = measure.marching_cubes(
        phi_vals, level=0.0, spacing=(pitch, pitch, pitch)
    )
    verts = verts - 1.0 - pitch/2
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)  

    pred_sklt = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    idx = ((verts + 1) / pitch).astype(int)  # map [-1,1] → [0, grid_size)
    for v in idx:
        if np.all((v >= 0) & (v < grid_size)):
            pred_sklt[tuple(v)] = 1

    return gt_mask, gt_sklt, pred_mask, pred_sklt



def visualize_masks(gt_mask, pred_mask, slice_idx=None, axis=0):
    """
    Visualize 2D slices of ground-truth and predicted masks.

    Args:
        gt_mask (ndarray): Ground-truth mask (3D binary array).
        pred_mask (ndarray): Predicted mask (3D binary array).
        slice_idx (int): Which slice index to show (default: middle).
        axis (int): Axis along which to slice (0=z, 1=y, 2=x).
    """
    assert gt_mask.shape == pred_mask.shape, "GT and prediction must have same shape"
    shape = gt_mask.shape

    # Choose middle slice if none provided
    if slice_idx is None:
        slice_idx = shape[axis] // 2

    if axis == 0:   # z-slice
        gt_slice = gt_mask[slice_idx, :, :]
        pred_slice = pred_mask[slice_idx, :, :]
    elif axis == 1: # y-slice
        gt_slice = gt_mask[:, slice_idx, :]
        pred_slice = pred_mask[:, slice_idx, :]
    elif axis == 2: # x-slice
        gt_slice = gt_mask[:, :, slice_idx]
        pred_slice = pred_mask[:, :, slice_idx]
    else:
        raise ValueError("axis must be 0, 1, or 2")

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    axs[0].imshow(gt_slice, cmap="gray")
    axs[0].set_title("Ground Truth Mask")
    axs[0].axis("off")

    axs[1].imshow(pred_slice, cmap="gray")
    axs[1].set_title("Predicted Mask")
    axs[1].axis("off")

    # Overlay
    axs[2].imshow(gt_slice, cmap="gray")
    axs[2].imshow(pred_slice, cmap="jet", alpha=0.5)
    axs[2].set_title("Overlay")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


def load_accuracy_progress(path, step_shift=0):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")

    data = np.load(path, allow_pickle=True)
    steps = data["steps"].astype(np.int64) + step_shift

    out = {
        "steps":   steps,
        "dice":    data["dice"],
        "volume":  data["volume"],
        "area":    data["area"],
        "bending": data["bending"],
        "meta":    data["meta"].item(),  # stored as object
    }
    return out



def smooth(y, window=5):
    if window <= 1:
        return y

    y = np.asarray(y)
    pad = window // 2
    y_pad = np.pad(y, pad_width=pad, mode="reflect")
    kernel = np.ones(window) / window
    return np.convolve(y_pad, kernel, mode="valid")




def plot_accuracy_progress(
    curves,
    shape_list,
    lambda_2_list,
    legend_order=None,
    figsize=(14, 3.2),
    show_legend=True,
    smooth_window=7,
):
    """
    Parameters
    ----------
    curves : dict
        curves[(shape, lambda_2)] = {
            "steps", "dice", "dV", "dA", "dB"
        }

    shape_list : list of str
        Shapes to plot (defines color order)

    lambda_2_list : list of int
        Lambda_2 values (defines linestyle)

    legend_order : list of (shape, lambda_2), optional
        Explicit legend order. If None, defaults to shape-major order.

    Returns
    -------
    fig, axes
    """

    # ----------------------------
    # Figure / axes
    # ----------------------------
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=True)
    ax_dice, ax_vol, ax_area, ax_bend = axes

    for ax in axes:
        ax.set_xlabel("Step (×10³)")
        ax.grid(True, alpha=0.2)

    ax_dice.set_title("Surface Dice")
    ax_vol.set_title("Volume")
    ax_area.set_title("Surface Area")
    ax_bend.set_title("Bending Energy")

    ax_dice.set_ylabel("1 - Dice")
    ax_vol.set_ylabel(r"$\Delta V / V_0$")
    ax_area.set_ylabel(r"$\Delta A / A_0$")
    ax_bend.set_ylabel(r"$\Delta E_b / E_{b0}$")

    ax_dice.set_yticks([0, 0.2])
    ax_vol.set_yticks([0, 0.15])
    ax_area.set_yticks([0, 0.25])
    ax_bend.set_yticks([0, 13])

    ax_dice.set_ylim(-0.01, 0.21)
    # ax_dice.set_ylim(0, 0.2)
    # ax_vol.set_ylim(0, 1)
    # ax_area.set_ylim(0, 2)
    # ax_bend.set_ylim(0, 600)

    ax_dice.yaxis.labelpad = -4
    ax_vol.yaxis.labelpad  = -4
    ax_area.yaxis.labelpad = -4
    ax_bend.yaxis.labelpad = -6



    # ax_dice.set_yscale("log")
    # ax_vol.set_yscale("log")
    # ax_area.set_yscale("log")
    # ax_bend.set_yscale("log")

    # ----------------------------
    # Color map: one color per shape
    # ----------------------------
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {
        "biconcave": "red",  
        "bud_04":    "blue", 
        "multi":     "green",
    }    
    # color_map = {
    #     shape: default_colors[i % len(default_colors)]
    #     for i, shape in enumerate(shape_list)
    # }

    # ----------------------------
    # Legend order
    # ----------------------------
    if legend_order is None:
        legend_order = [
            (shape, lambda_2)
            for shape in shape_list
            for lambda_2 in lambda_2_list
        ]

    # ----------------------------
    # Plot
    # ----------------------------
    handles, labels = [], []

    for shape, lambda_2 in legend_order:
        # ls  = "--" if lambda_2 == 0 else "-"
        ls  = (0, (4, 2)) if lambda_2 == 0 else "-"
        # ls  = ":" if lambda_2 == 0 else "-"
        col = color_map[shape]

        c = curves[(shape, lambda_2)]
        x = c["steps"] / 1000.0
        lw = 0.6
        # smooth_window = 7

        y_dice = smooth(c["dice"], window=smooth_window)
        y_dV   = smooth(c["dV"],   window=smooth_window)
        y_dA   = smooth(c["dA"],   window=smooth_window)
        y_dB   = smooth(c["dB"],   window=smooth_window)        

        h, = ax_dice.plot(
            x, y_dice,
            linestyle=ls, color=col, linewidth=lw
        )
        ax_vol.plot(
            x, y_dV,
            linestyle=ls, color=col, linewidth=lw
        )
        ax_area.plot(
            x, y_dA,
            linestyle=ls, color=col, linewidth=lw
        )
        ax_bend.plot(
            x, y_dB,
            linestyle=ls, color=col, linewidth=lw
        )

        handles.append(h)
        labels.append(f"λ2={lambda_2}, {shape}")

    if show_legend:
        ax_dice.legend(handles, labels, loc="best", fontsize=9)

    for ax in axes:
        ax.grid(False)

        # vertical line for step = 10000
        ax.axvline(
            10,
            linestyle="--",
            color="k",
            linewidth=0.4,
            alpha=0.7,
        )

        ax.set_xticks([0, 10, 20])
        # ax.yaxis.labelpad = -2

        ax.set_title(ax.get_title(), pad=2)
        ax.set_xlabel(ax.get_xlabel(), labelpad=0)


    fig.subplots_adjust(wspace=0.6)

    return fig, axes



from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.ticker import MaxNLocator, ScalarFormatter

def plot_loss_panels(
    assembled_loss,
    shape_list,
    lambda_2_list,
    legend_order=None,
    figsize=(14, 3.2),
    logscale_keys=None,
    show_legend=True,
    smooth_window=7,
    linewidth=0.6,
):
    """
    1×3 loss panels:
      [data_loss | sign_loss | phys_loss]
    Legend occupies the previous 'total_loss' position on the right.
    Matches plot_accuracy_progress() appearance exactly.
    """

    loss_keys   = ["data_loss", "sign_loss", "phys_loss"]
    loss_titles = ["Data Loss", "Boundary Loss", "Physics Loss"]

    if logscale_keys is None:
        logscale_keys = []

    # --- same colors as plot_accuracy_progress ---
    color_map = {
        "biconcave": "red",
        "bud_04":    "blue",
        "multi":     "green",
        "czii_gl_1":  "black",
        "mend_3":     "black",
    }

    # --- same linestyle logic (λ2=0 dashed) ---
    def ls_for_lambda(lambda_2):
        return (0, (4, 2)) if int(lambda_2) == 0 else "-"

    if legend_order is None:
        legend_order = [
            (shape, lambda_2)
            for shape in shape_list
            for lambda_2 in lambda_2_list
        ]

    # ----------------------------
    # Figure / axes (1×3 panels)
    # ----------------------------
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)

    handles, labels = [], []

    for ax, key, title in zip(axes, loss_keys, loss_titles):

        for shape, lambda_2 in legend_order:
            hist = assembled_loss[shape][lambda_2]

            x = np.asarray(hist["step"]) / 1000.0
            y = np.asarray(hist[key])

            # ---- smoothing (same as accuracy plot) ----
            y_s = smooth(y, window=smooth_window)

            h, = ax.plot(
                x, y_s,
                linestyle=ls_for_lambda(lambda_2),
                color=color_map[shape],
                linewidth=linewidth,
            )

            # collect legend entries once
            if ax is axes[0]:
                handles.append(h)
                labels.append(f"λ2={lambda_2}, {shape}")

        # ---- styling (identical to accuracy plot) ----
        ax.set_title(title, pad=2)
        ax.set_xlabel("Step (×10³)", labelpad=0)
        ax.set_ylabel("Loss")
        ax.grid(False)

        if key in logscale_keys:
            ax.set_yscale("log")

        ax.axvline(
            10,
            linestyle="--",
            color="k",
            linewidth=0.4,
            alpha=0.7,
        )

        ax.set_xticks([0, 10, 20])
        # ax.yaxis.labelpad = 0


    # ----------------------------
    # Legend placed in the old 4th column
    # ----------------------------
    if show_legend:
        fig.legend(
            handles,
            [
                r"Closed, $\lambda_p=0$",
                r"Closed, $\lambda_p=10$",
                r"Open, $\lambda_p=0$",
                r"Open, $\lambda_p=10$",
                r"Multiple, $\lambda_p=0$",
                r"Multiple, $\lambda_p=10$",
            ],
            loc="center right",
            frameon=False,
            labelspacing=0.0,
        )

    # Leave space on the right for the legend
    # fig.subplots_adjust(wspace=0.6, right=0.7)
    fig.subplots_adjust(wspace=0.7, left=0.2)

    return fig, axes


