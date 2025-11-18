import numpy as np
import jax
import jax.numpy as jnp
from pinn.model import PINN
from pinn.model import phase_volume, phase_surface, phase_bend
import mrcfile
from skimage import measure
import trimesh
import matplotlib.pyplot as plt

def dice_loss_pinn_vs_mask(mask_file_path, checkpoint, band=(-0.8, 0.8), verbose=True):
    from pinn.grid import phi_on_cryo_grid_xyz
    from pinn.plot import _to_zyx
    with mrcfile.open(mask_file_path, permissive=True) as mrc:
        gt_mask = np.asarray(mrc.data, dtype=np.float32)
    gt_mask = np.where(gt_mask > 0.5, 1, 0)

    state = checkpoint["state"]; params = state["params"]
    model = PINN()
    phi_fn = lambda x: model.apply(params, x.reshape(-1, 3))
    phi_xyz, _, _ = phi_on_cryo_grid_xyz(phi_fn, gt_mask.shape, lo=-1.0, hi=1.0)
    phi_zyx = _to_zyx(np.array(phi_xyz))  
    lo, hi = band
    pred_mask = (phi_zyx > lo) & (phi_zyx < hi)

    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}")
    
    intersection = np.count_nonzero(gt_mask & pred_mask)
    gt_sum = np.count_nonzero(gt_mask)
    pd_sum = np.count_nonzero(pred_mask)
    volume_sum = gt_sum + pd_sum
    dice = (2.0 * intersection) / (volume_sum + 1e-8)

    fig, axes = plt.subplots(3, 4, figsize=(12, 12))  # 4행 3열

    # 슬라이스 인덱스 계산
    x_indices = np.linspace(0, gt_mask.shape[2] - 1, 6, dtype=int)
    y_indices = np.linspace(0, gt_mask.shape[1] - 1, 6, dtype=int)
    z_indices = np.linspace(0, gt_mask.shape[0] - 1, 6, dtype=int)

    # 각 방향별 슬라이스 시각화
    for i in range(1, 5):
        # Z-slice
        z = z_indices[i]
        gt_z = gt_mask[z, :, :]
        pd_z = pred_mask[z, :, :]
        axes[0, i-1].imshow(gt_z, cmap="gray", alpha=1.0)
        axes[0, i-1].imshow(pd_z, cmap="Blues", alpha=0.6)
        axes[0, i-1].set_title(f"Z-slice {z}")
        axes[0, i-1].axis("off")

        # Y-slice
        y = y_indices[i]
        gt_y = gt_mask[:, y, :]
        pd_y = pred_mask[:, y, :]
        axes[1, i-1].imshow(gt_y, cmap="gray", alpha=1.0)
        axes[1, i-1].imshow(pd_y, cmap="Blues", alpha=0.6)
        axes[1, i-1].set_title(f"Y-slice {y}")
        axes[1, i-1].axis("off")

        # X-slice
        x = x_indices[i]
        gt_x = gt_mask[:, :, x]
        pd_x = pred_mask[:, :, x]
        axes[2, i-1].imshow(gt_x, cmap="gray", alpha=1.0)
        axes[2, i-1].imshow(pd_x, cmap="Blues", alpha=0.6)
        axes[2, i-1].set_title(f"X-slice {x}")
        axes[2, i-1].axis("off")

    plt.suptitle("White = GT Mask, Blue = Prediction", fontsize=16)
    plt.tight_layout()
    plt.show()


    if verbose:
        print(f"intersection: {intersection}")
        print(f"intersection/mask: {intersection / gt_sum}")
        print(f"gt: {gt_sum} / {gt_mask.size}")
        print(f"pd: {pd_sum} / {pred_mask.size}")
        print(f"volume_sum: {volume_sum}")
        print(f"Dice: {dice:.6f}")

    return 1.0 - dice


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

    # Prediction membrane mask: band around phi=0
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

    print("intersection: ", intersection)
    print("gt: ", np.sum(gt_mask))
    print("pd: ", np.sum(pred_mask))
    print("volume_sum: ", volume_sum)

    return 1 - dice   # Dice loss



def get_masks(mrc_path, dat_seg, grid_size=64, threshold=0.8, band_thickness=0.05):
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

    # # mask using band_thickness
    # pred_mask = (np.abs(phi_vals) < band_thickness).astype(np.uint8)

    # mask using marching cube
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

    return gt_mask, pred_mask



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
