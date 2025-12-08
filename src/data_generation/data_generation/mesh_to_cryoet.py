import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from pathlib import Path
import mrcfile
import trimesh
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import label, generate_binary_structure, binary_dilation

from pathlib import Path
from typing import Sequence, Union


def load_membrane_mesh(input_file, frame=-1):
    """
    Load membrane coordinates and topology from a NetCDF file.

    Args:
        input_file (str or Path): Path to the input NetCDF (`traj.nc`).

    Returns:
        trimesh.Trimesh: A triangular mesh representing the membrane.
        np.ndarray: Nx3 array of vertex coordinates.
    """
    input_file = Path(input_file)
    data = Dataset(input_file, 'r')

    # Extract the last frame of coordinates
    coordinates = data.groups['Trajectory'].variables['coordinates'][frame]
    coords = coordinates.reshape(-1, 3)  # Convert to (N,3)

    # Extract topology (assuming it defines triangular faces)
    topology = data.groups['Trajectory'].variables['topology'][frame]
    faces = np.array(topology).reshape(-1, 3)  # Convert to (M,3)

    return trimesh.Trimesh(vertices=coords, faces=faces), coords


def generate_voxel_grid(mesh, coords, grid_size=128, margin_ratio=0.4):
    """
    Converts a membrane mesh into a voxelized representation.

    Args:
        mesh (trimesh.Trimesh): The triangular membrane mesh.
        coords (np.ndarray): The Nx3 array of vertex coordinates.
        grid_size (int): Size of the output voxel grid.
        margin_ratio (float): Extra padding to prevent cut-offs.

    Returns:
        np.ndarray: 3D voxel grid where membrane regions are marked as 1.
    """
    # Compute bounding box of membrane
    min_bound = np.min(coords, axis=0)
    max_bound = np.max(coords, axis=0)
    center = (max_bound + min_bound) / 2
    extent = max_bound - min_bound

    # Use the largest dimension for uniform voxel scaling
    max_extent = np.max(extent) * (1 + margin_ratio)
    voxel_size = max_extent / (grid_size - 2)  # Avoid boundary cut-off

    # Compute new bounding box
    new_min_bound = center - max_extent / 2
    new_max_bound = center + max_extent / 2

    # Compute voxel positions
    voxels = mesh.voxelized(pitch=voxel_size)
    voxel_indices = np.floor((voxels.points - new_min_bound) / voxel_size).astype(int)
    voxel_indices = np.clip(voxel_indices, 0, grid_size - 2)  # Prevent out-of-bounds

    # Create voxel grid
    volume = np.zeros((grid_size, grid_size, grid_size))
    for voxel in voxel_indices:
        volume[tuple(voxel)] = 1  # Mark membrane positions

    return volume

def add_random_missing_data(volume, missing_ratio=0.3, seed=None):
    """
    Randomly remove a portion of the voxel data to simulate missing regions.
    
    Args:
        volume (np.ndarray): The original voxel grid.
        missing_ratio (float): Fraction of non-zero voxels to set to zero.
        seed (int or None): Random seed for reproducibility.
        
    Returns:
        np.ndarray: Voxel grid with missing data.
    """
    if seed is not None:
        np.random.seed(seed)
        
    volume_missing = volume.copy()
    membrane_voxels = np.argwhere(volume == 1)
    num_to_remove = int(len(membrane_voxels) * missing_ratio)
    
    indices_to_remove = membrane_voxels[np.random.choice(len(membrane_voxels), num_to_remove, replace=False)]
    for idx in indices_to_remove:
        volume_missing[tuple(idx)] = 0
    
    return volume_missing


def add_random_noise(volume, flip_ratio=0.05, seed=None):
    """
    Randomly flip voxel values (0 to 1, or 1 to 0) to simulate noise.
    
    Args:
        volume (np.ndarray): The original voxel grid.
        flip_ratio (float): Fraction of voxels to flip.
        seed (int or None): Random seed for reproducibility.
        
    Returns:
        np.ndarray: Voxel grid with added noise.
    """
    if seed is not None:
        np.random.seed(seed)
        
    volume_noisy = volume.copy()
    all_voxel_indices = np.argwhere(np.ones_like(volume))  # All indices
    num_to_flip = int(len(all_voxel_indices) * flip_ratio)
    
    indices_to_flip = all_voxel_indices[np.random.choice(len(all_voxel_indices), num_to_flip, replace=False)]
    for idx in indices_to_flip:
        volume_noisy[tuple(idx)] = 1 - volume_noisy[tuple(idx)]  # Flip 0 <-> 1
    
    return volume_noisy


def add_gaussian_noise(volume, mean=0.0, std=0.05, seed=None):
    """
    Add additive Gaussian noise to the volume.
    
    Args:
        volume (np.ndarray): The original voxel grid.
        mean (float): Mean of the Gaussian distribution.
        std (float): Standard deviation of the Gaussian distribution.
        seed (int or None): Random seed for reproducibility.
        
    Returns:
        np.ndarray: Voxel grid with added Gaussian noise.
    """
    if seed is not None:
        np.random.seed(seed)
        
    noise = np.random.normal(loc=mean, scale=std, size=volume.shape)
    volume_noisy = volume + noise
    volume_noisy = np.clip(volume_noisy, 0, 1) # ensure volume_noise is within [0,1]
    
    return volume_noisy


def remove_random_voxel_patches(volume, num_patch=5, rad_patch=3, seed=None):
    """
    Remove small local patches from a connected membrane by region growing.

    Args:
        volume (np.ndarray): 3D binary voxel grid with membrane (1s).
        num_patch (int): Number of patches to remove.
        rad_patch (int): Controls size of each removed patch (dilation radius).
        seed (int or None): For reproducibility.

    Returns:
        np.ndarray: Modified volume with holes.
    """
    if seed is not None:
        np.random.seed(seed)

    volume = volume.copy()
    membrane_voxels = np.argwhere(volume == 1)
    struct = np.ones((3, 3, 3))  # 26-connectivity

    for _ in range(num_patch):
        if len(membrane_voxels) == 0:
            break
        # Pick a random voxel on the membrane
        center = membrane_voxels[np.random.randint(len(membrane_voxels))]
        patch = np.zeros_like(volume)
        patch[tuple(center)] = 1

        # Dilate around the seed point to form a patch
        for _ in range(rad_patch):
            patch = binary_dilation(patch, structure=struct)

        # Remove only where membrane exists
        volume[np.logical_and(patch, volume == 1)] = 0

    return volume



def apply_distance_transform(volume, max_distance=10):
    """
    Applies a distance transform to the voxel grid to simulate density.

    Args:
        volume (np.ndarray): 3D voxel grid with membrane regions marked as 1.
        max_distance (int): Maximum allowed distance for the transform.

    Returns:
        np.ndarray: Transformed intensity grid.
    """
    volume = scipy.ndimage.binary_closing(volume, structure=np.ones((2,2,2)))
    distance_map = scipy.ndimage.distance_transform_edt(1 - volume)
    
    # Limit max distance to avoid over-smoothing
    distance_map = np.clip(distance_map, 0, max_distance)

    return np.exp(-distance_map / 3)  # Simulate electron attenuation


# def generate_pseudo_cryoet(
#     input_file, 
#     output_file=None,
#     grid_size=128, 
#     sigma=1.0, 
#     missing_ratio=None, 
#     flip_ratio=None,
#     num_patch=None,
#     rad_patch=None,
#     frame=-1,
#     margin_ratio=0.4,
#     ):
#     """
#     Full pipeline to generate pseudo cryo-ET data from a membrane mesh.

#     Args:
#         input_file (str or Path): Path to the Mem3DG-generated `.nc` file.
#         output_file (str or Path): Path to save the generated MRC file.
#         grid_size (int): Size of the voxel grid.
#         sigma (float): Standard deviation for Gaussian blur.
    
#     Returns:
#         np.ndarray: Generated pseudo cryo-ET volume.
#     """
#     input_file = Path(input_file)

#     if output_file is not None:
#         output_file = Path(output_file)
#         output_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

#     mesh, coords = load_membrane_mesh(input_file, frame)
#     voxel_grid = generate_voxel_grid(mesh, coords, grid_size, margin_ratio=margin_ratio)
#     if missing_ratio is not None:
#         voxel_grid = add_random_missing_data(voxel_grid, missing_ratio=missing_ratio)
#     if flip_ratio is not None:
#         voxel_grid = add_random_noise(voxel_grid, flip_ratio=flip_ratio)
#     if num_patch is not None and rad_patch is not None:
#         voxel_grid = remove_random_voxel_patches(voxel_grid, num_patch=num_patch, rad_patch=rad_patch, seed=0)

#     pseudo_cryoET = apply_distance_transform(voxel_grid)

#     # Apply Gaussian blur for realistic effect
#     pseudo_cryoET = scipy.ndimage.gaussian_filter(pseudo_cryoET, sigma=sigma)
#     pseudo_cryoET = pseudo_cryoET / pseudo_cryoET.max() # added 2025/07/09
#     pseudo_cryoET = np.transpose(pseudo_cryoET, (2, 1, 0))  # Rotate Z-axis

#     if output_file is not None:
#         # Save as MRC file
#         with mrcfile.new(output_file, overwrite=True) as mrc:
#             mrc.set_data(pseudo_cryoET.astype(np.float32))
#         print(f"Pseudo cryo-ET data saved to: {output_file}")

#     return pseudo_cryoET



def remove_wedge(
    volume: np.ndarray,
    tilt_max_deg: float = 60.0,
    axis = 'z',  # 'x'|'y'|'z' or 0|1|2 or a 3-vector (any length)
) -> np.ndarray:
    """
    Apply a missing-wedge mask in Fourier space to a 3D volume with
    a controllable beam axis.

    Parameters
    ----------
    volume : (Z, Y, X) np.ndarray
        3D intensity grid.
    tilt_max_deg : float
        Max tilt magnitude (±tilt_max_deg) *around the acquisition plane*.
        Frequencies whose angle from that plane exceeds this are zeroed.
        Typical cryo-ET: 60–70.
    axis : {'x','y','z', 0,1,2, or 3-vector}
        Beam direction (the direction along which information is missing).
        - 'z' or 2  → missing near ±z (default cryo-ET convention)
        - 'y' or 1  → missing near ±y
        - 'x' or 0  → missing near ±x
        - any 3-vector (ax, ay, az) to define an arbitrary beam direction

    Returns
    -------
    np.ndarray
        Real-space volume after wedge removal (dtype matches input).
    """
    if volume.ndim != 3:
        raise ValueError("remove_wedge expects a 3D array.")

    # --- Parse/normalize the beam axis u (unit vector in Z,Y,X index order) ---
    if isinstance(axis, str):
        axis = axis.lower()
        if axis not in ('x', 'y', 'z'):
            raise ValueError("axis must be 'x', 'y', or 'z' when given as str.")
        u = np.array([1.0, 0.0, 0.0]) if axis == 'z' else \
            np.array([0.0, 1.0, 0.0]) if axis == 'y' else \
            np.array([0.0, 0.0, 1.0])  # NOTE: mapping to (Z,Y,X) index order
        # Explanation: volume is indexed (Z,Y,X). A 'z' beam means along +Z,
        # which corresponds to unit vector [1,0,0] in (Z,Y,X) coordinates.
    elif isinstance(axis, int):
        if axis not in (0, 1, 2):
            raise ValueError("axis int must be 0 (Z), 1 (Y), or 2 (X).")
        u = np.zeros(3, dtype=float); u[axis] = 1.0
    else:
        u = np.asarray(axis, dtype=float)
        if u.shape != (3,):
            raise ValueError("axis 3-vector must have shape (3,).")
        n = np.linalg.norm(u)
        if n == 0:
            raise ValueError("axis vector must be non-zero.")
        u = u / n

    # --- Build centered frequency index grid in (Z, Y, X) order ---
    F = np.fft.fftn(volume)
    F = np.fft.fftshift(F)

    nz, ny, nx = F.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2

    z = np.arange(nz) - cz
    y = np.arange(ny) - cy
    x = np.arange(nx) - cx
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    # k-vectors at each voxel (in index space; proportional to spatial freq)
    # k = (kz, ky, kx) in (Z, Y, X) order
    # Decompose k into components parallel and perpendicular to beam axis u
    #   k_par = (k·u) u,    k_perp = k - k_par
    k = np.stack((Z, Y, X), axis=0)              # shape (3, nz, ny, nx)
    k_dot_u = k[0]*u[0] + k[1]*u[1] + k[2]*u[2]  # shape (nz, ny, nx)

    # Angle from the acquisition plane (plane ⟂ u):
    # angle_from_plane = arctan( |k_par| / |k_perp| )
    # We KEEP frequencies with angle_from_plane <= tilt_max
    eps = 1e-12
    k_par_norm  = np.abs(k_dot_u)
    k_perp_norm = np.sqrt(np.maximum(k[0]**2 + k[1]**2 + k[2]**2 - k_dot_u**2, 0.0))
    angle_from_plane = np.arctan2(k_par_norm, k_perp_norm + eps)

    theta_max = np.deg2rad(tilt_max_deg)
    keep = angle_from_plane <= theta_max

    F_masked = F * keep

    vol_masked = np.fft.ifftn(np.fft.ifftshift(F_masked))
    return np.asarray(vol_masked.real, dtype=volume.dtype)



def generate_pseudo_cryoet(
    input_files,  # <-- can be str, Path, or list of str/Path
    output_file=None,
    grid_size=128,
    sigma=1.0,
    missing_ratio=None,
    flip_ratio=None,
    num_patch=None,
    rad_patch=None,
    frame=-1,
    margin_ratio=0.4,
    gauss_noise=None,  # std of Gaussian noise. Set value given signal is 1. 
    missing_wedge=True,
    wedge_axis='z',
):
    """
    Full pipeline to generate pseudo cryo-ET data from one or more membrane meshes.

    Args:
        input_files (str, Path, or list): Path(s) to Mem3DG-generated `.nc` file(s).
        output_file (str or Path): Path to save the generated MRC file.
        grid_size (int): Size of the voxel grid.
        sigma (float): Standard deviation for Gaussian blur.

    Returns:
        np.ndarray: Generated pseudo cryo-ET volume.
    """
    # Normalize input to a list
    if isinstance(input_files, (str, Path)):
        input_files = [input_files]
    input_files = [Path(f) for f in input_files]

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load and combine meshes
    meshes, coords_list = [], []
    for f in input_files:
        mesh, coords = load_membrane_mesh(f, frame)
        meshes.append(mesh)
        coords_list.append(coords)

    # Combine into a single mesh + coords
    combined_mesh = trimesh.util.concatenate(meshes)
    combined_coords = np.vstack(coords_list)

    # Voxelization
    voxel_grid = generate_voxel_grid(combined_mesh, combined_coords, grid_size, margin_ratio=margin_ratio)

    # Data corruptions / augmentations
    if missing_ratio is not None:
        voxel_grid = add_random_missing_data(voxel_grid, missing_ratio=missing_ratio)
    if flip_ratio is not None:
        voxel_grid = add_random_noise(voxel_grid, flip_ratio=flip_ratio)
    if num_patch is not None and rad_patch is not None:
        voxel_grid = remove_random_voxel_patches(voxel_grid, num_patch=num_patch, rad_patch=rad_patch, seed=0)
    if gauss_noise is not None:
        voxel_grid = add_gaussian_noise(voxel_grid, std=gauss_noise)

    # Distance transform
    pseudo_cryoET = apply_distance_transform(voxel_grid)
    pseudo_cryoET = np.transpose(pseudo_cryoET, (2, 1, 0))  # Rotate Z-axis

    # Missing wedge effect
    if missing_wedge is True:
        pseudo_cryoET = remove_wedge(pseudo_cryoET, axis=wedge_axis)

    # Gaussian blur for realism
    pseudo_cryoET = scipy.ndimage.gaussian_filter(pseudo_cryoET, sigma=sigma)

    # Normalization
    pseudo_cryoET = pseudo_cryoET / pseudo_cryoET.max()    


    if output_file is not None:
        with mrcfile.new(output_file, overwrite=True) as mrc:
            mrc.set_data(pseudo_cryoET.astype(np.float32))
        print(f"Pseudo cryo-ET data saved to: {output_file}")

    return pseudo_cryoET



def plot_single_slice(pseudo_cryoET, ax, axis='z', slice_index=None, thre=None, show_title=True):
    """
    Plots a 2D slice of the generated pseudo cryo-ET data.

    Args:
        pseudo_cryoET (np.ndarray): 3D simulated cryo-ET data.
        ax (matplotlib.axes.Axes): Axes to plot on.
        axis (str): Direction to slice ('x', 'y', or 'z'). Default is 'z'.
        slice_index (int or None): Index of the slice to display. If None, uses middle slice.
        thre (float or None): Threshold to binarize data. If None, no thresholding.
        show_title (bool): Whether to display the title on the plot.
    """

    if axis not in ('x', 'y', 'z'):
        raise ValueError("axis must be 'x', 'y', or 'z'.")

    grid_size = pseudo_cryoET.shape  # (Nx, Ny, Nz)
    if thre is not None:
        pseudo_cryoET = np.where(pseudo_cryoET > thre, 1, 0)

    if axis == 'x':
        max_index = grid_size[0]
        slice_data = pseudo_cryoET[slice_index if slice_index is not None else max_index // 2, :, :]
        # title = f"Pseudo Cryo-ET Slice (X={slice_index if slice_index is not None else max_index // 2})"
        title = f"X={slice_index if slice_index is not None else max_index // 2}/{max_index}"
    elif axis == 'y':
        max_index = grid_size[1]
        slice_data = pseudo_cryoET[:, slice_index if slice_index is not None else max_index // 2, :]
        # title = f"Pseudo Cryo-ET Slice (Y={slice_index if slice_index is not None else max_index // 2})"
        title = f"Y={slice_index if slice_index is not None else max_index // 2}/{max_index}"
    else:  # 'z'
        max_index = grid_size[2]
        slice_data = pseudo_cryoET[:, :, slice_index if slice_index is not None else max_index // 2]
        # title = f"Pseudo Cryo-ET Slice (Z={slice_index if slice_index is not None else max_index // 2})"
        title = f"Z={slice_index if slice_index is not None else max_index // 2}/{max_index}"


    custom_gray = LinearSegmentedColormap.from_list(
        'custom_gray', ['#f0f0f0', '#111111']  # light gray to dark gray
    )
    # plt.imshow(slice_data, cmap='gray')
    # ax.imshow(slice_data, cmap=custom_gray)
    ax.imshow(np.flipud(slice_data), cmap=custom_gray)
    # plt.colorbar()
    if show_title:
        ax.set_title(title, fontsize=20)
    ax.axis('off')
    # plt.show()



def plot_multiple_slices(volume, axis='z', num_slices=5, thre=None):
    """
    Visualizes multiple slices of a 3D volume along the chosen axis.

    Parameters:
    - volume: 3D numpy array (pseudo cryo-ET data)
    - axis: 'x', 'y', or 'z' (direction of slicing)
    - num_slices: Number of slices to display (default: 5)
    """
    if axis not in ('x', 'y', 'z'):
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    if thre is not None:
        volume = np.where(volume > thre, 1, 0)

    # Get the dimension size for the chosen axis
    dim_size = volume.shape[{'x': 0, 'y': 1, 'z': 2}[axis]]

    # Choose evenly spaced slice indices
    slices = np.linspace(0, dim_size - 1, num_slices, dtype=int)

    # Create figure
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 5))

    for i, idx in enumerate(slices):
        if axis == 'x':
            img = volume[idx, :, :]  # Slice along X-axis (YZ plane)
        elif axis == 'y':
            img = volume[:, idx, :]  # Slice along Y-axis (XZ plane)
        else:  # Default is Z-axis
            img = volume[:, :, idx]  # Slice along Z-axis (XY plane)

        custom_gray = LinearSegmentedColormap.from_list(
            'custom_gray', ['#f0f0f0', '#111111']  # light gray to dark gray
        )

        # axes[i].imshow(img, cmap='gray')
        # axes[i].imshow(img, cmap='gray_r')
        axes[i].imshow(img, cmap=custom_gray, vmin=0, vmax=1)
        axes[i].set_title(f"{axis.upper()}={idx}/{dim_size}")
        axes[i].axis('off')

    plt.show()


