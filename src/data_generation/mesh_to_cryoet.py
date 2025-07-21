import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from pathlib import Path
import mrcfile
import trimesh
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import label, generate_binary_structure

def load_membrane_mesh(input_file):
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
    coordinates = data.groups['Trajectory'].variables['coordinates'][-1]
    coords = coordinates.reshape(-1, 3)  # Convert to (N,3)

    # Extract topology (assuming it defines triangular faces)
    topology = data.groups['Trajectory'].variables['topology'][-1]  
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

import numpy as np
from scipy.ndimage import binary_dilation

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


def generate_pseudo_cryoet(
    input_file, 
    output_file=None,
    grid_size=128, 
    sigma=1.0, 
    missing_ratio=None, 
    flip_ratio=None,
    num_patch=None,
    rad_patch=None,
    ):
    """
    Full pipeline to generate pseudo cryo-ET data from a membrane mesh.

    Args:
        input_file (str or Path): Path to the Mem3DG-generated `.nc` file.
        output_file (str or Path): Path to save the generated MRC file.
        grid_size (int): Size of the voxel grid.
        sigma (float): Standard deviation for Gaussian blur.
    
    Returns:
        np.ndarray: Generated pseudo cryo-ET volume.
    """
    input_file = Path(input_file)

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    mesh, coords = load_membrane_mesh(input_file)
    voxel_grid = generate_voxel_grid(mesh, coords, grid_size)
    if missing_ratio is not None:
        voxel_grid = add_random_missing_data(voxel_grid, missing_ratio=missing_ratio)
    if flip_ratio is not None:
        voxel_grid = add_random_noise(voxel_grid, flip_ratio=flip_ratio)
    if num_patch is not None and rad_patch is not None:
        voxel_grid = remove_random_voxel_patches(voxel_grid, num_patch=num_patch, rad_patch=rad_patch, seed=0)

    pseudo_cryoET = apply_distance_transform(voxel_grid)

    # Apply Gaussian blur for realistic effect
    pseudo_cryoET = scipy.ndimage.gaussian_filter(pseudo_cryoET, sigma=sigma)
    pseudo_cryoET = pseudo_cryoET / pseudo_cryoET.max() # added 2025/07/09
    pseudo_cryoET = np.transpose(pseudo_cryoET, (2, 1, 0))  # Rotate Z-axis

    if output_file is not None:
        # Save as MRC file
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
    ax.imshow(slice_data, cmap=custom_gray)
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


