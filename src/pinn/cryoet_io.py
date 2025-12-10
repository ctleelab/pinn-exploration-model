import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import mrcfile

# Define a synthetic cryo-ET dataset where the membrane edge has higher intensity
def generate_synthetic_cryoET(grid_size=64, radius=1.0, edge_thickness=0.05, noise_level=0.1):
    """Generate a synthetic cryo-ET image where the membrane boundary is highlighted."""
    
    # Create a 3D grid
    x = jnp.linspace(-1.5, 1.5, grid_size)
    y = jnp.linspace(-1.5, 1.5, grid_size)
    z = jnp.linspace(-1.5, 1.5, grid_size)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    # Compute the distance from the sphere surface
    # distance = jnp.abs(jnp.sqrt(X**2 + Y**2 + Z**2) - radius)
    distance = jnp.maximum(jnp.maximum(jnp.abs(X) - radius, jnp.abs(Y) - radius), jnp.abs(Z) - radius)


    # Create an intensity map where the membrane edge has the highest intensity
    cryoET_data = jnp.exp(-distance**2 / (2 * edge_thickness**2))  # Gaussian-like edge contrast

    # Normalize intensity to [0, 1]
    cryoET_data = cryoET_data / jnp.max(cryoET_data)
    cryoET_data = jnp.where(cryoET_data > 0.8, 1.0, 0.0)

    # Apply Gaussian blur to simulate imaging resolution
    # cryoET_data = gaussian_filter(cryoET_data, sigma=1.5)

    # Add Gaussian noise to simulate cryo-ET artifacts
    # noise = np.random.normal(0, noise_level, cryoET_data.shape)
    # cryoET_data = cryoET_data + noise
    # cryoET_data = jnp.clip(cryoET_data, 0, 1)  # Ensure values stay within [0,1]

    return cryoET_data


def plot_synthetic_cryoET(cryoET_data, grid_size=64):
    """
    Plots a central slice of the synthetic cryo-ET data, highlighting the membrane boundary.
    
    Parameters:
        cryoET_data (jnp.ndarray): The synthetic 3D cryo-ET data.
        grid_size (int): The resolution of the grid.
    """
    # Define the real coordinate values
    x = jnp.linspace(-1.5, 1.5, grid_size)
    y = jnp.linspace(-1.5, 1.5, grid_size)
    z = jnp.linspace(-1.5, 1.5, grid_size)

    # Extract a central slice (xy-plane at z=0)
    mid_slice = grid_size // 2
    fig, ax = plt.subplots(figsize=(6, 6))
    
    img = ax.imshow(cryoET_data[:, :, mid_slice].T, cmap='gray', origin='lower',
                    extent=[x.min(), x.max(), y.min(), y.max()])  # Ensure correct scaling

    plt.colorbar(img, label="Intensity (Membrane Boundary)")
    ax.set_title("Synthetic Cryo-ET Data (Z=0 Slice)")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # Set appropriate ticks to match physical coordinates
    tick_positions = jnp.linspace(x.min(), x.max(), num=5)  # 5 major ticks
    tick_labels = [f"{val:.1f}" for val in tick_positions]  # Format labels
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    plt.show()


def load_mrc_data(file_path, grid_size=64):
    """
    Loads a 3D cryo-ET membrane image from an MRC file and normalizes it.
    
    Args:
        file_path (str): Path to the MRC file.
        grid_size (int or tuple of 3 ints): Output size (uniform or (Z, Y, X)).
        
    Returns:
        jnp.ndarray: Normalized 3D cryo-ET data.
    """
    with mrcfile.open(file_path, permissive=True) as mrc:
        mrc_data = mrc.data.astype(jnp.float32)  # Convert to JAX-compatible float32

    # Normalize intensity to [0,1]
    mrc_data = (mrc_data - jnp.min(mrc_data)) / (jnp.max(mrc_data) - jnp.min(mrc_data))

    # Determine target shape
    if isinstance(grid_size, int):
        target_shape = (grid_size, grid_size, grid_size)
    elif isinstance(grid_size, tuple) and len(grid_size) == 3:
        target_shape = grid_size
    else:
        raise ValueError("grid_size must be either an int or a tuple of 3 integers.")

    # Resize if needed
    if mrc_data.shape != target_shape:
        # print(f"Resizing MRC data from {mrc_data.shape} to {target_shape}")
        from skimage.transform import resize
        mrc_data = resize(mrc_data, target_shape, mode='reflect', anti_aliasing=True)
        mrc_data = jnp.array(mrc_data)

    return mrc_data


