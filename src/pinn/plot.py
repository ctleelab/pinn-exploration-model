import matplotlib.pyplot as plt
from skimage import measure
import jax
import jax.numpy as jnp
import numpy as np

def visualize_results(phi_fn, grid_size=64, step=None, cryoET_data=None):
    """
    Visualizes the learned level-set function by plotting φ=0 contours.
    """
    x = jnp.linspace(-1.5, 1.5, grid_size)  # Define the real coordinate values
    y = jnp.linspace(-1.5, 1.5, grid_size)
    z = jnp.linspace(-1.5, 1.5, grid_size)
    
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    grid_points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    phi_values = phi_fn(grid_points).reshape(grid_size, grid_size, grid_size)

    # Extract a central slice (xy-plane at z=0)
    mid_slice = grid_size // 2
    fig, ax = plt.subplots(figsize=(6, 6))

    img = ax.imshow(phi_values[:, :, mid_slice].T, cmap='bwr', origin='lower',
            extent=[x.min(), x.max(), y.min(), y.max()], 
            vmin=-0.5, vmax=0.5, alpha=1.0)  # Ensure correct scaling
    plt.colorbar(img, label="φ(x,y,z)")


    # Plot the cryo-ET data as a background heatmap
    if cryoET_data is not None:
        cryoET_numpy = np.array(cryoET_data[:, :, mid_slice])
        alpha_mask = np.where(cryoET_numpy == 1, 0.5, 0.0)
        ax.imshow(np.ones_like(cryoET_numpy), cmap='grey', origin='lower',
          extent=[x.min(), x.max(), y.min(), y.max()], alpha=alpha_mask)



    # Plot contour lines for phi = 0
    contour = ax.contour(X[:, :, mid_slice], Y[:, :, mid_slice], 
                         phi_values[:, :, mid_slice], levels=[0], colors='black')

    ax.clabel(contour, fmt="φ=0", colors='black')  # Label contour line

    
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # Set appropriate ticks to match the physical coordinates
    tick_positions = jnp.linspace(x.min(), x.max(), num=5)  # 5 major ticks
    tick_labels = [f"{val:.1f}" for val in tick_positions]  # Format labels
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    if step is not None:
        ax.set_title(f"Level-Set Function at z=0 (Step {step})")

    # plt.show()
