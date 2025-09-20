import matplotlib.pyplot as plt
from skimage import measure
import jax
import jax.numpy as jnp
import numpy as np
from pinn.model import PINN, laplacian_phi, grad_phi, hessian_phi
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pinn.model import phase_volume, phase_surface
from matplotlib.colors import LinearSegmentedColormap

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
        # alpha_mask = np.where(cryoET_numpy == 1, 0.5, 0.0)
        alpha_mask = np.where(cryoET_numpy > 0.7, 0.5, 0.0)  # Show all intensities > 0.7
        ax.imshow(np.ones_like(cryoET_numpy), cmap='grey', origin='lower',
          extent=[x.min(), x.max(), y.min(), y.max()], alpha=alpha_mask)



    # Plot contour lines for phi = 0
    contour = ax.contour(X[:, :, mid_slice], Y[:, :, mid_slice], 
                         phi_values[:, :, mid_slice], levels=[0], colors='black')

    # ax.clabel(contour, fmt="φ=0", colors='black')  # Label contour line

    
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




def visualize_checkpoint_result(ax, step, checkpoint, cryoET_data=None, grid_size=64, slice_index=32, axis="z", colorbar=False):
    """
    Compute and visualize the level-set function from a given checkpoint.
    Uses voxel-based coordinates, consistent with `visualize_cryoET_with_contours`.
    """
    state = checkpoint["state"]  # Extract saved model state
    params = state["params"]  # Extract trained parameters

    model = PINN()
    phi_fn = lambda x: model.apply(params, x)
    
    x = jnp.linspace(-1, 1, grid_size)
    y = jnp.linspace(-1, 1, grid_size)
    z = jnp.linspace(-1, 1, grid_size)

    # Generate a 3D grid
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    grid_points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    # Compute φ values
    phi_values = phi_fn(grid_points).reshape(grid_size, grid_size, grid_size)

    # Extract the selected slice based on axis
    if axis == "z":
        # cryoET_numpy = np.array(cryoET_data[:, :, slice_index]) if cryoET_data is not None else None
        slice_data = phi_values[:, :, slice_index]
        x_extent, y_extent = x, y
        xlabel, ylabel = "X-axis (voxels)", "Y-axis (voxels)"
        contour_x, contour_y = X[:, :, slice_index], Y[:, :, slice_index]
    elif axis == "y":
        # cryoET_numpy = np.array(cryoET_data[:, slice_index, :]) if cryoET_data is not None else None
        slice_data = phi_values[:, slice_index, :]
        x_extent, y_extent = x, z
        xlabel, ylabel = "X-axis (voxels)", "Z-axis (voxels)"
        contour_x, contour_y = X[:, slice_index, :], Z[:, slice_index, :]
    elif axis == "x":
        # cryoET_numpy = np.array(cryoET_data[slice_index, :, :]) if cryoET_data is not None else None
        slice_data = phi_values[slice_index, :, :]
        x_extent, y_extent = y, z
        xlabel, ylabel = "Y-axis (voxels)", "Z-axis (voxels)"
        contour_x, contour_y = Y[slice_index, :, :], Z[slice_index, :, :]

    # **1. Plot the Level-Set Function φ values using imshow**
    img = ax.imshow(slice_data, cmap="bwr", origin="lower",
                    extent=[x_extent.min(), x_extent.max(), y_extent.min(), y_extent.max()],
                    # vmin=-0.5, vmax=0.5, alpha=1.0)
                    vmin=-1.0, vmax=1.0, alpha=1.0)

    # **2. Overlay the CryoET grayscale image (if available)**
    # if cryoET_numpy is not None:
    #     alpha_mask = np.where(cryoET_numpy > 0.7, 0.5, 0.0)  # Only show intensities > 0.7
    #     ax.imshow(np.ones_like(cryoET_numpy), cmap="gray", origin="lower",
    #               extent=[x_extent.min(), x_extent.max(), y_extent.min(), y_extent.max()], alpha=alpha_mask)

    # **3. Overlay contour lines for φ=0**
    contour = ax.contour(contour_x.T, contour_y.T, slice_data, levels=[0], colors="black", linewidths=1.5)
    # ax.clabel(contour, fmt="φ=0", colors="black")  # Label contour line

    # Set axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Tick labels based on voxel indices (0 to grid_size-1)
    tick_positions = np.linspace(-1, 1, num=5)  # Normalized positions
    voxel_labels = np.linspace(0, grid_size - 1, num=5).astype(int)  # Corresponding voxel indices

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(voxel_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(voxel_labels)

    if colorbar is True:
        cbar = plt.colorbar(img, ax=ax, shrink=0.6)


    ax.set_title(f"Step {step}, {axis}-slice={slice_index}/{grid_size}")

    return img



def visualize_cryoET_with_contours(
    ax, 
    step, 
    checkpoint, 
    cryoET_data, 
    grid_size=64, 
    slice_index=32, 
    axis="z", 
    no_label=False,
    thresholding=False,
    ):
    """
    Overlay extracted φ=0 contours on the original CryoET grayscale image.

    - Uses the actual CryoET voxel grid instead of a fixed [-1.5, 1.5] range.
    - Extracts and overlays level-set contours at φ=0.
    - Allows visualization at a specified slice in x, y, or z directions.
    """
    state = checkpoint["state"]  # Extract saved model state
    params = state["params"]  # Extract trained parameters

    model = PINN()
    phi_fn = lambda x: model.apply(params, x.reshape(-1, 3))

    # Define the coordinate grid (0 to grid_size)
    x = jnp.linspace(-1, 1, grid_size)
    y = jnp.linspace(-1, 1, grid_size)
    z = jnp.linspace(-1, 1, grid_size)

    # Generate a 3D grid
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    grid_points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    # Compute φ values
    phi_values = phi_fn(grid_points).reshape(grid_size, grid_size, grid_size)

    if thresholding is True:
        cryoET_data = jnp.where(cryoET_data > 0.8, 1.0, 0.0)

    # Extract the CryoET grayscale image at the selected slice
    if axis == "z":
        cryoET_numpy = np.array(cryoET_data[:, :, slice_index])  # XY-plane
        slice_data = phi_values[:, :, slice_index]
        x_extent, y_extent = x, y
        xlabel, ylabel = "X-axis (voxels)", "Y-axis (voxels)"
        contour_x, contour_y = X[:, :, slice_index], Y[:, :, slice_index]
    elif axis == "y":
        cryoET_numpy = np.array(cryoET_data[:, slice_index, :])  # XZ-plane
        slice_data = phi_values[:, slice_index, :]
        x_extent, y_extent = x, z
        xlabel, ylabel = "X-axis (voxels)", "Z-axis (voxels)"
        contour_x, contour_y = X[:, slice_index, :], Z[:, slice_index, :]
    elif axis == "x":
        cryoET_numpy = np.array(cryoET_data[slice_index, :, :])  # YZ-plane
        slice_data = phi_values[slice_index, :, :]
        x_extent, y_extent = y, z
        xlabel, ylabel = "Y-axis (voxels)", "Z-axis (voxels)"
        contour_x, contour_y = Y[slice_index, :, :], Z[slice_index, :, :]

    # Plot CryoET grayscale image
    custom_gray = LinearSegmentedColormap.from_list(
        'custom_gray', ['#f0f0f0', '#777777']  # light gray to dark gray
    )
    # ax.imshow(cryoET_numpy, cmap='gray_r', origin='lower',
    ax.imshow(cryoET_numpy, cmap=custom_gray, origin='lower',
              extent=[x_extent.min(), x_extent.max(), y_extent.min(), y_extent.max()], alpha=1.0)

    # binary_mask = np.where(cryoET_numpy > 0.8, 1, 0)
    # ax.imshow(binary_mask, cmap='gray', origin='lower',
    #           extent=[x_extent.min(), x_extent.max(), y_extent.min(), y_extent.max()], alpha=1.0)

    # Extract contour lines for φ=0 and overlay them
    # contour = ax.contour(contour_x.T, contour_y.T, slice_data, levels=[0], colors='red', linewidths=1.5)
    contour = ax.contour(contour_x.T, contour_y.T, slice_data, levels=[0], colors='red', linewidths=2.0)
    # contour = ax.contour(contour_x.T, contour_y.T, slice_data, levels=[0], colors=["#d0e3e9"], linewidths=1.5)
    # ax.clabel(contour, fmt="φ=0", colors='red')  # Label contour line


    if no_label is True:
        ax.set_xticks([])        # Remove x-axis ticks
        ax.set_yticks([])        # Remove y-axis ticks

        ax.set_xticklabels([])   # Remove x-axis tick labels (numbers)
        ax.set_yticklabels([])   # Remove y-axis tick labels (numbers)

        for spine in ax.spines.values():
            spine.set_visible(False)  # Remove border box
    else:
        # Set axis labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Set tick positions and labels (Voxel-based)
        tick_positions = np.linspace(-1, 1, num=5)  # Normalized positions
        tick_labels = np.linspace(0, grid_size - 1, num=5).astype(int)  # Voxel indices


        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

        ax.set_title(f"Step {step}, {axis}-slice={slice_index}/{grid_size}")



def visualize_physics_loss(
    ax, 
    epsilon, 
    component, 
    grid_size=64, 
    slice_index=32, 
    axis="z", 
    vmin=None, 
    vmax=None, 
    colorbar=True,
    checkpoint=None,
    phi_fn=None,
    step=None,
    title=None,
    no_label=False,
    cryoET_data=None,
    threshold=0.8,
    ):
    """
    Visualize the residual of the Allen-Cahn PDE:
    Δφ - (1/ε²)(φ² - 1)φ on a 2D slice from the 3D domain.
    """
    assert (checkpoint is not None) or (phi_fn is not None), "Must provide either a checkpoint or an analytical phi_fn."

    valid_components = ["phi", "data", "residual", "laplacian", "nonlinear", "grad_x", "grad_y", "grad_z", "hess_xx", "hess_yy", "hess_zz", "grad_norm2", "tension"]
    if component not in valid_components:
        raise ValueError(f"Invalid component '{component}'. Must be one of {valid_components}.")

    # Restore phi_fn from checkpoint if needed
    if checkpoint is not None:
        state = checkpoint["state"]
        params = state["params"]
        model = PINN()
        phi_fn = lambda x: model.apply(params, x)

    if cryoET_data is not None:
        binary_mask = jnp.where(cryoET_data > threshold, 1.0, 0.0)
        binary_mask = binary_mask.reshape(-1)
        weight = 0.8
        w_in  = weight / jnp.sum(binary_mask)
        w_out = (1-weight) / jnp.sum(1 - binary_mask)

    x = jnp.linspace(-1, 1, grid_size)
    y = jnp.linspace(-1, 1, grid_size)
    z = jnp.linspace(-1, 1, grid_size)

    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    grid_points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    # Compute φ values and Laplacian
    phi_vals = phi_fn(grid_points).squeeze()
    lap_phi  = laplacian_phi(phi_fn, grid_points)
    nl_term  = (phi_vals**2 - 1) * phi_vals
    residual = lap_phi - (1 / epsilon**2) * (phi_vals**2 - 1) * phi_vals
    gradient = grad_phi(phi_fn, grid_points)
    hessian  = hessian_phi(phi_fn, grid_points)
    sq_grad  = jnp.sum(gradient**2, axis=1)
    areadist = sq_grad + (0.5 / epsilon**2) * (phi_vals**2 - 1)**2
    if cryoET_data is not None:
        data_dot = w_in * binary_mask * phi_vals**2 + w_out * (1-binary_mask) * (phi_vals**2-1)**2
        # print("max: ", data_dot.max())
        # print("min: ", data_dot.min())

    if component == "phi":
        values = phi_vals
    elif component == "data":
        values = data_dot
    elif component == "residual":
        values = residual
    elif component == "laplacian":
        values = lap_phi
    elif component == "nonlinear":
        values = nl_term
    elif component == "grad_x":
        values = gradient[:, 0]
    elif component == "grad_y":
        values = gradient[:, 1]
    elif component == "grad_z":
        values = gradient[:, 2]
    elif component == "hess_xx":
        values = hessian[:, 0, 0]
    elif component == "hess_yy":
        values = hessian[:, 1, 1]
    elif component == "hess_zz":
        values = hessian[:, 2, 2]
    elif component == "grad_norm2":
        values = jnp.sum(gradient ** 2, axis=1)
    elif component == "tension":
        values = areadist

    if component is not "phi":
        values = jnp.abs(values)
    values = values.reshape(grid_size, grid_size, grid_size)

    if axis == "z":
        slice_data = values[:, :, slice_index]
        x_extent, y_extent = x, y
        xlabel, ylabel = "X-axis (voxels)", "Y-axis (voxels)"
    elif axis == "y":
        slice_data = values[:, slice_index, :]
        x_extent, y_extent = x, z
        xlabel, ylabel = "X-axis (voxels)", "Z-axis (voxels)"
    elif axis == "x":
        slice_data = values[slice_index, :, :]
        x_extent, y_extent = y, z
        xlabel, ylabel = "Y-axis (voxels)", "Z-axis (voxels)"


    if vmin is None:
        vmin = float(slice_data.min())
    if vmax is None:
        vmax = float(slice_data.max())

    img = ax.imshow(slice_data, cmap="coolwarm", origin="lower",
                    extent=[x_extent.min(), x_extent.max(), y_extent.min(), y_extent.max()],
                    vmin=vmin, vmax=vmax, alpha=1.0)

    if no_label is True:
        ax.set_xticks([])        # Remove x-axis ticks
        ax.set_yticks([])        # Remove y-axis ticks

        ax.set_xticklabels([])   # Remove x-axis tick labels (numbers)
        ax.set_yticklabels([])   # Remove y-axis tick labels (numbers)

        for spine in ax.spines.values():
            spine.set_visible(False)  # Remove border box
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if colorbar is True:
            cbar = plt.colorbar(img, ax=ax, shrink=0.6)
            # cbar.set_label("Residual value")        

        tick_positions = np.linspace(-1, 1, num=5)
        voxel_labels = np.linspace(0, grid_size - 1, num=5).astype(int)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(voxel_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(voxel_labels)


    if title is not None:
        ax.set_title(title)
    if step is not None:
        ax.set_title(f"Step {step}, {axis}-slice={slice_index}/{grid_size}")


    return img






def plot_3d_isosurface(ax, step, checkpoint, grid_size=64, no_label=False):
    """
    Load a checkpoint, compute φ values, extract the isosurface, and plot it.

    Args:
        ax: Matplotlib subplot axis to plot on.
        checkpoint_path (str): Path to the checkpoint directory.
        checkpoint_label (str): Label for the plot title.
    """

    # Define normalized coordinate grid (-1 to 1)
    x = jnp.linspace(-1, 1, grid_size)
    y = jnp.linspace(-1, 1, grid_size)
    z = jnp.linspace(-1, 1, grid_size)
    
    # Generate a 3D grid using meshgrid
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    grid_points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    
    # Load model and parameters
    model = PINN()
    state = checkpoint["state"]
    params = state["params"]
    phi_fn = lambda x: model.apply(params, x.reshape(-1, 3))

    # Compute φ values over a 3D grid
    phi_values = phi_fn(grid_points).reshape(grid_size, grid_size, grid_size)
    phi_values = phi_values.T

    # Convert φ values from JAX to NumPy for visualization
    phi_values_np = np.array(phi_values)

    # Apply marching cubes to extract the isosurface
    verts, faces, _, _ = marching_cubes(phi_values_np, level=0, spacing=(2/grid_size, 2/grid_size, 2/grid_size))
    verts -= 1.0

    # Add the surface mesh with improved appearance
    mesh = Poly3DCollection(verts[faces], alpha=0.1, edgecolor="k", linewidth=0.2, facecolor="cyan")
    # mesh = Poly3DCollection(verts[faces], alpha=0.1, edgecolor="k", linewidth=0.2, facecolor=["#d0e3e9"])
    ax.add_collection3d(mesh)

    # Improve visualization by adding a wireframe effect
    # ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color="gray", alpha=0.15, edgecolor="black", linewidth=0.05)


    # Set axis labels
    # ax.set_xlabel("X-axis", fontsize=10, labelpad=8)
    # ax.set_ylabel("Y-axis", fontsize=10, labelpad=8)
    # ax.set_zlabel("Z-axis", fontsize=10, labelpad=8)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])


    if no_label is True:
        ax.set_xticks([])        # Remove x-axis ticks
        ax.set_yticks([])        # Remove y-axis ticks
        ax.set_zticks([])        # Remove y-axis ticks

        ax.set_xticklabels([])   # Remove x-axis tick labels (numbers)
        ax.set_yticklabels([])   # Remove y-axis tick labels (numbers)
        ax.set_zticklabels([])   # Remove y-axis tick labels (numbers)

        # Make axis lines and panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        ax.xaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo['grid']['color'] = (1, 1, 1, 0)

        ax.xaxis.line.set_color((1, 1, 1, 0))
        ax.yaxis.line.set_color((1, 1, 1, 0))
        ax.zaxis.line.set_color((1, 1, 1, 0))

        # Adjust camera angle & axis limits
        limit_val = 0.6
        ax.set_xlim(-limit_val, limit_val)
        ax.set_ylim(-limit_val, limit_val)
        # ax.set_zlim(-limit_val, limit_val)
        ax.set_zlim(-0.5, 0.5)

    else:
        ax.set_title(f"Step {step}", fontsize=12, y=0.9)

        # Adjust camera angle & axis limits
        ax.view_init(elev=30, azim=45)  # Adjust view angle
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

    # Improve aesthetics
    ax.grid(False)  # Hide the default grid
    ax.set_facecolor("white")  # Change background color




def plot_loss_history(assembled_loss):
    """
    Plot the loss function over the entire training process.

    Args:
        assembled_loss (dict): Aggregated loss history containing 'step', 'total_loss', 'data_loss', 'physics_loss'.
    """
    plt.figure(figsize=(8, 5))
    
    # Plot the losses
    plt.plot(assembled_loss["step"], assembled_loss["data_loss"], label='Data Loss', marker='o', linestyle='-')
    plt.plot(assembled_loss["step"], assembled_loss["physics_loss"], label='Physics Loss', marker='s', linestyle='-')
    plt.plot(assembled_loss["step"], assembled_loss["total_loss"], label='Total Loss', marker='^', linestyle='-')

    # Use log scale for better visualization (especially if physics loss is large)
    plt.yscale('log')

    # Labels and title
    plt.xlabel('Training Steps', fontsize=12, labelpad=0)
    plt.ylabel('Loss Value (log scale)', fontsize=12, labelpad=0)
    plt.title('Loss Function Over Entire Training Process', fontsize=12, pad=10)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add more margin around the figure
    plt.subplots_adjust(left=0.25, right=0.75, top=0.8, bottom=0.2)


    # Show the plot
    plt.show()


def plot_loss_history_ax(ax, assembled_loss):
    """
    Plot the loss function over the entire training process.

    Args:
        assembled_loss (dict): Aggregated loss history containing 'step', 'total_loss', 'data_loss', 'physics_loss'.
    """
    
    # Plot the losses
    ax.plot(assembled_loss["step"], assembled_loss["data_loss"], label='Data Loss', marker='o', linestyle='-')
    ax.plot(assembled_loss["step"], assembled_loss["physics_loss"], label='Physics Loss', marker='s', linestyle='-')
    ax.plot(assembled_loss["step"], assembled_loss["total_loss"], label='Total Loss', marker='^', linestyle='-')

    # Use log scale for better visualization (especially if physics loss is large)
    ax.set_yscale('log')

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss Function Over Training')

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()



def plot_normalized_loss_history(assembled_loss):
    """
    Plot three normalized loss functions (Data Loss, Physics Loss, and Total Loss) side by side.
    Each function is scaled by its initial value at step = 0.

    Args:
        assembled_loss (dict): Aggregated loss history containing 'step', 'total_loss', 'data_loss', 'physics_loss'.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

    # Normalize loss values by their initial step=0 value
    data_loss_norm = assembled_loss["data_loss"] / assembled_loss["data_loss"][0]
    physics_loss_norm = assembled_loss["physics_loss"] / assembled_loss["physics_loss"][0]
    total_loss_norm = assembled_loss["total_loss"] / assembled_loss["total_loss"][0]

    # Plot Data Loss
    axes[0].plot(assembled_loss["step"], data_loss_norm, marker='o', linestyle='-')
    axes[0].set_yscale('log')
    axes[0].set_title("Normalized Data Loss")
    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Loss (scaled)")
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot Physics Loss
    axes[1].plot(assembled_loss["step"], physics_loss_norm, marker='s', linestyle='-')
    axes[1].set_yscale('log')
    axes[1].set_title("Normalized Physics Loss")
    axes[1].set_xlabel("Training Steps")
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot Total Loss
    axes[2].plot(assembled_loss["step"], total_loss_norm, marker='^', linestyle='-')
    axes[2].set_yscale('log')
    axes[2].set_title("Normalized Total Loss")
    axes[2].set_xlabel("Training Steps")
    axes[2].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adjust layout for clarity
    plt.tight_layout()
    plt.show()



def plot_normalized_loss_history_ax(ax, id, assembled_loss):
    """
    Plot three normalized loss functions (Data Loss, Physics Loss, and Total Loss) side by side.
    Each function is scaled by its initial value at step = 0.

    Args:
        assembled_loss (dict): Aggregated loss history containing 'step', 'total_loss', 'data_loss', 'physics_loss'.
    """

    # Normalize loss values by their initial step=0 value
    data_loss_norm = assembled_loss["data_loss"] / assembled_loss["data_loss"][0]
    physics_loss_norm = assembled_loss["physics_loss"] / assembled_loss["physics_loss"][0]
    total_loss_norm = assembled_loss["total_loss"] / assembled_loss["total_loss"][0]


    if id == 0:     # Plot Data Loss
        ax.plot(assembled_loss["step"], data_loss_norm, marker='o', linestyle='-')
        ax.set_yscale('log')
        ax.set_title("Normalized Data Loss", )
        ax.set_xlabel("Training Steps")
        # ax.set_xlabel("Steps")
        ax.set_ylabel("Loss Value (scaled)")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    elif id == 1:    # Plot Physics Loss
        ax.plot(assembled_loss["step"], physics_loss_norm, marker='s', linestyle='-')
        ax.set_yscale('log')
        ax.set_title("Normalized Physics Loss")
        ax.set_xlabel("Training Steps")
        # ax.set_xlabel("Steps")
        ax.set_ylabel("Loss Value (scaled)")        
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    elif id == 2:    # Plot Total Loss
        ax.plot(assembled_loss["step"], total_loss_norm, marker='^', linestyle='-')
        ax.set_yscale('log')
        ax.set_title("Normalized Total Loss")
        ax.set_xlabel("Training Steps")
        # ax.set_xlabel("Steps")
        ax.set_ylabel("Loss Value (scaled)")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)


def plot_phase_metrics_ax(ax, checkpoint, metrics, epsilon=0.05, grid_size=64, V_0=None, A_0=None):

    x = jnp.linspace(-1, 1, grid_size)
    y = jnp.linspace(-1, 1, grid_size)
    z = jnp.linspace(-1, 1, grid_size)

    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    # grid_points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    key = jax.random.PRNGKey(0)
    num_collocation = 1000
    grid_points = (jax.random.uniform(key, (num_collocation, 3), minval=-1, maxval=1)) # Sampled from [-1, 1]^3

    assembled_data = {"step": [], "value": []}

    for step in checkpoint:
        checkpoint_data = checkpoint[step]

        state  = checkpoint_data["state"]
        params = state["params"]

        model = PINN()
        phi_fn = lambda x: model.apply(params, x)

        if metrics == "volume":
            value = phase_volume(phi_fn, grid_points)
        elif metrics == "area":
            value = phase_surface(phi_fn, grid_points, epsilon)
        else:
            raise ValueError(f"Unknown metric type: {metrics}")

        assembled_data["step"].append(step)
        assembled_data["value"].append(value)

    ax.plot(assembled_data["step"], assembled_data["value"], label=metrics)

    if V_0 is not None:
        ax.axhline(y=V_0, linestyle="--", color="gray")
    elif A_0 is not None:
        ax.axhline(y=A_0, linestyle="--", color="gray")


    ax.set_xlabel("Step")
    ax.set_ylabel(metrics.capitalize())
    # ax.set_yscale('log')
    # ax.set_ylim(1e-1, 1e1)

    # ax.legend()
    

