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
from matplotlib.lines import Line2D

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




def visualize_checkpoint_result(ax, step, checkpoint, cryoET_data=None, grid_size=64, slice_index=32, axis="z", colorbar=False, show_contour=True):
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
    if show_contour is True:
        contour = ax.contour(contour_x.T, contour_y.T, slice_data, levels=[0], colors="black", linewidths=1.5)

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



def visualize_cryoET_with_contours_heavy(
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





import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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
    expand_xy=None,   # <-- add this
    voxel_scale=(1.0, 1.0, 1.0), 
    hidden_dim=128,
):
    state = checkpoint["state"]
    params = state["params"]
    model = PINN(hidden_dim=hidden_dim)
    voxel_scale = jnp.asarray(voxel_scale)

    @jax.jit
    def phi_batched(x):  # (N,3) -> (N,) or (N,1)
        x_scaled = x * voxel_scale[None, :]   # apply per-axis scaling
        # out = model.apply(params, x)
        out = model.apply(params, x_scaled)
        return out.reshape(-1)

    # ---- your “other function” mechanism, copied faithfully ----
    x = jnp.linspace(-1, 1, grid_size)
    y = jnp.linspace(-1, 1, grid_size)
    z = jnp.linspace(-1, 1, grid_size)
    if expand_xy is not None:
        x = jnp.linspace(-1, 1, grid_size)
        y = jnp.linspace(-expand_xy, expand_xy, grid_size)
        z = jnp.linspace(-expand_xy, expand_xy, grid_size)
    # -----------------------------------------------------------

    # Slice coordinate comes from the axis you are slicing along
    if axis == "z":
        s = z[slice_index]
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        pts = jnp.stack([X.ravel(), Y.ravel(), jnp.full(X.size, s)], axis=-1)
        contour_x, contour_y = X, Y
        cryoET_numpy = np.array(cryoET_data[:, :, slice_index])
        extent = [x.min(), x.max(), y.min(), y.max()]
        xlabel, ylabel = "X-axis (voxels)", "Y-axis (voxels)"

    elif axis == "y":
        s = y[slice_index]
        X, Z = jnp.meshgrid(x, z, indexing="ij")
        pts = jnp.stack([X.ravel(), jnp.full(X.size, s), Z.ravel()], axis=-1)
        contour_x, contour_y = X, Z
        cryoET_numpy = np.array(cryoET_data[:, slice_index, :])
        extent = [x.min(), x.max(), z.min(), z.max()]
        xlabel, ylabel = "X-axis (voxels)", "Z-axis (voxels)"

    elif axis == "x":
        s = x[slice_index]
        Y, Z = jnp.meshgrid(y, z, indexing="ij")
        pts = jnp.stack([jnp.full(Y.size, s), Y.ravel(), Z.ravel()], axis=-1)
        contour_x, contour_y = Y, Z
        cryoET_numpy = np.array(cryoET_data[slice_index, :, :])
        extent = [y.min(), y.max(), z.min(), z.max()]
        xlabel, ylabel = "Y-axis (voxels)", "Z-axis (voxels)"

    else:
        raise ValueError("axis must be one of {'x','y','z'}")

    # Evaluate phi only on slice (grid_size^2 points)
    slice_data = phi_batched(pts).reshape(grid_size, grid_size)

    if thresholding:
        cryoET_numpy = np.where(cryoET_numpy > 0.8, 1.0, 0.0)

    # Plot CryoET grayscale image with matching extent
    custom_gray = LinearSegmentedColormap.from_list("custom_gray", ["#f0f0f0", "#777777"])
    ax.imshow(cryoET_numpy, cmap=custom_gray, origin="lower", extent=extent, alpha=1.0)

    # Overlay φ=0 contour
    ax.contour(contour_x.T, contour_y.T, slice_data, levels=[0.0], colors="red", linewidths=2.0)

    if no_label:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xticklabels([]); ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

        # Note: these tick labels are “voxel index” labels; with expand_xy they no longer
        # correspond to physical coords. If you want physical ticks, we can adjust.
        tick_positions_x = np.linspace(float(extent[0]), float(extent[1]), num=5)
        tick_positions_y = np.linspace(float(extent[2]), float(extent[3]), num=5)
        tick_labels = np.linspace(0, grid_size - 1, num=5).astype(int)

        ax.set_xticks(tick_positions_x); ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions_y); ax.set_yticklabels(tick_labels)

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

    valid_components = ["phi", "phi2", "data", "residual", "laplacian", "nonlinear", "grad_x", "grad_y", "grad_z", "hess_xx", "hess_yy", "hess_zz", "grad_norm2", "tension"]
    if component not in valid_components:
        raise ValueError(f"Invalid component '{component}'. Must be one of {valid_components}.")

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
    elif component == "phi2":
        values = phi_vals**2
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

    if component != "phi":
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

        # if colorbar is True:
        #     cbar = plt.colorbar(img, ax=ax, orientation="horizontal", shrink=0.6, pad=0)
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
    ax.plot(assembled_loss["step"], assembled_loss["phys_loss"], label='Physics Loss', marker='s', linestyle='-')
    ax.plot(assembled_loss["step"], assembled_loss["sign_loss"], label='Sign Loss', marker='^', linestyle='-')
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
    sign_loss_norm = assembled_loss["sign_loss"] / assembled_loss["sign_loss"][0]
    phys_loss_norm = assembled_loss["phys_loss"] / assembled_loss["phys_loss"][0]
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
        ax.plot(assembled_loss["step"], phys_loss_norm, marker='s', linestyle='-')
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

    elif id == 3:    # Plot 
        ax.plot(assembled_loss["step"], sign_loss_norm, marker='^', linestyle='-')
        ax.set_yscale('log')
        ax.set_title("Normalized Sign Loss")
        ax.set_xlabel("Training Steps")
        # ax.set_xlabel("Steps")
        ax.set_ylabel("Loss Value (scaled)")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)


def plot_unnormalized_loss_history_ax(ax, id, assembled_loss):
    """
    Plot three normalized loss functions (Data Loss, Physics Loss, and Total Loss) side by side.
    Each function is scaled by its initial value at step = 0.

    Args:
        assembled_loss (dict): Aggregated loss history containing 'step', 'total_loss', 'data_loss', 'physics_loss'.
    """

    # Normalize loss values by their initial step=0 value
    data_loss = assembled_loss["data_loss"]
    sign_loss = assembled_loss["sign_loss"]
    phys_loss = assembled_loss["phys_loss"]
    curv_loss = assembled_loss["curv_loss"]
    total_loss = assembled_loss["total_loss"]


    if id == 0:
        ax.plot(assembled_loss["step"], total_loss, marker='o', linestyle='-')
        ax.set_yscale('log')
        ax.set_title("Total Loss")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss Value")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    elif id == 1: 
        ax.plot(assembled_loss["step"], data_loss, marker='o', linestyle='-')
        ax.set_yscale('log')
        ax.set_title("Data Loss")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss Value")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    elif id == 2:
        ax.plot(assembled_loss["step"], sign_loss, marker='o', linestyle='-')
        ax.set_yscale('log')
        ax.set_title("Boundary Loss")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss Value")        
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    elif id == 3:
        ax.plot(assembled_loss["step"], phys_loss, marker='o', linestyle='-')
        ax.set_yscale('log')
        ax.set_title("Physics Loss")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss Value")        
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)        

    elif id == 4:
        ax.plot(assembled_loss["step"], curv_loss, marker='o', linestyle='-')
        ax.set_yscale('log')
        ax.set_title("Smoothing Loss")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss Value")
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
    

import numpy as np
import jax.numpy as jnp
from skimage.measure import marching_cubes
import plotly.graph_objects as go


def compute_isosurface_mesh_from_checkpoint_(
    checkpoint,
    grid_size=64,
    level=0.0,
    transpose=True,
):
    # --- build normalized coordinate grid (-1 to 1) ---
    x = jnp.linspace(-1, 1, grid_size)
    y = jnp.linspace(-1, 1, grid_size)
    z = jnp.linspace(-1, 1, grid_size)

    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    grid_points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    # --- restore model params from checkpoint ---
    model = PINN()
    state = checkpoint["state"]
    params = state["params"]

    def phi_fn(xyz):
        return model.apply(params, xyz.reshape(-1, 3))

    # --- evaluate phi on grid ---
    phi_values = phi_fn(grid_points).reshape(grid_size, grid_size, grid_size)
    if transpose:
        phi_values = phi_values.T

    # IMPORTANT: make a *writeable* NumPy array
    phi_np = np.array(phi_values, copy=True)  # guarantees writeable

    # marching cubes spacing (keeping your convention)
    spacing = (2 / grid_size, 2 / grid_size, 2 / grid_size)

    verts, faces, _, _ = marching_cubes(phi_np, level=level, spacing=spacing)

    # shift [0,2] -> [-1,1]
    verts = verts - 1.0
    return verts, faces


import numpy as np
import jax.numpy as jnp
from skimage.measure import marching_cubes


def compute_isosurface_mesh_from_checkpoint(
    checkpoint,
    grid_size=64,
    level=0.0,
    transpose=True,
    x_range=None,   # e.g. (-0.5, 0.5)
    y_range=None,   # e.g. (-1.0, 0.2)
    z_range=None,   # e.g. (-0.5, 0.5)
    hidden_dim=128,
):
    # --- full normalized grids (-1..1) ---
    x = jnp.linspace(-1, 1, grid_size)
    y = jnp.linspace(-1, 1, grid_size)
    z = jnp.linspace(-1, 1, grid_size)

    # correct step for linspace endpoints
    d = 2.0 / (grid_size - 1)

    def _slice_from_range(arr, r):
        """Return (subarray, i0) where i0 is start index in the original array."""
        if r is None:
            return arr, 0
        amin, amax = r
        mask = (arr >= amin) & (arr <= amax)
        idx = jnp.where(mask)[0]
        if idx.size == 0:
            raise ValueError(f"range={r} selects no grid points.")
        i0 = int(idx[0])
        i1 = int(idx[-1]) + 1
        return arr[i0:i1], i0

    x_sub, x0 = _slice_from_range(x, x_range)
    y_sub, y0 = _slice_from_range(y, y_range)
    z_sub, z0 = _slice_from_range(z, z_range)

    # --- evaluate phi only on cropped grid ---
    X, Y, Z = jnp.meshgrid(x_sub, y_sub, z_sub, indexing="ij")  # axes (x,y,z)
    grid_points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    model = PINN(hidden_dim=hidden_dim)
    params = checkpoint["state"]["params"]

    def phi_fn(xyz):
        return model.apply(params, xyz.reshape(-1, 3))

    phi_values = phi_fn(grid_points).reshape(x_sub.shape[0], y_sub.shape[0], z_sub.shape[0])

    # Optional transpose (matches your old convention)
    # (x,y,z) -> (z,y,x)
    if transpose:
        phi_values = phi_values.transpose(2, 1, 0)

    phi_np = np.array(phi_values, copy=True)

    # spacing in array-axis order as seen by marching_cubes
    spacing = (d, d, d)

    verts, faces, _, _ = marching_cubes(phi_np, level=level, spacing=spacing)

    # --- add offsets due to cropping (in [0,2] coordinates before -1 shift) ---
    if transpose:
        # array axes are (z,y,x) so verts columns are (z,y,x)
        verts[:, 0] += z0 * d
        verts[:, 1] += y0 * d
        verts[:, 2] += x0 * d
    else:
        # array axes are (x,y,z) so verts columns are (x,y,z)
        verts[:, 0] += x0 * d
        verts[:, 1] += y0 * d
        verts[:, 2] += z0 * d

    # shift [0,2] -> [-1,1]
    verts = verts - 1.0

    # if transposed, reorder verts back to (x,y,z) for the caller
    if transpose:
        # currently (z,y,x) -> (x,y,z)
        verts = verts[:, [2, 1, 0]]

    return verts, faces




def show_isosurface_plotly(
    checkpoint,
    grid_size=64,
    level=0.0,
    opacity=0.2,
    transpose=True,
    show_axes=True,
    material="membrane",
    x_range=None,
    y_range=None,
    z_range=None,
    aspect="data",   # "cube" | "data" | "manual"
):
    verts, faces = compute_isosurface_mesh_from_checkpoint(
        checkpoint=checkpoint,
        grid_size=grid_size,
        level=level,
        transpose=transpose,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
    )

    i, j, k = faces.T

    materials = {
        "membrane": dict(ambient=0.15, diffuse=0.9, specular=0.3, roughness=0.6, fresnel=0.1),
        "glossy": dict(ambient=0.1, diffuse=0.7, specular=0.8, roughness=0.2, fresnel=0.3),
        "clay": dict(ambient=0.3, diffuse=0.8, specular=0.1, roughness=0.9, fresnel=0.0),
    }

    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=i, j=j, k=k,
            opacity=opacity,
            color="rgba(180, 190, 210, 1.0)",
            flatshading=False,
            lighting=materials[material],
            lightposition=dict(x=1, y=1, z=2),
        )
    ])

    scene = dict(
        xaxis=dict(visible=False, showbackground=False),
        yaxis=dict(visible=False, showbackground=False),
        zaxis=dict(visible=False, showbackground=False),
        bgcolor="white",
    )

    if aspect == "data":
        scene["aspectmode"] = "data"
    elif aspect == "manual":
        scene["aspectmode"] = "manual"
        scene["aspectratio"] = dict(x=1, y=1, z=0.3)  # pick what you like
    else:  # "cube"
        scene["aspectmode"] = "cube"
        # optional: enforce actual z range box
        scene["xaxis"]["range"] = [-1, 1]
        scene["yaxis"]["range"] = [-1, 1]
        scene["zaxis"]["range"] = [-1, 1] if z_range is None else list(z_range)

    fig.update_layout(scene=scene, margin=dict(l=0, r=0, t=0, b=0))
    return fig



def show_isosurface_plotly_(
    checkpoint,
    grid_size=64,
    level=0.0,
    opacity=0.2,
    transpose=True,
    show_axes=True,
    material="membrane",  # new
    z_range=None,
):
    verts, faces = compute_isosurface_mesh_from_checkpoint(
        checkpoint=checkpoint,
        grid_size=grid_size,
        level=level,
        transpose=transpose,
        z_range=z_range,
    )

    i, j, k = faces.T

    # Material presets
    materials = {
        "membrane": dict(
            ambient=0.15,
            diffuse=0.9,
            specular=0.3,
            roughness=0.6,
            fresnel=0.1,
        ),
        "glossy": dict(
            ambient=0.1,
            diffuse=0.7,
            specular=0.8,
            roughness=0.2,
            fresnel=0.3,
        ),
        "clay": dict(
            ambient=0.3,
            diffuse=0.8,
            specular=0.1,
            roughness=0.9,
            fresnel=0.0,
        ),
    }

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=i, j=j, k=k,
                opacity=opacity,
                color="rgba(180, 190, 210, 1.0)",
                flatshading=False,
                lighting=materials[material],
                lightposition=dict(x=1, y=1, z=2),
            )
        ]
    )

    # fig.update_layout(
    #     scene=dict(
    #         aspectmode="cube",
    #         xaxis=dict(visible=show_axes, range=[-1, 1]),
    #         yaxis=dict(visible=show_axes, range=[-1, 1]),
    #         zaxis=dict(visible=show_axes, range=[-1, 1]),
    #     ),
    #     margin=dict(l=0, r=0, t=30, b=0),
    #     title=f"Isosurface: level={level} | grid={grid_size}",
    # )

    fig.update_layout(
        scene=dict(
            aspectmode="cube",
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            bgcolor="white",  # or "rgba(0,0,0,0)"
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    
    return fig




def _make_slice_points(x, y, z, axis, slice_index):
    if axis == "z":
        zz = z[slice_index]
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        pts = jnp.stack([X.ravel(), Y.ravel(), jnp.full(X.size, zz)], axis=-1)
        shape2d = (x.size, y.size)
    elif axis == "y":
        yy = y[slice_index]
        X, Z = jnp.meshgrid(x, z, indexing="ij")
        pts = jnp.stack([X.ravel(), jnp.full(X.size, yy), Z.ravel()], axis=-1)
        shape2d = (x.size, z.size)
    elif axis == "x":
        xx = x[slice_index]
        Y, Z = jnp.meshgrid(y, z, indexing="ij")
        pts = jnp.stack([jnp.full(Y.size, xx), Y.ravel(), Z.ravel()], axis=-1)
        shape2d = (y.size, z.size)
    else:
        raise ValueError(f"axis must be x/y/z, got {axis}")
    return pts, shape2d

def _batch_apply(fn, pts, batch=4096):
    outs = []
    n = pts.shape[0]
    for i in range(0, n, batch):
        outs.append(fn(pts[i:i+batch]))
    return jnp.concatenate(outs, axis=0)


def visualize_phase(
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
    batch=4096,
    run_on_cpu=False,
    expand_xy=None, 
    voxel_scale=(1.0, 1.0, 1.0),
    hidden_dim=128,
):
    voxel_scale = jnp.asarray(voxel_scale)
    valid_components = ["phi", "tension", "bending"]
    if component not in valid_components:
        raise ValueError(f"Invalid component '{component}'. Must be one of {valid_components}.")

    # Build phi_fn only if not provided
    if phi_fn is None:
        if checkpoint is None:
            raise ValueError("Either `phi_fn` or `checkpoint` must be provided.")
        state = checkpoint["state"]
        params = state["params"]
        model = PINN(hidden_dim=hidden_dim)
        # phi_fn = lambda x: model.apply(params, x)
        phi_fn = lambda x: model.apply(params, x * voxel_scale[None, :])
    else:
        params = None  # unknown

    # grid axes
    x = jnp.linspace(-1, 1, grid_size)
    y = jnp.linspace(-1, 1, grid_size)
    z = jnp.linspace(-1, 1, grid_size)
    if expand_xy is not None:
        x = jnp.linspace(-1, 1, grid_size)
        y = jnp.linspace(-expand_xy, expand_xy, grid_size)
        z = jnp.linspace(-expand_xy, expand_xy, grid_size)
    pts, shape2d = _make_slice_points(x, y, z, axis, slice_index)

    # Optionally force CPU placement
    if run_on_cpu:
        cpu = jax.devices("cpu")[0]
        pts = jax.device_put(pts, cpu)
        # if params exist, move them too (prevents device mismatch / copies)
        if checkpoint is not None:
            state = checkpoint["state"]
            state = jax.device_put(state, cpu)
            params = state["params"]
            model = PINN()
            # phi_fn = lambda x: model.apply(params, x)
            phi_fn = lambda x: model.apply(params, x * voxel_scale[None, :])

    phi_batched = lambda P: _batch_apply(lambda Q: phi_fn(Q).squeeze(), P, batch=batch)

    if component == "phi":
        values = phi_batched(pts)

    elif component == "tension":
        def phi_scalar(p):
            return phi_fn(p[None, :]).squeeze()

        grad_single = jax.grad(phi_scalar)
        grad_batched = jax.jit(jax.vmap(grad_single))

        phi_vals = phi_batched(pts)
        grads = _batch_apply(grad_batched, pts, batch=batch)
        sq_grad = jnp.sum(grads**2, axis=1)
        values = epsilon**2 * sq_grad + 0.5 * (phi_vals**2 - 1)**2

    elif component == "bending":
        def phi_scalar(p):
            return phi_fn(p[None, :]).squeeze()

        # d2phi/dx_i dx_i (diagonal Hessian entries)
        def d2_diag_single(p, i):
            g = jax.grad(phi_scalar)
            return jax.grad(lambda pp: g(pp)[i])(p)[i]

        def d2_diag_batched(P, i):
            fn = jax.jit(jax.vmap(lambda p: d2_diag_single(p, i)))
            return _batch_apply(fn, P, batch=batch)

        d2xx = d2_diag_batched(pts, 0)
        d2yy = d2_diag_batched(pts, 1)
        d2zz = d2_diag_batched(pts, 2)
        lap_phi = d2xx + d2yy + d2zz

        phi_vals = phi_batched(pts)
        values = lap_phi - (1 / epsilon**2) * (phi_vals**2 - 1) * phi_vals

    slice_data = values.reshape(shape2d)
    slice_np = np.array(jax.device_get(slice_data))

    if axis == "z":
        x_extent, y_extent = np.array(x), np.array(y)
    elif axis == "y":
        x_extent, y_extent = np.array(x), np.array(z)
    else:
        x_extent, y_extent = np.array(y), np.array(z)

    img = ax.imshow(
        slice_np, cmap="coolwarm", origin="lower",
        # extent=[x_extent.min(), x_extent.max(), y_extent.min(), y_extent.max()],
        extent=[-1, 1, -1, 1],
        vmin=vmin, vmax=vmax, alpha=1.0
    )

    # ----------------------------
    # Label / axis handling
    # ----------------------------
    if no_label:
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        if title is not None:
            ax.set_title(title)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    # ax.set_xlim(-1/expand, 1/expand)
    # ax.set_ylim(-1/expand, 1/expand)

    return img




def plot_loss_panels(
    assembled_loss,
    shape_list,
    lambda_2_list,
    figsize=(16, 3.5),
    logscale_keys=None,
):
    loss_keys   = ["data_loss", "sign_loss", "phys_loss", "total_loss"]
    loss_titles = ["data loss", "sign loss", "physics loss", "total loss"]

    if logscale_keys is None:
        logscale_keys = []

    # linestyle mapping for lambda_2
    ls_map = {lambda_2_list[0]: "-", lambda_2_list[-1]: "--"}

    # color mapping for shapes
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {
        "biconcave": "red",  
        "bud_04":    "blue", 
        "multi":     "green",
    }
    # color_map = {shape: prop_cycle[i % len(prop_cycle)]
    #              for i, shape in enumerate(shape_list)}

    fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=True)

    for ax, key, title in zip(axes, loss_keys, loss_titles):
        for shape in shape_list:
            for lambda_2 in lambda_2_list:
                hist = assembled_loss[shape][lambda_2]

                ax.plot(
                    hist["step"],
                    hist[key],
                    color=color_map[shape],
                    linestyle=ls_map.get(lambda_2, "-"),
                    linewidth=1.8,
                )

        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")

        if key in logscale_keys:
            ax.set_yscale("log")

        ax.grid(True, alpha=0.3)

    # ---- legends ----
    shape_handles = [
        Line2D([0], [0], color=color_map[s], lw=2, label=s)
        for s in shape_list
    ]
    lambda_handles = [
        Line2D([0], [0], color="black", lw=2,
               linestyle=ls_map[l], label=f"λ2 = {l}")
        for l in lambda_2_list
    ]

    # axes[-1].legend(handles=shape_handles, title="shape",
    #                 loc="upper right", frameon=True)
    # leg2 = axes[-1].legend(handles=lambda_handles, title="lambda_2",
    #                        loc="lower right", frameon=True)
    # axes[-1].add_artist(leg2)

    # First legend: color → shape
    leg_shape = axes[-1].legend(
        handles=shape_handles,
        title="shape",
        loc="upper right",
        frameon=True,
    )    

    plt.tight_layout()
    return fig, axes


def show_cryoet_slice(
    ax,
    cryoet,
    slice_index,
    axis="z",
    cmap="gray_r",
    vmin=None,
    vmax=None,
    show_axis=False,
    show_title=True,
    interpolation="nearest",
):
    """
    Show a single raw CryoET slice on an existing axis.

    Title is shown as: slice_index / grid_size
    """
    vol = np.asarray(cryoet)
    if vol.ndim != 3:
        raise ValueError(f"cryoet must be 3D (Z,Y,X), got {vol.shape}")

    Z, Y, X = vol.shape
    axis = axis.lower()

    if axis == "z":
        img = vol[slice_index, :, :]
        grid = Z
    elif axis == "y":
        img = vol[:, slice_index, :]
        grid = Y
    elif axis == "x":
        img = vol[:, :, slice_index]
        grid = X
    else:
        raise ValueError("axis must be one of 'z', 'y', or 'x'")

    ax.imshow(
        img,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )

    if not show_axis:
        ax.set_axis_off()
    if show_title:
        ax.set_title(f"{slice_index} / {grid}")

    return ax


from pathlib import Path
import numpy as np

def write_vtk_point_cloud(filename, points, point_scalars=None):
    """
    Write a legacy ASCII VTK POLYDATA file for a point cloud.
    """
    filename = Path(filename)
    P = np.asarray(points, dtype=np.float64)

    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"points must be shape (N, 3), got {P.shape}")

    N = P.shape[0]
    point_scalars = {} if point_scalars is None else point_scalars

    for key, arr in point_scalars.items():
        arr = np.asarray(arr)
        if arr.shape != (N,):
            raise ValueError(f"Scalar '{key}' must have shape ({N},), got {arr.shape}")

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Combined point cloud\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        f.write(f"POINTS {N} float\n")
        for p in P:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

        f.write(f"\nVERTICES {N} {2*N}\n")
        for i in range(N):
            f.write(f"1 {i}\n")

        if point_scalars:
            f.write(f"\nPOINT_DATA {N}\n")
            for key, arr in point_scalars.items():
                arr = np.asarray(arr, dtype=np.float64)
                f.write(f"SCALARS {key} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for v in arr:
                    if np.isnan(v):
                        f.write("nan\n")
                    else:
                        f.write(f"{v}\n")


def combine_datasets_to_vtk(filename, data_edge, data_sign, data_phys):

    pts_edge = np.asarray(data_edge["points"], dtype=np.float32)
    pts_sign = np.asarray(data_sign["points"], dtype=np.float32)
    pts_phys = np.asarray(data_phys["points"], dtype=np.float32)

    n_edge = len(pts_edge)
    n_sign = len(pts_sign)
    n_phys = len(pts_phys)

    # Combine points in the requested order
    points = np.concatenate([pts_edge, pts_sign, pts_phys], axis=0)

    # Dataset ID
    source_id = np.concatenate([
        np.full(n_edge, 0, dtype=np.float32),  # edge
        np.full(n_sign, 1, dtype=np.float32),  # sign
        np.full(n_phys, 2, dtype=np.float32),  # phys
    ])

    # Labels
    edge_label = np.concatenate([
        np.asarray(data_edge["label"], dtype=np.float32),
        np.full(n_sign, np.nan, dtype=np.float32),
        np.full(n_phys, np.nan, dtype=np.float32),
    ])

    sign_label = np.concatenate([
        np.full(n_edge, np.nan, dtype=np.float32),
        np.asarray(data_sign["label"], dtype=np.float32),
        np.full(n_phys, np.nan, dtype=np.float32),
    ])

    write_vtk_point_cloud(
        filename,
        points,
        point_scalars={
            "source_id": source_id,
            "edge_label": edge_label,
            "sign_label": sign_label,
        },
    )