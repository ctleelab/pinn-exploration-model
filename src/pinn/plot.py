import matplotlib.pyplot as plt
from skimage import measure
import jax
import jax.numpy as jnp
import numpy as np
from pinn.model import PINN, laplacian_phi, grad_phi, hessian_phi
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pinn.model import phase_volume, phase_surface
from pinn.grid import phi_on_cryo_grid_xyz, axes_from_cryo_shape
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, Union
import napari


def visualize_zyx_midslice(data, title=None):
    z, y, x = data.shape
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(data[int(z//2),:,:], cmap="gray", alpha=1.0)
    axes[0].set_title(f"mid z ({int(z//2)}/{z})")
    axes[1].imshow(data[:,int(y//2),:], cmap="gray", alpha=1.0)
    axes[1].set_title(f"mid y ({int(y//2)}/{y})")
    axes[2].imshow(data[:,:,int(x//2)], cmap="gray", alpha=1.0)
    axes[2].set_title(f"mid x ({int(x//2)}/{x})")
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def visualize_zyx_midslice_vs_overlay(data_ori, data_pred, title=None):
    z, y, x = data_ori.shape
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(data_ori[int(z//2),:,:], cmap="gray", alpha=1.0)
    axes[0].imshow(data_pred[int(z//2),:,:], cmap="copper", alpha=0.3)
    axes[0].set_title(f"mid z ({int(z//2)}/{z})")
    axes[1].imshow(data_ori[:,int(y//2),:], cmap="gray", alpha=1.0)
    axes[1].imshow(data_pred[:,int(y//2),:], cmap="copper", alpha=0.3)
    axes[1].set_title(f"mid y ({int(y//2)}/{y})")
    axes[2].imshow(data_ori[:,:,int(x//2)], cmap="gray", alpha=1.0)
    axes[2].imshow(data_pred[:,:,int(x//2)], cmap="copper", alpha=0.3)
    axes[2].set_title(f"mid x ({int(x//2)}/{x})")

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def napari_view(data, mask):
    viewer = napari.Viewer()
    viewer.add_image(data, name='tomogram', colormap='gray', blending='additive')
    if mask is not None:
        viewer.add_image(mask, name='mask', colormap='red', blending='additive', opacity=0.5)
    napari.run()

def _normalize_grid_shape(grid_size: Union[int, Tuple[int, int, int]], cryo_shape=None):
    """Return (X, Y, Z) grid sizes.

    grid_size may be:
      - int -> (grid_size, grid_size, grid_size)
      - tuple (nx, ny, nz)
    If cryo_shape is provided (Z,Y,X), prefer using that to derive (X,Y,Z).
    """
    if cryo_shape is not None:
        Z, Y, X = cryo_shape
        return int(X), int(Y), int(Z)
    if isinstance(grid_size, int):
        return grid_size, grid_size, grid_size
    if isinstance(grid_size, tuple) and len(grid_size) == 3:
        return int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
    raise ValueError("grid_size must be int or tuple(len=3)")


def _axis_coords_from_shape(shape: Tuple[int, int, int]):
    """Given cryo shape (Z,Y,X) return x_coords,y_coords,z_coords (1D arrays)."""
    Z, Y, X = shape
    x = np.linspace(-1, 1, X)
    y = np.linspace(-1, 1, Y)
    z = np.linspace(-1, 1, Z)
    return x, y, z

def _to_zyx(arr: np.ndarray) -> np.ndarray:
    """Ensure array is ordered (Z, Y, X).

    Accepts arrays in (X,Y,Z) or (Z,Y,X) and returns (Z,Y,X).
    """
    if arr.ndim != 3:
        raise ValueError("expected 3D array")
    # Common shapes: (X,Y,Z) or (Z,Y,X)
    a, b, c = arr.shape
    if arr.shape[0] != arr.shape[2]:
        return np.transpose(arr, (2, 1, 0))
    # cube case: still transpose to ensure consistent (Z,Y,X)
    return np.transpose(arr, (2, 1, 0))

def visualize_results(phi_fn, step=None, cryoET_data=None):
    """
    Visualizes the learned level-set function by plotting φ=0 contours.
    """
    assert cryoET_data is not None
    # derive axis coords from cryo shape (Z, Y, X)
    Z, Y, X = cryoET_data.shape
    x_coords, y_coords, z_coords = _axis_coords_from_shape(cryoET_data.shape)

    # phi_on_cryo_grid_xyz may return phi in (X,Y,Z) ordering; ensure final array is (Z,Y,X)
    phi_xyz, (Xg, Yg, Zg), _ = phi_on_cryo_grid_xyz(phi_fn, cryoET_data.shape, lo=-1.5, hi=1.5)
    phi_zyx = _to_zyx(np.array(phi_xyz))

    # central slice index in Z
    mid_Z = Z // 2
    fig, ax = plt.subplots(figsize=(6, 6))

    # imshow expects data in (rows=Y, cols=X)
    img = ax.imshow(
        phi_zyx[mid_Z], cmap='bwr', origin='lower',
        extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
        vmin=-0.5, vmax=0.5, alpha=1.0
    )
    plt.colorbar(img, label="φ(x,y,z)")

    # overlay cryoET slice if present (same ordering: cryoET[mid_Z] -> (Y,X))
    cryo_slice = np.array(cryoET_data[mid_Z])  # (Y,X)
    alpha_mask = np.where(cryo_slice > 0.7, 0.5, 0.0)
    ax.imshow(np.ones_like(cryo_slice), cmap='gray', origin='lower',
              extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()], alpha=alpha_mask)

    # φ=0 contour line: build meshgrid matching slice shape (rows=Y, cols=X)
    X2, Y2 = np.meshgrid(x_coords, y_coords, indexing='xy')
    ax.contour(X2, Y2, phi_zyx[mid_Z], levels=[0], colors='black')
    ax.set_xlabel("X-axis"); ax.set_ylabel("Y-axis")
    if step is not None:
        ax.set_title(f"Level-Set at z={float(z_coords[mid_Z]):.2f} (Step {step})")


    # # Set appropriate ticks to match the physical coordinates
    # tick_positions = jnp.linspace(x.min(), x.max(), num=5)  # 5 major ticks
    # tick_labels = [f"{val:.1f}" for val in tick_positions]  # Format labels
    
    # ax.set_xticks(tick_positions)
    # ax.set_xticklabels(tick_labels)
    # ax.set_yticks(tick_positions)
    # ax.set_yticklabels(tick_labels)

    # if step is not None:
    #     ax.set_title(f"Level-Set Function at z=0 (Step {step})")

    # plt.show()



def visualize_checkpoint_level_set(ax, step, checkpoint, data_mask=None, grid_size=None, slice_index=32, axis="z", colorbar=False):
    """
    Compute and visualize the level-set function from a given checkpoint.
    Uses voxel-based coordinates, consistent with `visualize_cryoET_with_contours`.
    """
    state = checkpoint["state"]  # Extract saved model state
    params = state["params"]  # Extract trained parameters

    model = PINN()
    phi_fn = lambda x: model.apply(params, x)

    Z_len = data_mask.shape[0]    
    Y_len = data_mask.shape[1]    
    X_len = data_mask.shape[2]

    phi_xyz, _, (x_axis, y_axis, z_axis) = phi_on_cryo_grid_xyz(phi_fn, data_mask.shape, lo=-1.0, hi=1.0)
    phi_zyx = _to_zyx(np.array(phi_xyz))

    # Extract selected slice and corresponding axis coords
    if axis == "z":
        slice_phi = phi_zyx[int(slice_index)]            # (Y,X)
        col_axis, row_axis = x_axis, y_axis
        xlabel, ylabel = "X-axis (voxels)", "Y-axis (voxels)"
        voxel_cols, voxel_rows = X_len, Y_len
    elif axis == "y":
        slice_phi = phi_zyx[:, int(slice_index), :]      # (Z,X)
        col_axis, row_axis = x_axis, z_axis
        xlabel, ylabel = "X-axis (voxels)", "Z-axis (voxels)"
        voxel_cols, voxel_rows = X_len, Z_len
    elif axis == "x":
        slice_phi = phi_zyx[:, :, int(slice_index)]      # (Z,Y)
        col_axis, row_axis = y_axis, z_axis
        xlabel, ylabel = "Y-axis (voxels)", "Z-axis (voxels)"
        voxel_cols, voxel_rows = Y_len, Z_len

    # Plot
    img = ax.imshow(slice_phi, cmap="bwr", origin="lower",
                    extent=[col_axis.min(), col_axis.max(), row_axis.min(), row_axis.max()],
                    vmin=-1.0, vmax=1.0, alpha=1.0)

    # Contour: build meshgrid matching slice shape
    X2, Y2 = np.meshgrid(col_axis, row_axis, indexing='xy')
    # contour = ax.contour(X2, Y2, slice_phi, levels=[0], colors="black", linewidths=1.5)

    # Set axis labels and ticks
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    tick_positions = np.linspace(-1, 1, num=5)
    voxel_labels_cols = np.linspace(0, voxel_cols - 1, num=5).astype(int)
    voxel_labels_rows = np.linspace(0, voxel_rows - 1, num=5).astype(int)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(voxel_labels_cols)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(voxel_labels_rows)

    if colorbar is True:
        cbar = plt.colorbar(img, ax=ax, shrink=0.6)

    ax.set_title(f"Step {step}, {axis}-slice={slice_index}/{max(X_len, Y_len, Z_len)}")

    return img


def visualize_checkpoint_cryoET_with_contours(
    ax, step, checkpoint, data_mask, slice_index=32, axis="z",
    no_label=False,
):
    state = checkpoint["state"]; params = state["params"]
    model = PINN()
    phi_fn = lambda x: model.apply(params, x.reshape(-1, 3))

    # Compute phi on cryo grid and ensure ordering (Z,Y,X)
    phi_xyz, _, (x_axis, y_axis, z_axis) = phi_on_cryo_grid_xyz(phi_fn, data_mask.shape, lo=-1.0, hi=1.0)
    phi_zyx = _to_zyx(np.array(phi_xyz))

    # axis coords from cryo shape
    # x_coords, y_coords, z_coords = _axis_coords_from_shape(cryoET_data.shape)

    if axis == "z":
        cryo_slice = np.array(data_mask[int(slice_index)])   # (Y,X)
        slice_phi  = phi_zyx[int(slice_index)]                 # (Y,X)
        col_axis, row_axis = x_axis, y_axis
        xlabel, ylabel = "X-axis (voxels)", "Y-axis (voxels)"
        N = data_mask.shape[2]
    elif axis == "y":
        cryo_slice = np.array(data_mask[:, int(slice_index), :])  # (Z,X)
        slice_phi  = phi_zyx[:, int(slice_index), :]                # (Z,X)
        col_axis, row_axis = x_axis, z_axis
        xlabel, ylabel = "X-axis (voxels)", "Z-axis (voxels)"
        N = data_mask.shape[0]
    elif axis == "x":
        cryo_slice = np.array(data_mask[:, :, int(slice_index)])  # (Z,Y)
        slice_phi  = phi_zyx[:, :, int(slice_index)]                # (Z,Y)
        col_axis, row_axis = y_axis, z_axis
        xlabel, ylabel = "Y-axis (voxels)", "Z-axis (voxels)"
        N = data_mask.shape[0]

    # CryoET colormap
    custom_gray = LinearSegmentedColormap.from_list('custom_gray', ['#f0f0f0', '#777777'])

    ax.imshow(cryo_slice, cmap=custom_gray, origin='lower',
              extent=[col_axis.min(), col_axis.max(), row_axis.min(), row_axis.max()], alpha=1.0)
    
    # φ=0 contour: build meshgrid matching slice shape
    X2, Y2 = np.meshgrid(col_axis, row_axis, indexing='xy')
    ax.contour(X2, Y2, slice_phi, levels=[0], colors='red', linewidths=2.0)

    if no_label:
        ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    else:
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f"Step {step}, {axis}-slice={slice_index}/{N}")
        ticks = np.linspace(-1, 1, 5)
        ax.set_xticks(ticks); ax.set_yticks(ticks)



def visualize_physics_loss(
    ax, 
    epsilon, 
    component, 
    grid_size=None, 
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
    binary_mask=None,
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

    # If cryoET_data is provided, prefer to use its shape for the grid so
    # computed φ values align with the binary mask. CryoET_data is (Z,Y,X).
    if cryoET_data is not None:
        X_len, Y_len, Z_len = _normalize_grid_shape(grid_size, cryo_shape=cryoET_data.shape)
        binary_mask = (cryoET_data > threshold).astype(float)  # (Z,Y,X)
        # reorder to (X,Y,Z) then flatten to match meshgrid(indexing='ij') ordering used below
        binary_mask_flat = jnp.ravel(jnp.transpose(binary_mask, (2, 1, 0)))
        weight = 0.8
        w_in = weight / jnp.sum(binary_mask_flat)
        w_out = (1.0 - weight) / jnp.sum(1.0 - binary_mask_flat)
    else:
        X_len, Y_len, Z_len = _normalize_grid_shape(grid_size)

    # Grid dims (X_len, Y_len, Z_len) already set above; build per-axis linspaces
    x = jnp.linspace(-1, 1, X_len)
    y = jnp.linspace(-1, 1, Y_len)
    z = jnp.linspace(-1, 1, Z_len)

    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    grid_points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    # Compute φ values and Laplacian
    phi_vals = phi_fn(grid_points).squeeze()
    L = (2.0, 2.0, 2.0)
    lap_phi  = laplacian_phi(phi_fn, grid_points, L)
    nl_term  = (phi_vals**2 - 1) * phi_vals
    residual = lap_phi - (1 / epsilon**2) * (phi_vals**2 - 1) * phi_vals
    gradient = grad_phi(phi_fn, grid_points)
    hessian  = hessian_phi(phi_fn, grid_points)
    sq_grad  = jnp.sum(gradient**2, axis=1)
    areadist = sq_grad + (0.5 / epsilon**2) * (phi_vals**2 - 1)**2
    if cryoET_data is not None:
        # use the flattened binary mask computed earlier
        data_dot = w_in * binary_mask_flat * (phi_vals**2) + w_out * (1.0 - binary_mask_flat) * ((phi_vals**2 - 1.0)**2)
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

    if component != "phi":
        values = jnp.abs(values)
    values = values.reshape(X_len, Y_len, Z_len)
    values = _to_zyx(np.array(values))  # now (Z,Y,X)

    # Per-axis extraction: determine column and row coords for plotting
    x_coords = np.linspace(-1, 1, X_len)
    y_coords = np.linspace(-1, 1, Y_len)
    z_coords = np.linspace(-1, 1, Z_len)

    if axis == "z":
        slice_data = values[int(slice_index)]            # (Y,X)
        col_coords, row_coords = x_coords, y_coords
        xlabel, ylabel = "X-axis (voxels)", "Y-axis (voxels)"
    elif axis == "y":
        slice_data = values[:, int(slice_index), :]      # (Z,X)
        col_coords, row_coords = x_coords, z_coords
        xlabel, ylabel = "X-axis (voxels)", "Z-axis (voxels)"
    elif axis == "x":
        slice_data = values[:, :, int(slice_index)]      # (Z,Y)
        col_coords, row_coords = y_coords, z_coords
        xlabel, ylabel = "Y-axis (voxels)", "Z-axis (voxels)"


    if vmin is None:
        vmin = float(slice_data.min())
    if vmax is None:
        vmax = float(slice_data.max())

    img = ax.imshow(slice_data, cmap="coolwarm", origin="lower",
                    extent=[col_coords.min(), col_coords.max(), row_coords.min(), row_coords.max()],
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
    voxel_labels_cols = np.linspace(0, len(col_coords) - 1, num=5).astype(int)
    voxel_labels_rows = np.linspace(0, len(row_coords) - 1, num=5).astype(int)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(voxel_labels_cols)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(voxel_labels_rows)


    if title is not None:
        ax.set_title(title)
    if step is not None:
        ax.set_title(f"Step {step}, {axis}-slice={slice_index}/{grid_size}")


    return img



def plot_3d_isosurface(ax, step, checkpoint, shape, no_label=False):
    """
    Load a checkpoint, compute φ values, extract the isosurface, and plot it.

    Args:
        ax: Matplotlib subplot axis to plot on.
        checkpoint_path (str): Path to the checkpoint directory.
        checkpoint_label (str): Label for the plot title.
    """

    # grid_size may be int or tuple; normalize to (X_len, Y_len, Z_len)
    X_len, Y_len, Z_len = shape

    model = PINN()
    params = checkpoint["state"]["params"]
    phi_fn = lambda x: model.apply(params, x.reshape(-1, 3))

    phi_xyz, _, _ = phi_on_cryo_grid_xyz(phi_fn, shape=shape, lo=-1, hi=1)
    # Convert to (Z,Y,X) ordering and make a contiguous numpy float32 array for skimage
    phi_zyx = _to_zyx(np.array(phi_xyz))


    # Determine safe isosurface level: prefer 0 if it's within the data range,
    # otherwise choose midpoint of the data range and warn the user.
    vmin = float(phi_zyx.min())
    vmax = float(phi_zyx.max())
    if vmin == vmax:
        print(f"Warning: φ is constant (value={vmin:.6g}); no isosurface to extract.")
        return None

    if not (vmin <= 0.0 <= vmax):
        # 0 not inside the range; pick midpoint
        level = 0.5 * (vmin + vmax)
        print(f"Warning: φ range = [{vmin:.4g}, {vmax:.4g}] does not include 0; using level={level:.6g} for isosurface.")
    else:
        level = 0.0

    # spacing for marching_cubes -> spacing = (dz, dy, dx) matching phi_zyx order
    sp = (1.0, 1.0, 1.0)

    verts, faces, _, _ = marching_cubes(phi_zyx, level=level, spacing=sp)

    mesh = Poly3DCollection(verts[faces], alpha=0.1, edgecolor="k", linewidth=0.2, facecolor="cyan")
    ax.add_collection3d(mesh)

    # Improve visualization by adding a wireframe effect
    # ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color="gray", alpha=0.15, edgecolor="black", linewidth=0.05)

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
        ax.set_xlim(0, X_len)
        ax.set_ylim(0, Y_len)
        ax.set_zlim(0, Z_len)

    else:
        ax.set_title(f"Step {step}", fontsize=12, y=0.9)
        ax.set_xlim(0, X_len)
        ax.set_ylim(0, Y_len)
        ax.set_zlim(0, Z_len)

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


def plot_phase_metrics_ax(ax, checkpoint, metrics, epsilon=0.05, grid_size=None, V_0=None, A_0=None):

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
    

