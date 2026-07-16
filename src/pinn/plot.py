from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from pinn.model import (
    PINN,
    grad_phi,
    hessian_phi,
    laplacian_phi,
    phase_surface,
    phase_volume,
)


LOSS_PLOT_SETTINGS = {
    "data_loss": ("Data loss", "o"),
    "phys_loss": ("Physics loss", "s"),
    "sign_loss": ("Sign loss", "^"),
    "curv_loss": ("Curvature loss", "D"),
}

GRAY_COLORMAP = LinearSegmentedColormap.from_list(
    "custom_gray",
    ["#f0f0f0", "#777777"],
)


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------


def _validate_axis(axis: str) -> str:
    """Validate and normalize a spatial-axis name."""
    axis = axis.lower()

    if axis not in {"x", "y", "z"}:
        raise ValueError(
            f"axis must be 'x', 'y', or 'z', but received {axis!r}."
        )

    return axis


def _make_coordinate_axes(
    grid_size: int,
    expand_xy: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Create the coordinate axes used for model evaluation."""
    x = jnp.linspace(-1.0, 1.0, grid_size)

    if expand_xy is None:
        y = jnp.linspace(-1.0, 1.0, grid_size)
        z = jnp.linspace(-1.0, 1.0, grid_size)
    else:
        y = jnp.linspace(-expand_xy, expand_xy, grid_size)
        z = jnp.linspace(-expand_xy, expand_xy, grid_size)

    return x, y, z


def _make_slice_points(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    axis: str,
    slice_index: int,
) -> tuple[jax.Array, tuple[int, int], tuple[jax.Array, jax.Array]]:
    """Create model-input points for one two-dimensional slice."""
    axis = _validate_axis(axis)

    if axis == "z":
        slice_coordinate = z[slice_index]
        first, second = jnp.meshgrid(x, y, indexing="ij")

        points = jnp.stack(
            [
                first.ravel(),
                second.ravel(),
                jnp.full(first.size, slice_coordinate),
            ],
            axis=-1,
        )
        shape = (x.size, y.size)

    elif axis == "y":
        slice_coordinate = y[slice_index]
        first, second = jnp.meshgrid(x, z, indexing="ij")

        points = jnp.stack(
            [
                first.ravel(),
                jnp.full(first.size, slice_coordinate),
                second.ravel(),
            ],
            axis=-1,
        )
        shape = (x.size, z.size)

    else:
        slice_coordinate = x[slice_index]
        first, second = jnp.meshgrid(y, z, indexing="ij")

        points = jnp.stack(
            [
                jnp.full(first.size, slice_coordinate),
                first.ravel(),
                second.ravel(),
            ],
            axis=-1,
        )
        shape = (y.size, z.size)

    return points, shape, (first, second)


def _slice_extent(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    axis: str,
) -> list[float]:
    """Return the plotting extent for a selected slice orientation."""
    axis = _validate_axis(axis)

    if axis == "z":
        horizontal, vertical = x, y
    elif axis == "y":
        horizontal, vertical = x, z
    else:
        horizontal, vertical = y, z

    return [
        float(horizontal.min()),
        float(horizontal.max()),
        float(vertical.min()),
        float(vertical.max()),
    ]


def _slice_axis_labels(axis: str) -> tuple[str, str]:
    """Return labels for the two displayed coordinates."""
    axis = _validate_axis(axis)

    if axis == "z":
        return "X-axis (voxels)", "Y-axis (voxels)"
    if axis == "y":
        return "X-axis (voxels)", "Z-axis (voxels)"

    return "Y-axis (voxels)", "Z-axis (voxels)"


def _batch_apply(
    fn,
    points: jax.Array,
    batch_size: int = 4096,
) -> jax.Array:
    """Apply a function to points in smaller batches."""
    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, but received {batch_size}."
        )

    outputs = [
        fn(points[start : start + batch_size])
        for start in range(0, points.shape[0], batch_size)
    ]

    return jnp.concatenate(outputs, axis=0)


def _hide_2d_axis(ax: Any) -> None:
    """Remove labels, ticks, and borders from a Matplotlib axis."""
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False)

    for spine in ax.spines.values():
        spine.set_visible(False)


def _set_voxel_ticks(
    ax: Any,
    extent: list[float],
    grid_size: int,
) -> None:
    """Set normalized coordinate positions with voxel-index labels."""
    x_positions = np.linspace(extent[0], extent[1], num=5)
    y_positions = np.linspace(extent[2], extent[3], num=5)
    labels = np.linspace(0, grid_size - 1, num=5).astype(int)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)


def _build_phi_fn(
    checkpoint: Mapping[str, Any],
    hidden_dim: int = 128,
    voxel_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """Construct a model-evaluation function from a checkpoint."""
    params = checkpoint["state"]["params"]
    model = PINN(hidden_dim=hidden_dim)
    scale = jnp.asarray(voxel_scale)

    def phi_fn(points: jax.Array) -> jax.Array:
        scaled_points = points * scale[None, :]
        return model.apply(params, scaled_points)

    return phi_fn


# ---------------------------------------------------------------------
# CryoET contour visualization
# ---------------------------------------------------------------------


def visualize_cryoET_with_contours(
    ax: Any,
    step: int,
    checkpoint: Mapping[str, Any],
    cryoET_data: Any,
    grid_size: int = 64,
    slice_index: int = 32,
    axis: str = "z",
    no_label: bool = False,
    thresholding: bool = False,
    expand_xy: float | None = None,
    voxel_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    hidden_dim: int = 128,
) -> None:
    """Overlay the learned zero level set on a CryoET slice."""
    axis = _validate_axis(axis)
    x, y, z = _make_coordinate_axes(grid_size, expand_xy)

    base_phi_fn = _build_phi_fn(
        checkpoint,
        hidden_dim=hidden_dim,
        voxel_scale=voxel_scale,
    )
    phi_fn = jax.jit(lambda points: base_phi_fn(points).reshape(-1))

    points, shape, contour_coordinates = _make_slice_points(
        x,
        y,
        z,
        axis,
        slice_index,
    )
    contour_x, contour_y = contour_coordinates

    phase_slice = phi_fn(points).reshape(shape)

    cryoet = np.asarray(cryoET_data)

    if axis == "z":
        cryoet_slice = cryoet[:, :, slice_index]
    elif axis == "y":
        cryoet_slice = cryoet[:, slice_index, :]
    else:
        cryoet_slice = cryoet[slice_index, :, :]

    if thresholding:
        cryoet_slice = np.where(
            cryoet_slice > 0.8,
            1.0,
            0.0,
        )

    extent = _slice_extent(x, y, z, axis)
    xlabel, ylabel = _slice_axis_labels(axis)

    ax.imshow(
        cryoet_slice,
        cmap=GRAY_COLORMAP,
        origin="lower",
        extent=extent,
        alpha=1.0,
    )

    ax.contour(
        contour_x.T,
        contour_y.T,
        phase_slice,
        levels=[0.0],
        colors="red",
        linewidths=2.0,
    )

    if no_label:
        _hide_2d_axis(ax)
        return

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _set_voxel_ticks(ax, extent, grid_size)
    ax.set_title(
        f"Step {step}, {axis}-slice={slice_index}/{grid_size}"
    )


# ---------------------------------------------------------------------
# Physics-component visualization
# ---------------------------------------------------------------------


def visualize_physics_loss(
    ax: Any,
    epsilon: float,
    component: str,
    grid_size: int = 64,
    slice_index: int = 32,
    axis: str = "z",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    checkpoint: Mapping[str, Any] | None = None,
    phi_fn=None,
    step: int | None = None,
    title: str | None = None,
    no_label: bool = False,
    cryoET_data: Any | None = None,
    threshold: float = 0.8,
):
    """Visualize a phase-field or derivative-based diagnostic component."""
    axis = _validate_axis(axis)

    valid_components = {
        "phi",
        "phi2",
        "data",
        "residual",
        "laplacian",
        "nonlinear",
        "grad_x",
        "grad_y",
        "grad_z",
        "hess_xx",
        "hess_yy",
        "hess_zz",
        "grad_norm2",
        "tension",
    }

    if component not in valid_components:
        raise ValueError(
            f"Unknown component {component!r}. "
            f"Choose from {sorted(valid_components)}."
        )

    if phi_fn is None:
        if checkpoint is None:
            raise ValueError(
                "Either checkpoint or phi_fn must be provided."
            )

        phi_fn = _build_phi_fn(checkpoint)

    if component == "data" and cryoET_data is None:
        raise ValueError(
            "cryoET_data is required when component='data'."
        )

    x, y, z = _make_coordinate_axes(grid_size)
    first, second, third = jnp.meshgrid(
        x,
        y,
        z,
        indexing="ij",
    )
    grid_points = jnp.stack(
        [
            first.ravel(),
            second.ravel(),
            third.ravel(),
        ],
        axis=-1,
    )

    phi_values = phi_fn(grid_points).reshape(-1)
    laplacian = laplacian_phi(phi_fn, grid_points)
    nonlinear = (phi_values**2 - 1.0) * phi_values
    residual = laplacian - nonlinear / epsilon**2

    gradient = grad_phi(phi_fn, grid_points)
    hessian = hessian_phi(phi_fn, grid_points)

    gradient_norm_squared = jnp.sum(gradient**2, axis=1)
    tension_density = (
        gradient_norm_squared
        + 0.5 * (phi_values**2 - 1.0) ** 2 / epsilon**2
    )

    data_density = None
    if cryoET_data is not None:
        binary_mask = (
            jnp.asarray(cryoET_data) > threshold
        ).astype(jnp.float32)
        binary_mask = binary_mask.reshape(-1)

        n_inside = jnp.sum(binary_mask)
        n_outside = jnp.sum(1.0 - binary_mask)

        if float(n_inside) == 0.0 or float(n_outside) == 0.0:
            raise ValueError(
                "The thresholded CryoET mask must contain both foreground "
                "and background voxels."
            )

        foreground_weight = 0.8 / n_inside
        background_weight = 0.2 / n_outside

        data_density = (
            foreground_weight * binary_mask * phi_values**2
            + background_weight
            * (1.0 - binary_mask)
            * (phi_values**2 - 1.0) ** 2
        )

    component_values = {
        "phi": phi_values,
        "phi2": phi_values**2,
        "data": data_density,
        "residual": residual,
        "laplacian": laplacian,
        "nonlinear": nonlinear,
        "grad_x": gradient[:, 0],
        "grad_y": gradient[:, 1],
        "grad_z": gradient[:, 2],
        "hess_xx": hessian[:, 0, 0],
        "hess_yy": hessian[:, 1, 1],
        "hess_zz": hessian[:, 2, 2],
        "grad_norm2": gradient_norm_squared,
        "tension": tension_density,
    }

    values = component_values[component]

    if component != "phi":
        values = jnp.abs(values)

    values = values.reshape(
        grid_size,
        grid_size,
        grid_size,
    )

    if axis == "z":
        slice_data = values[:, :, slice_index]
    elif axis == "y":
        slice_data = values[:, slice_index, :]
    else:
        slice_data = values[slice_index, :, :]

    extent = _slice_extent(x, y, z, axis)
    xlabel, ylabel = _slice_axis_labels(axis)

    if vmin is None:
        vmin = float(slice_data.min())

    if vmax is None:
        vmax = float(slice_data.max())

    image = ax.imshow(
        slice_data,
        cmap="coolwarm",
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        alpha=1.0,
    )

    if no_label:
        _hide_2d_axis(ax)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _set_voxel_ticks(ax, extent, grid_size)

        if colorbar:
            plt.colorbar(image, ax=ax, shrink=0.6)

        if title is not None:
            ax.set_title(title)
        elif step is not None:
            ax.set_title(
                f"Step {step}, {axis}-slice={slice_index}/{grid_size}"
            )

    return image


# ---------------------------------------------------------------------
# Efficient slice visualization
# ---------------------------------------------------------------------


def visualize_phase(
    ax: Any,
    epsilon: float,
    component: str,
    grid_size: int = 64,
    slice_index: int = 32,
    axis: str = "z",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = False,
    checkpoint: Mapping[str, Any] | None = None,
    phi_fn=None,
    step: int | None = None,
    title: str | None = None,
    no_label: bool = False,
    batch_size: int = 4096,
    run_on_cpu: bool = False,
    expand_xy: float | None = None,
    voxel_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    hidden_dim: int = 128,
):
    """Visualize phase, tension density, or bending residual on one slice."""
    axis = _validate_axis(axis)

    valid_components = {"phi", "tension", "bending"}
    if component not in valid_components:
        raise ValueError(
            f"Unknown component {component!r}. "
            f"Choose from {sorted(valid_components)}."
        )

    if phi_fn is None:
        if checkpoint is None:
            raise ValueError(
                "Either checkpoint or phi_fn must be provided."
            )

        phi_fn = _build_phi_fn(
            checkpoint,
            hidden_dim=hidden_dim,
            voxel_scale=voxel_scale,
        )

    x, y, z = _make_coordinate_axes(grid_size, expand_xy)
    points, shape, _ = _make_slice_points(
        x,
        y,
        z,
        axis,
        slice_index,
    )

    if run_on_cpu:
        cpu = jax.devices("cpu")[0]
        points = jax.device_put(points, cpu)

        if checkpoint is not None:
            cpu_checkpoint = jax.device_put(checkpoint, cpu)
            phi_fn = _build_phi_fn(
                cpu_checkpoint,
                hidden_dim=hidden_dim,
                voxel_scale=voxel_scale,
            )

    def evaluate_phi(batch_points):
        return phi_fn(batch_points).reshape(-1)

    def evaluate_all_phi(input_points):
        return _batch_apply(
            evaluate_phi,
            input_points,
            batch_size=batch_size,
        )

    if component == "phi":
        values = evaluate_all_phi(points)

    elif component == "tension":
        def scalar_phi(point):
            return phi_fn(point[None, :]).reshape(())

        gradient_fn = jax.jit(
            jax.vmap(jax.grad(scalar_phi))
        )

        phi_values = evaluate_all_phi(points)
        gradients = _batch_apply(
            gradient_fn,
            points,
            batch_size=batch_size,
        )
        gradient_norm_squared = jnp.sum(
            gradients**2,
            axis=1,
        )

        values = (
            epsilon**2 * gradient_norm_squared
            + 0.5 * (phi_values**2 - 1.0) ** 2
        )

    else:
        def scalar_phi(point):
            return phi_fn(point[None, :]).reshape(())

        hessian_fn = jax.jit(
            jax.vmap(jax.hessian(scalar_phi))
        )

        hessians = _batch_apply(
            hessian_fn,
            points,
            batch_size=batch_size,
        )
        laplacian = jnp.trace(
            hessians,
            axis1=1,
            axis2=2,
        )

        phi_values = evaluate_all_phi(points)
        values = (
            laplacian
            - (phi_values**2 - 1.0)
            * phi_values
            / epsilon**2
        )

    slice_data = values.reshape(shape)
    slice_numpy = np.asarray(
        jax.device_get(slice_data)
    )

    extent = _slice_extent(x, y, z, axis)
    xlabel, ylabel = _slice_axis_labels(axis)

    image = ax.imshow(
        slice_numpy,
        cmap="coolwarm",
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        alpha=1.0,
    )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    if no_label:
        _hide_2d_axis(ax)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)
        elif step is not None:
            ax.set_title(
                f"Step {step}, {axis}-slice={slice_index}/{grid_size}"
            )

        if colorbar:
            plt.colorbar(image, ax=ax, shrink=0.6)

    return image


# ---------------------------------------------------------------------
# Isosurface visualization
# ---------------------------------------------------------------------


def compute_isosurface_mesh_from_checkpoint(
    checkpoint: Mapping[str, Any],
    grid_size: int = 64,
    level: float = 0.0,
    transpose: bool = True,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    hidden_dim: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an isosurface mesh from a trained PINN checkpoint."""
    if grid_size < 2:
        raise ValueError(
            f"grid_size must be at least 2, but received {grid_size}."
        )

    x = jnp.linspace(-1.0, 1.0, grid_size)
    y = jnp.linspace(-1.0, 1.0, grid_size)
    z = jnp.linspace(-1.0, 1.0, grid_size)

    spacing = 2.0 / (grid_size - 1)

    def select_range(
        coordinates: jax.Array,
        coordinate_range: tuple[float, float] | None,
    ) -> tuple[jax.Array, int]:
        if coordinate_range is None:
            return coordinates, 0

        minimum, maximum = coordinate_range
        if minimum > maximum:
            raise ValueError(
                f"Invalid coordinate range {coordinate_range}."
            )

        mask = (
            (coordinates >= minimum)
            & (coordinates <= maximum)
        )
        indices = jnp.where(mask)[0]

        if indices.size == 0:
            raise ValueError(
                f"Range {coordinate_range} selects no grid points."
            )

        start = int(indices[0])
        stop = int(indices[-1]) + 1
        return coordinates[start:stop], start

    x_sub, x_start = select_range(x, x_range)
    y_sub, y_start = select_range(y, y_range)
    z_sub, z_start = select_range(z, z_range)

    first, second, third = jnp.meshgrid(
        x_sub,
        y_sub,
        z_sub,
        indexing="ij",
    )
    grid_points = jnp.stack(
        [
            first.ravel(),
            second.ravel(),
            third.ravel(),
        ],
        axis=-1,
    )

    phi_fn = _build_phi_fn(
        checkpoint,
        hidden_dim=hidden_dim,
    )

    phase_values = phi_fn(grid_points).reshape(
        x_sub.size,
        y_sub.size,
        z_sub.size,
    )

    if transpose:
        phase_values = phase_values.transpose(2, 1, 0)

    phase_numpy = np.array(
        phase_values,
        copy=True,
    )

    vertices, faces, _, _ = marching_cubes(
        phase_numpy,
        level=level,
        spacing=(spacing, spacing, spacing),
    )

    if transpose:
        vertices[:, 0] += z_start * spacing
        vertices[:, 1] += y_start * spacing
        vertices[:, 2] += x_start * spacing
    else:
        vertices[:, 0] += x_start * spacing
        vertices[:, 1] += y_start * spacing
        vertices[:, 2] += z_start * spacing

    vertices -= 1.0

    if transpose:
        vertices = vertices[:, [2, 1, 0]]

    return vertices, faces


def plot_3d_isosurface(
    ax: Any,
    step: int,
    checkpoint: Mapping[str, Any],
    grid_size: int = 64,
    no_label: bool = False,
    hidden_dim: int = 128,
) -> None:
    """Plot the zero level-set isosurface on a Matplotlib 3D axis."""
    vertices, faces = compute_isosurface_mesh_from_checkpoint(
        checkpoint,
        grid_size=grid_size,
        level=0.0,
        hidden_dim=hidden_dim,
    )

    mesh = Poly3DCollection(
        vertices[faces],
        alpha=0.1,
        edgecolor="k",
        linewidth=0.2,
        facecolor="cyan",
    )
    ax.add_collection3d(mesh)

    if no_label:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)

        ax.xaxis.line.set_color((1, 1, 1, 0))
        ax.yaxis.line.set_color((1, 1, 1, 0))
        ax.zaxis.line.set_color((1, 1, 1, 0))

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(-0.5, 0.5)

    else:
        ax.set_title(
            f"Step {step}",
            fontsize=12,
            y=0.9,
        )
        ax.view_init(elev=30, azim=45)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)

    ax.grid(False)
    ax.set_facecolor("white")


def show_isosurface_plotly(
    checkpoint: Mapping[str, Any],
    grid_size: int = 64,
    level: float = 0.0,
    opacity: float = 0.2,
    transpose: bool = True,
    show_axes: bool = False,
    material: str = "membrane",
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    aspect: str = "data",
    hidden_dim: int = 128,
):
    """Create an interactive Plotly isosurface figure."""
    materials = {
        "membrane": {
            "ambient": 0.15,
            "diffuse": 0.9,
            "specular": 0.3,
            "roughness": 0.6,
            "fresnel": 0.1,
        },
        "glossy": {
            "ambient": 0.1,
            "diffuse": 0.7,
            "specular": 0.8,
            "roughness": 0.2,
            "fresnel": 0.3,
        },
        "clay": {
            "ambient": 0.3,
            "diffuse": 0.8,
            "specular": 0.1,
            "roughness": 0.9,
            "fresnel": 0.0,
        },
    }

    if material not in materials:
        raise ValueError(
            f"Unknown material {material!r}. "
            f"Choose from {tuple(materials)}."
        )

    if aspect not in {"data", "manual", "cube"}:
        raise ValueError(
            "aspect must be 'data', 'manual', or 'cube'."
        )

    vertices, faces = compute_isosurface_mesh_from_checkpoint(
        checkpoint=checkpoint,
        grid_size=grid_size,
        level=level,
        transpose=transpose,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        hidden_dim=hidden_dim,
    )

    first, second, third = faces.T

    figure = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=first,
                j=second,
                k=third,
                opacity=opacity,
                color="rgba(180, 190, 210, 1.0)",
                flatshading=False,
                lighting=materials[material],
                lightposition={"x": 1, "y": 1, "z": 2},
            )
        ]
    )

    scene = {
        "xaxis": {
            "visible": show_axes,
            "showbackground": False,
        },
        "yaxis": {
            "visible": show_axes,
            "showbackground": False,
        },
        "zaxis": {
            "visible": show_axes,
            "showbackground": False,
        },
        "bgcolor": "white",
    }

    if aspect == "data":
        scene["aspectmode"] = "data"

    elif aspect == "manual":
        scene["aspectmode"] = "manual"
        scene["aspectratio"] = {
            "x": 1.0,
            "y": 1.0,
            "z": 0.3,
        }

    else:
        scene["aspectmode"] = "cube"
        scene["xaxis"]["range"] = (
            [-1.0, 1.0]
            if x_range is None
            else list(x_range)
        )
        scene["yaxis"]["range"] = (
            [-1.0, 1.0]
            if y_range is None
            else list(y_range)
        )
        scene["zaxis"]["range"] = (
            [-1.0, 1.0]
            if z_range is None
            else list(z_range)
        )

    figure.update_layout(
        scene=scene,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    return figure


# ---------------------------------------------------------------------
# Loss and metric plotting
# ---------------------------------------------------------------------


def plot_loss_history_ax(
    ax: Any,
    loss_history: Mapping[str, np.ndarray],
) -> None:
    """Plot all loss components on one Matplotlib axis."""
    for key, (label, marker) in LOSS_PLOT_SETTINGS.items():
        ax.plot(
            loss_history["step"],
            loss_history[key],
            label=label,
            marker=marker,
            linestyle="-",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
    )
    ax.legend()


def plot_unnormalized_loss_history_ax(
    ax: Any,
    component_index: int,
    loss_history: Mapping[str, np.ndarray],
) -> None:
    """Plot one unnormalized loss component on an existing axis."""
    components = {
        0: ("total_loss", "Total loss"),
        1: ("data_loss", "Data loss"),
        2: ("sign_loss", "Sign loss"),
        3: ("phys_loss", "Physics loss"),
        4: ("curv_loss", "Curvature loss"),
    }

    if component_index not in components:
        raise ValueError(
            "component_index must be an integer from 0 through 4."
        )

    key, title = components[component_index]

    ax.plot(
        loss_history["step"],
        loss_history[key],
        marker="o",
        linestyle="-",
    )
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
    )


def plot_phase_metrics_ax(
    ax: Any,
    checkpoints: Mapping[int, Mapping[str, Any]],
    metric: str,
    epsilon: float = 0.05,
    num_collocation: int = 1000,
    reference_volume: float | None = None,
    reference_area: float | None = None,
) -> None:
    """Plot estimated phase volume or surface area across checkpoints."""
    if metric not in {"volume", "area"}:
        raise ValueError(
            f"metric must be 'volume' or 'area', but received {metric!r}."
        )

    key = jax.random.PRNGKey(0)
    sample_points = jax.random.uniform(
        key,
        shape=(num_collocation, 3),
        minval=-1.0,
        maxval=1.0,
    )

    steps = []
    values = []

    for step, checkpoint in checkpoints.items():
        phi_fn = _build_phi_fn(checkpoint)

        if metric == "volume":
            value = phase_volume(
                phi_fn,
                sample_points,
            )
        else:
            value = phase_surface(
                phi_fn,
                sample_points,
                epsilon,
            )

        steps.append(step)
        values.append(float(value))

    ax.plot(
        steps,
        values,
        label=metric,
    )

    if metric == "volume" and reference_volume is not None:
        ax.axhline(
            y=reference_volume,
            linestyle="--",
            color="gray",
        )

    if metric == "area" and reference_area is not None:
        ax.axhline(
            y=reference_area,
            linestyle="--",
            color="gray",
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel(metric.capitalize())


# ---------------------------------------------------------------------
# VTK export
# ---------------------------------------------------------------------


def write_vtk_point_cloud(
    filename: str | Path,
    points: Any,
    point_scalars: Mapping[str, Any] | None = None,
) -> None:
    """Write a point cloud as a legacy ASCII VTK POLYDATA file."""
    filename = Path(filename)
    points_array = np.asarray(
        points,
        dtype=np.float64,
    )

    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(
            "points must have shape (N, 3), "
            f"but received {points_array.shape}."
        )

    n_points = points_array.shape[0]
    scalar_arrays = (
        {}
        if point_scalars is None
        else dict(point_scalars)
    )

    for name, values in scalar_arrays.items():
        values = np.asarray(values)

        if values.shape != (n_points,):
            raise ValueError(
                f"Scalar {name!r} must have shape ({n_points},), "
                f"but received {values.shape}."
            )

    with filename.open("w", encoding="utf-8") as vtk_file:
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("Combined point cloud\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET POLYDATA\n")

        vtk_file.write(f"POINTS {n_points} float\n")
        for point in points_array:
            vtk_file.write(
                f"{point[0]} {point[1]} {point[2]}\n"
            )

        vtk_file.write(
            f"\nVERTICES {n_points} {2 * n_points}\n"
        )
        for index in range(n_points):
            vtk_file.write(f"1 {index}\n")

        if scalar_arrays:
            vtk_file.write(f"\nPOINT_DATA {n_points}\n")

            for name, values in scalar_arrays.items():
                values = np.asarray(
                    values,
                    dtype=np.float64,
                )

                vtk_file.write(
                    f"SCALARS {name} float 1\n"
                )
                vtk_file.write(
                    "LOOKUP_TABLE default\n"
                )

                for value in values:
                    if np.isnan(value):
                        vtk_file.write("nan\n")
                    else:
                        vtk_file.write(f"{value}\n")


def combine_datasets_to_vtk(
    filename: str | Path,
    data_edge: Mapping[str, Any],
    data_sign: Mapping[str, Any],
    data_phys: Mapping[str, Any],
) -> None:
    """Combine edge, sign, and physics samples into one VTK point cloud."""
    edge_points = np.asarray(
        data_edge["points"],
        dtype=np.float32,
    )
    sign_points = np.asarray(
        data_sign["points"],
        dtype=np.float32,
    )
    physics_points = np.asarray(
        data_phys["points"],
        dtype=np.float32,
    )

    n_edge = edge_points.shape[0]
    n_sign = sign_points.shape[0]
    n_physics = physics_points.shape[0]

    points = np.concatenate(
        [
            edge_points,
            sign_points,
            physics_points,
        ],
        axis=0,
    )

    source_id = np.concatenate(
        [
            np.full(n_edge, 0, dtype=np.float32),
            np.full(n_sign, 1, dtype=np.float32),
            np.full(n_physics, 2, dtype=np.float32),
        ]
    )

    edge_label = np.concatenate(
        [
            np.asarray(
                data_edge["label"],
                dtype=np.float32,
            ),
            np.full(n_sign, np.nan, dtype=np.float32),
            np.full(n_physics, np.nan, dtype=np.float32),
        ]
    )

    sign_label = np.concatenate(
        [
            np.full(n_edge, np.nan, dtype=np.float32),
            np.asarray(
                data_sign["label"],
                dtype=np.float32,
            ),
            np.full(n_physics, np.nan, dtype=np.float32),
        ]
    )

    write_vtk_point_cloud(
        filename,
        points,
        point_scalars={
            "source_id": source_id,
            "edge_label": edge_label,
            "sign_label": sign_label,
        },
    )


