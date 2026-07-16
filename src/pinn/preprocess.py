from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy import ndimage as ndi
from skimage.transform import resize
import jax


GRAY_COLORMAP = LinearSegmentedColormap.from_list(
    "custom_gray",
    ["#f0f0f0", "#111111"],
)

AXIS_TO_DIM = {
    "x": 0,
    "y": 1,
    "z": 2,
}

MASK_COLORS = {
    "edge": "Greens",
    "bulk": "Oranges",
    "outside": "Reds",
    "inside": "Blues",
}


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------


def _validate_axis(axis: str) -> tuple[str, int]:
    """Validate an axis name and return its array dimension."""
    axis = axis.lower()

    if axis not in AXIS_TO_DIM:
        raise ValueError(
            f"axis must be one of {tuple(AXIS_TO_DIM)}, "
            f"but received {axis!r}."
        )

    return axis, AXIS_TO_DIM[axis]


def _validate_slice_index(
    slice_index: int,
    dimension_size: int,
    axis: str,
) -> int:
    """Validate a slice index."""
    slice_index = int(slice_index)

    if not 0 <= slice_index < dimension_size:
        raise ValueError(
            f"slice_index {slice_index} is out of bounds for "
            f"axis {axis!r} with size {dimension_size}."
        )

    return slice_index


def _extract_slice(
    volume: Any,
    axis: str,
    slice_index: int,
) -> np.ndarray:
    """Extract a two-dimensional slice from a three-dimensional volume."""
    axis, dimension = _validate_axis(axis)
    volume = np.asarray(volume)

    if volume.ndim != 3:
        raise ValueError(
            f"volume must be three-dimensional, but received {volume.shape}."
        )

    slice_index = _validate_slice_index(
        slice_index,
        volume.shape[dimension],
        axis,
    )

    return np.take(
        volume,
        slice_index,
        axis=dimension,
    )


def _voxel_indices_to_normalized(
    indices: Any,
    volume_shape: Sequence[int],
) -> np.ndarray:
    """Convert voxel indices to normalized coordinates in [-1, 1]."""
    indices = np.asarray(
        indices,
        dtype=np.float32,
    )
    shape = np.asarray(
        volume_shape,
        dtype=np.float32,
    )

    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(
            "indices must have shape (N, 3), "
            f"but received {indices.shape}."
        )

    if shape.shape != (3,):
        raise ValueError(
            "volume_shape must contain three values."
        )

    if np.any(shape <= 1):
        raise ValueError(
            "Every volume dimension must be greater than one."
        )

    return 2.0 * indices / (shape[None, :] - 1.0) - 1.0


def _normalized_xyz_to_voxel_zyx(
    points_xyz: Any,
    shape_zyx: Sequence[int],
) -> np.ndarray:
    """Convert normalized xyz points to floating-point zyx voxel coordinates."""
    points = np.asarray(
        points_xyz,
        dtype=np.float32,
    )
    shape = np.asarray(
        shape_zyx,
        dtype=np.float32,
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "points_xyz must have shape (N, 3), "
            f"but received {points.shape}."
        )

    if shape.shape != (3,):
        raise ValueError(
            "shape_zyx must contain three values."
        )

    x_voxel = (points[:, 0] + 1.0) * 0.5 * (shape[2] - 1.0)
    y_voxel = (points[:, 1] + 1.0) * 0.5 * (shape[1] - 1.0)
    z_voxel = (points[:, 2] + 1.0) * 0.5 * (shape[0] - 1.0)

    voxel_points = np.stack(
        [z_voxel, y_voxel, x_voxel],
        axis=1,
    )

    lower = np.zeros(3, dtype=np.float32)
    upper = shape - 1.0

    return np.clip(
        voxel_points,
        lower,
        upper,
    )


def _voxel_zyx_to_normalized_xyz(
    points_zyx: Any,
    shape_zyx: Sequence[int],
) -> np.ndarray:
    """Convert zyx voxel coordinates to normalized xyz coordinates."""
    points = np.asarray(
        points_zyx,
        dtype=np.float32,
    )
    shape = np.asarray(
        shape_zyx,
        dtype=np.float32,
    )

    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "points_zyx must have shape (N, 3), "
            f"but received {points.shape}."
        )

    if np.any(shape <= 1):
        raise ValueError(
            "Every volume dimension must be greater than one."
        )

    z_normalized = 2.0 * points[:, 0] / (shape[0] - 1.0) - 1.0
    y_normalized = 2.0 * points[:, 1] / (shape[1] - 1.0) - 1.0
    x_normalized = 2.0 * points[:, 2] / (shape[2] - 1.0) - 1.0

    return np.stack(
        [x_normalized, y_normalized, z_normalized],
        axis=1,
    ).astype(np.float32)


def _project_points_to_slice(
    points: np.ndarray,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D normalized coordinates onto a displayed slice."""
    if dimension == 0:
        return points[:, 1], points[:, 2]

    if dimension == 1:
        return points[:, 0], points[:, 2]

    return points[:, 0], points[:, 1]


def _plot_mask_overlay(
    ax: Any,
    background: np.ndarray,
    masks: Mapping[str, np.ndarray],
    alpha: float = 0.35,
) -> None:
    """Plot a grayscale background with colored binary-mask overlays."""
    ax.imshow(
        background,
        cmap=GRAY_COLORMAP,
        origin="lower",
    )

    for name, mask in masks.items():
        if name not in MASK_COLORS:
            raise ValueError(
                f"Unknown mask name {name!r}."
            )

        mask = np.asarray(
            mask,
            dtype=np.float32,
        )

        ax.imshow(
            np.ma.masked_where(mask == 0, mask),
            cmap=MASK_COLORS[name],
            vmin=0,
            vmax=1,
            alpha=alpha,
            origin="lower",
        )


def _mask_legend_elements() -> list[Patch]:
    """Build standard legend handles for preprocessing masks."""
    return [
        Patch(
            facecolor=plt.get_cmap(MASK_COLORS["edge"])(0.8),
            edgecolor="none",
            label="Edge",
        ),
        Patch(
            facecolor=plt.get_cmap(MASK_COLORS["bulk"])(0.8),
            edgecolor="none",
            label="Bulk",
        ),
        Patch(
            facecolor=plt.get_cmap(MASK_COLORS["outside"])(0.8),
            edgecolor="none",
            label="Outside",
        ),
        Patch(
            facecolor=plt.get_cmap(MASK_COLORS["inside"])(0.8),
            edgecolor="none",
            label="Inside",
        ),
    ]


def _masks_to_labeled_points(
    masks: Sequence[Any],
    labels: Sequence[float],
    volume_shape: Sequence[int],
    axis_perm: tuple[int, int, int] = (2, 1, 0),
    max_points_per_mask: int | None = None,
    seed: int = 0,
) -> dict[str, jax.Array]:
    """Convert multiple voxel masks into normalized labeled point samples."""
    if len(masks) != len(labels):
        raise ValueError(
            "masks and labels must have the same length."
        )

    if sorted(axis_perm) != [0, 1, 2]:
        raise ValueError(
            "axis_perm must be a permutation of (0, 1, 2)."
        )

    rng = np.random.default_rng(seed)
    shape_permuted = np.asarray(volume_shape)[list(axis_perm)]

    point_groups = []
    label_groups = []

    for mask, label in zip(masks, labels):
        indices = np.argwhere(
            np.asarray(mask).astype(bool)
        )

        if (
            max_points_per_mask is not None
            and indices.shape[0] > max_points_per_mask
        ):
            selected = rng.choice(
                indices.shape[0],
                size=max_points_per_mask,
                replace=False,
            )
            indices = indices[selected]

        indices = indices[:, axis_perm]
        points = _voxel_indices_to_normalized(
            indices,
            shape_permuted,
        )

        point_groups.append(points)
        label_groups.append(
            np.full(
                points.shape[0],
                label,
                dtype=np.float32,
            )
        )

    if not point_groups:
        points = np.empty((0, 3), dtype=np.float32)
        point_labels = np.empty((0,), dtype=np.float32)
    else:
        points = np.concatenate(point_groups, axis=0)
        point_labels = np.concatenate(label_groups, axis=0)

    return {
        "points": jnp.asarray(points),
        "label": jnp.asarray(point_labels),
    }


# ---------------------------------------------------------------------
# Intensity preprocessing
# ---------------------------------------------------------------------


def clahe_volume_slicewise(
    volume: Any,
    axis: int = 0,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE independently to slices along one volume axis."""
    volume = np.asarray(
        volume,
        dtype=np.float32,
    )

    if volume.ndim != 3:
        raise ValueError(
            f"volume must be three-dimensional, but received {volume.shape}."
        )

    if axis not in {0, 1, 2}:
        raise ValueError(
            f"axis must be 0, 1, or 2, but received {axis}."
        )

    volume = np.clip(volume, 0.0, 1.0)
    volume_uint8 = np.round(
        volume * 255.0
    ).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    moved = np.moveaxis(
        volume_uint8,
        axis,
        0,
    )
    output = np.empty_like(moved)

    for index in range(moved.shape[0]):
        output[index] = clahe.apply(moved[index])

    return (
        np.moveaxis(output, 0, axis).astype(np.float32)
        / 255.0
    )


def crop_pad_and_resample_volume(
    cryoet_data: Any,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    z_start: int,
    z_end: int,
    *,
    grid_size: int = 64,
    intensity_max: float = 1.0,
    shrink_z_layers: int = 1,
    zero_xy_layers: int | None = None,
) -> np.ndarray:
    """Crop, pad to a cube, resample, mask padding, and normalize a volume."""
    volume = np.asarray(
        cryoet_data,
        dtype=np.float32,
    )

    cropped = volume[
        z_start:z_end,
        y_start:y_end,
        x_start:x_end,
    ]

    if cropped.size == 0:
        raise ValueError(
            "The requested crop is empty."
        )

    z_size, y_size, x_size = cropped.shape
    target_size = max(
        z_size,
        y_size,
        x_size,
    )

    def padding_for(size: int) -> tuple[int, int]:
        total = target_size - size
        before = total // 2
        return before, total - before

    z_padding = padding_for(z_size)
    y_padding = padding_for(y_size)
    x_padding = padding_for(x_size)

    pad_width = (
        z_padding,
        y_padding,
        x_padding,
    )

    padded = np.pad(
        cropped,
        pad_width=pad_width,
        mode="constant",
        constant_values=0,
    )

    valid_mask = np.pad(
        np.ones_like(cropped, dtype=np.uint8),
        pad_width=pad_width,
        mode="constant",
        constant_values=0,
    )

    output_shape = (
        grid_size,
        grid_size,
        grid_size,
    )

    resized_volume = resize(
        padded,
        output_shape,
        mode="constant",
        cval=0,
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)

    resized_mask = resize(
        valid_mask.astype(np.float32),
        output_shape,
        mode="constant",
        cval=0,
        anti_aliasing=False,
        order=0,
        preserve_range=True,
    ) > 0.5

    if zero_xy_layers is not None and zero_xy_layers > 0:
        layer_count = int(zero_xy_layers)

        resized_mask[:, :, :layer_count] = False
        resized_mask[:, :, -layer_count:] = False
        resized_mask[:, :layer_count, :] = False
        resized_mask[:, -layer_count:, :] = False

    valid_z = np.where(
        resized_mask.any(axis=(1, 2))
    )[0]

    if (
        shrink_z_layers > 0
        and valid_z.size > 0
    ):
        first_valid = valid_z[0]
        last_valid = valid_z[-1]
        count = int(shrink_z_layers)

        resized_mask[
            first_valid : first_valid + count,
            :,
            :,
        ] = False
        resized_mask[
            last_valid - count + 1 : last_valid + 1,
            :,
            :,
        ] = False

    resized_volume[~resized_mask] = 0.0

    valid_values = resized_volume[resized_mask]

    if valid_values.size == 0:
        raise ValueError(
            "No valid voxels remain after masking."
        )

    minimum = float(valid_values.min())
    maximum = float(valid_values.max())

    if maximum > minimum:
        normalized = (
            resized_volume - minimum
        ) / (maximum - minimum)
    else:
        normalized = np.zeros_like(
            resized_volume,
            dtype=np.float32,
        )

    normalized *= intensity_max
    normalized = np.clip(
        normalized,
        0.0,
        intensity_max,
    )
    normalized[~resized_mask] = 0.0

    return normalized.astype(np.float32)


def center_crop_or_pad_to_cube(
    volume: Any,
    size: int,
    fill_value: float = 0,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Centrally crop or pad a 3D volume to a cubic shape."""
    volume = np.asarray(volume)

    if volume.ndim != 3:
        raise ValueError(
            f"volume must be three-dimensional, but received {volume.shape}."
        )

    if size <= 0:
        raise ValueError(
            f"size must be positive, but received {size}."
        )

    output = np.full(
        (size, size, size),
        fill_value,
        dtype=volume.dtype,
    )

    source_slices = []
    destination_slices = []
    transforms = []

    for old_size in volume.shape:
        if old_size >= size:
            source_start = (old_size - size) // 2
            source_end = source_start + size
            destination_start = 0
            destination_end = size
        else:
            source_start = 0
            source_end = old_size
            destination_start = (size - old_size) // 2
            destination_end = destination_start + old_size

        source_slices.append(
            slice(source_start, source_end)
        )
        destination_slices.append(
            slice(destination_start, destination_end)
        )

        transforms.append(
            {
                "start_src": source_start,
                "end_src": source_end,
                "start_dst": destination_start,
                "end_dst": destination_end,
                "shift": destination_start - source_start,
            }
        )

    output[tuple(destination_slices)] = volume[
        tuple(source_slices)
    ]

    return output, transforms


# ---------------------------------------------------------------------
# Point-overlay visualization
# ---------------------------------------------------------------------

def overlay_points_on_cryoet(
    cryoet,
    overlays=None,
    axis="z",
    slice_index=None,
    threshold=None,
    pts_alpha=0.8,
    cryo_alpha=1.0,
    pts_size=20,
    tol=None,
    show_title=False,
    show_legend=False,
    figsize=(4, 4),
    ncols=6,
    save_path=None,
    expand_xy=None,
):
    """Plot one or more cryo-ET slices with optional point overlays."""
    cryoet = np.asarray(cryoet)

    axis_to_dim = {
        "x": 0,
        "y": 1,
        "z": 2,
    }

    if axis not in axis_to_dim:
        raise ValueError(
            f"axis must be one of {tuple(axis_to_dim)}, got {axis!r}."
        )

    dim = axis_to_dim[axis]
    dim_size = cryoet.shape[dim]

    if slice_index is None:
        slice_list = [dim_size // 2]
    elif np.isscalar(slice_index):
        slice_list = [int(slice_index)]
    else:
        slice_list = [int(index) for index in slice_index]

    if not slice_list:
        raise ValueError("At least one slice index is required.")

    for index in slice_list:
        if index < 0 or index >= dim_size:
            raise ValueError(
                f"slice_index {index} is out of bounds for "
                f"axis {axis!r} with size {dim_size}."
            )

    n_slices = len(slice_list)
    ncols = min(max(int(ncols), 1), n_slices)
    nrows = int(np.ceil(n_slices / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            figsize[0] * ncols,
            figsize[1] * nrows,
        ),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, index in zip(axes_flat, slice_list):
        overlay_points_on_cryoet_ax(
            ax=ax,
            cryoet=cryoet,
            overlays=overlays,
            threshold=threshold,
            axis=axis,
            slice_index=index,
            pts_alpha=pts_alpha,
            cryo_alpha=cryo_alpha,
            pts_size=pts_size,
            tol=tol,
            show_title=show_title,
            show_legend=show_legend,
            expand_xy=expand_xy,
        )

    for ax in axes_flat[n_slices:]:
        ax.axis("off")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            save_path,
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()

    return fig, axes


def overlay_points_on_cryoet_ax(
    ax,
    cryoet,
    overlays=None,
    threshold=None,
    axis="z",
    slice_index=None,
    pts_alpha=0.8,
    cryo_alpha=1.0,
    pts_size=20,
    tol=None,
    show_title=False,
    show_legend=False,
    expand_xy=None,
):
    """Plot a cryo-ET slice and normalized point overlays on an axis."""
    cryoet = np.asarray(cryoet)

    axis_to_dim = {
        "x": 0,
        "y": 1,
        "z": 2,
    }

    if axis not in axis_to_dim:
        raise ValueError(
            f"axis must be one of {tuple(axis_to_dim)}, got {axis!r}."
        )

    dim = axis_to_dim[axis]
    dim_size = cryoet.shape[dim]

    if slice_index is None:
        slice_index = dim_size // 2

    if slice_index < 0 or slice_index >= dim_size:
        raise ValueError(
            f"slice_index {slice_index} is out of bounds for "
            f"axis {axis!r} with size {dim_size}."
        )

    display_volume = cryoet
    if threshold is not None:
        display_volume = np.where(
            cryoet > threshold,
            1.0,
            0.0,
        )

    if tol is None:
        tol = 1.0 / (dim_size - 1)

    cryo_slice = np.take(
        display_volume,
        slice_index,
        axis=dim,
    )

    custom_gray = LinearSegmentedColormap.from_list(
        "custom_gray",
        ["#f0f0f0", "#111111"],
    )

    ax.imshow(
        cryo_slice,
        cmap=custom_gray,
        origin="lower",
        extent=(-1.0, 1.0, -1.0, 1.0),
        alpha=cryo_alpha,
    )

    slice_position = (
        -1.0
        + 2.0 * slice_index / (dim_size - 1)
    )

    legend_elements = []

    if overlays is not None:
        for overlay in overlays:
            points = np.asarray(
                overlay["data"],
                dtype=float,
            )

            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(
                    "overlay['data'] must have shape (N, 3)."
                )

            label = overlay.get("label", "Overlay")
            color = overlay.get("color", None)
            point_size = overlay.get(
                "point_size",
                pts_size,
            )

            mask = (
                np.abs(points[:, dim] - slice_position)
                <= tol
            )
            selected_points = points[mask]

            if selected_points.shape[0] > 0:
                if dim == 0:
                    px = selected_points[:, 1]
                    py = selected_points[:, 2]
                elif dim == 1:
                    px = selected_points[:, 0]
                    py = selected_points[:, 2]
                else:
                    px = selected_points[:, 0]
                    py = selected_points[:, 1]

                if expand_xy is not None:
                    if expand_xy <= 0:
                        raise ValueError(
                            "expand_xy must be positive."
                        )

                    in_view = (
                        (np.abs(px) <= expand_xy)
                        & (np.abs(py) <= expand_xy)
                    )
                    px = px[in_view] / expand_xy
                    py = py[in_view] / expand_xy

                if px.size > 0:
                    # Preserve the original image-coordinate convention.
                    ax.scatter(
                        py,
                        px,
                        s=point_size,
                        color=color,
                        alpha=pts_alpha,
                        edgecolors="none",
                    )

            legend_elements.append(
                Patch(
                    facecolor=color,
                    edgecolor="none",
                    label=label,
                )
            )

    if show_title:
        ax.set_title(
            f"{axis} = {slice_index}/{dim_size}",
            fontsize=20,
        )

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.axis("off")

    if show_legend and legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            frameon=True,
            fontsize=9,
        )


# ---------------------------------------------------------------------
# Mask visualization
# ---------------------------------------------------------------------


def overlay_masks_on_cryoet_slices(
    cryoet: Any,
    edge: Any,
    bulk: Any,
    outside: Any,
    inside: Any,
    axis: str = "z",
    slice_indices: Sequence[int] = (0,),
    alpha: float = 0.35,
    show_title: bool = True,
    figsize_per_panel: float = 3.0,
) -> tuple[Any, np.ndarray]:
    """Plot mask overlays for one or more cryo-ET slices."""
    axis, dimension = _validate_axis(axis)
    cryoet = np.asarray(cryoet)

    slice_indices = [
        int(index)
        for index in slice_indices
    ]

    if not slice_indices:
        raise ValueError(
            "slice_indices must contain at least one index."
        )

    dimension_size = cryoet.shape[dimension]
    for index in slice_indices:
        _validate_slice_index(
            index,
            dimension_size,
            axis,
        )

    figure, axes = plt.subplots(
        1,
        len(slice_indices),
        figsize=(
            figsize_per_panel * len(slice_indices),
            figsize_per_panel,
        ),
        squeeze=False,
    )
    axes = axes[0]

    volumes = {
        "edge": np.asarray(edge),
        "bulk": np.asarray(bulk),
        "outside": np.asarray(outside),
        "inside": np.asarray(inside),
    }

    for ax, index in zip(
        axes,
        slice_indices,
    ):
        mask_slices = {
            name: np.take(
                volume,
                index,
                axis=dimension,
            ).astype(np.float32)
            for name, volume in volumes.items()
        }

        cryo_slice = np.take(
            cryoet,
            index,
            axis=dimension,
        )

        _plot_mask_overlay(
            ax,
            cryo_slice,
            mask_slices,
            alpha=alpha,
        )

        if show_title:
            ax.set_title(
                f"{axis}={index}/{dimension_size - 1}",
                fontsize=10,
            )

        ax.axis("off")

    figure.legend(
        handles=_mask_legend_elements(),
        loc="center right",
        frameon=True,
        fontsize=9,
    )
    figure.tight_layout(
        rect=[0, 0, 0.92, 1]
    )

    return figure, axes


def plot_preprocessing_masks(
    cryoet: Any,
    edge: Any,
    bulk: Any,
    outside: Any,
    inside: Any,
    axis: str = "z",
    slice_index: int | None = None,
    alpha: float = 0.35,
    show_title: bool = True,
) -> tuple[Any, np.ndarray]:
    """Show the original volume, individual masks, and their overlay."""
    axis, dimension = _validate_axis(axis)
    cryoet = np.asarray(cryoet)
    dimension_size = cryoet.shape[dimension]

    if slice_index is None:
        slice_index = dimension_size // 2

    slice_index = _validate_slice_index(
        slice_index,
        dimension_size,
        axis,
    )

    cryo_slice = _extract_slice(
        cryoet,
        axis,
        slice_index,
    )
    edge_slice = _extract_slice(
        edge,
        axis,
        slice_index,
    ).astype(np.float32)
    bulk_slice = _extract_slice(
        bulk,
        axis,
        slice_index,
    ).astype(np.float32)
    outside_slice = _extract_slice(
        outside,
        axis,
        slice_index,
    ).astype(np.float32)
    inside_slice = _extract_slice(
        inside,
        axis,
        slice_index,
    ).astype(np.float32)

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 7),
        squeeze=False,
    )

    panels = [
        (
            axes[0, 0],
            cryo_slice,
            GRAY_COLORMAP,
            "Original",
        ),
        (
            axes[0, 1],
            edge_slice,
            "gray_r",
            "Edge",
        ),
        (
            axes[0, 2],
            bulk_slice,
            "gray_r",
            "Bulk",
        ),
        (
            axes[1, 1],
            outside_slice,
            "gray_r",
            "Outside",
        ),
        (
            axes[1, 2],
            inside_slice,
            "gray_r",
            "Inside",
        ),
    ]

    for ax, image, cmap, title in panels:
        ax.imshow(
            image,
            cmap=cmap,
            vmin=(
                0
                if cmap == "gray_r"
                else None
            ),
            vmax=(
                1
                if cmap == "gray_r"
                else None
            ),
        )

        if show_title:
            ax.set_title(
                f"{title} "
                f"({axis}: {slice_index}/{dimension_size})"
            )

    overlay_axis = axes[1, 0]
    _plot_mask_overlay(
        overlay_axis,
        cryo_slice,
        {
            "edge": edge_slice,
            "bulk": bulk_slice,
            "outside": outside_slice,
            "inside": inside_slice,
        },
        alpha=alpha,
    )

    if show_title:
        overlay_axis.set_title(
            f"Overlay "
            f"({axis}: {slice_index}/{dimension_size})"
        )

    overlay_axis.legend(
        handles=_mask_legend_elements(),
        loc="upper right",
        frameon=True,
        fontsize=9,
    )

    for ax in axes.ravel():
        ax.axis("off")

    figure.tight_layout()
    return figure, axes


# ---------------------------------------------------------------------
# Mask processing and training-data construction
# ---------------------------------------------------------------------


def downsample_binary_mask_to_fraction(
    mask: Any,
    target_frac: float,
    seed: int | None = None,
) -> np.ndarray:
    """Randomly remove positive voxels until the requested global fraction remains."""
    if not 0.0 <= target_frac <= 1.0:
        raise ValueError(
            "target_fraction must be between 0 and 1."
        )

    random_generator = np.random.default_rng(seed)
    mask_boolean = np.asarray(mask).astype(
        bool,
        copy=False,
    )

    total_voxels = mask_boolean.size
    current_positive = int(
        mask_boolean.sum()
    )
    desired_positive = int(
        round(
            target_frac
            * total_voxels
        )
    )

    if current_positive <= desired_positive:
        return mask_boolean.astype(
            np.float32
        )

    positive_indices = np.flatnonzero(
        mask_boolean
    )
    n_remove = (
        current_positive
        - desired_positive
    )

    remove_indices = random_generator.choice(
        positive_indices,
        size=n_remove,
        replace=False,
    )

    output = mask_boolean.copy().ravel()
    output[remove_indices] = False

    return output.reshape(
        mask_boolean.shape
    ).astype(np.float32)


def build_data_sign_from_masks(
    outside: Any,
    inside: Any,
    volume_shape: Sequence[int],
    axis_perm: tuple[int, int, int] = (2, 1, 0),
    subsample: int | None = None,
    seed: int = 0,
) -> dict[str, jax.Array]:
    """Build normalized sign-training data from outside and inside masks."""
    return _masks_to_labeled_points(
        masks=[outside, inside],
        labels=[1.0, -1.0],
        volume_shape=volume_shape,
        axis_perm=axis_perm,
        max_points_per_mask=subsample,
        seed=seed,
    )


def build_data_edge_from_masks(
    edge: Any,
    bulk: Any,
    volume_shape: Sequence[int],
    axis_perm: tuple[int, int, int] = (2, 1, 0),
    subsample: int | None = None,
    seed: int = 0,
) -> dict[str, jax.Array]:
    """Build normalized edge-training data from edge and bulk masks."""
    return _masks_to_labeled_points(
        masks=[edge, bulk],
        labels=[1.0, 0.0],
        volume_shape=volume_shape,
        axis_perm=axis_perm,
        max_points_per_mask=subsample,
        seed=seed,
    )


def keep_if_enough_neighbors(
    mask: Any,
    minimum_neighbors: int = 6,
    radius: int = 1,
    connectivity: str = "26",
    include_center: bool = False,
) -> np.ndarray:
    """Keep positive voxels having at least a minimum number of neighbors."""
    binary = (
        np.asarray(mask) > 0
    ).astype(np.uint8)

    if radius < 1:
        raise ValueError(
            "radius must be at least one."
        )

    if connectivity == "26":
        structure = np.ones(
            (2 * radius + 1,) * 3,
            dtype=np.uint8,
        )

    elif connectivity == "6":
        if radius != 1:
            raise ValueError(
                "connectivity='6' supports radius=1 only."
            )

        structure = ndi.generate_binary_structure(
            3,
            1,
        ).astype(np.uint8)

    else:
        raise ValueError(
            "connectivity must be '6' or '26'."
        )

    if not include_center:
        structure = structure.copy()
        structure[
            radius,
            radius,
            radius,
        ] = 0

    neighbor_count = ndi.convolve(
        binary,
        structure,
        mode="constant",
        cval=0,
    )

    return (
        (binary == 1)
        & (neighbor_count >= minimum_neighbors)
    ).astype(np.uint8)


# ---------------------------------------------------------------------
# Napari sign labeling
# ---------------------------------------------------------------------


def signs_to_napari_points(
    signs: Mapping[str, Any],
    shape_zyx: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Split normalized sign points into positive and negative Napari points."""
    points = np.asarray(
        signs["points"],
        dtype=np.float32,
    )
    labels = np.asarray(
        signs["label"],
        dtype=np.float32,
    ).reshape(-1)

    if labels.shape[0] != points.shape[0]:
        raise ValueError(
            "sign points and labels must have the same length."
        )

    voxel_points = _normalized_xyz_to_voxel_zyx(
        points,
        shape_zyx,
    )

    positive = voxel_points[labels > 0]
    negative = voxel_points[labels < 0]

    return (
        positive.reshape(-1, 3),
        negative.reshape(-1, 3),
    )


def napari_sign_labeler(
    cryoet_data: Any,
    save_path: str | Path = "manual_signs.npz",
    lower_percentile: float = 2,
    upper_percentile: float = 98,
    point_size: float = 1,
):
    """Open an interactive Napari viewer for manually labeling sign points."""
    try:
        import napari
    except ImportError as exc:
        raise ImportError(
            "napari_sign_labeler requires the optional 'napari' package."
        ) from exc

    volume = np.asarray(cryoet_data)
    save_path = Path(save_path)

    viewer = napari.Viewer(
        ndisplay=2
    )
    viewer.add_image(
        volume,
        name="cryoET",
        contrast_limits=(
            np.percentile(
                volume,
                lower_percentile,
            ),
            np.percentile(
                volume,
                upper_percentile,
            ),
        ),
        colormap="gray",
    )

    positive_layer = viewer.add_points(
        name="sign_plus (+1 outside)",
        ndim=3,
        size=point_size,
        face_color="red",
    )
    negative_layer = viewer.add_points(
        name="sign_minus (-1 inside)",
        ndim=3,
        size=point_size,
        face_color="blue",
    )

    if save_path.exists():
        with np.load(
            save_path,
            allow_pickle=True,
        ) as archive:
            if "points" not in archive.files:
                raise KeyError(
                    f"{save_path} is missing the required 'points' key."
                )

            if "label" in archive.files:
                labels = archive["label"]
            elif "sign" in archive.files:
                labels = archive["sign"]
            else:
                raise KeyError(
                    f"{save_path} must contain 'label' or 'sign'."
                )

            signs = {
                "points": archive["points"],
                "label": labels,
            }

        positive, negative = signs_to_napari_points(
            signs,
            volume.shape,
        )

        positive_layer.data = positive
        negative_layer.data = negative

        print(
            f"Loaded {positive.shape[0]} positive and "
            f"{negative.shape[0]} negative points."
        )

    @viewer.bind_key(
        "Shift-S",
        overwrite=True,
    )
    def save_signs(_viewer):
        positive_xyz = _voxel_zyx_to_normalized_xyz(
            positive_layer.data,
            volume.shape,
        )
        negative_xyz = _voxel_zyx_to_normalized_xyz(
            negative_layer.data,
            volume.shape,
        )

        points = np.concatenate(
            [positive_xyz, negative_xyz],
            axis=0,
        )
        labels = np.concatenate(
            [
                np.ones(
                    positive_xyz.shape[0],
                    dtype=np.float32,
                ),
                -np.ones(
                    negative_xyz.shape[0],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        np.savez(
            save_path,
            points=points,
            label=labels,
        )

        print(
            f"Saved {labels.shape[0]} points to {save_path} "
            f"(+1={positive_xyz.shape[0]}, "
            f"-1={negative_xyz.shape[0]})."
        )

    print(
        "\nNapari sign labeling\n"
        "- Existing points are loaded automatically.\n"
        "- Add points to the positive or negative layer.\n"
        "- Press Shift+S to save.\n"
    )

    return (
        viewer,
        positive_layer,
        negative_layer,
    )


