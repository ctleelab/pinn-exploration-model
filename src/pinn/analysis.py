from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from pinn.model import PINN, grad_phi, hessian_phi
from pinn.plot import compute_isosurface_mesh_from_checkpoint


# ---------------------------------------------------------------------
# Differential-geometry helpers
# ---------------------------------------------------------------------


def adjugate_3x3_batched(matrices: jax.Array) -> jax.Array:
    """Compute adjugates of batched 3×3 matrices.

    Parameters
    ----------
    matrices
        Array with shape ``(N, 3, 3)``.

    Returns
    -------
    jax.Array
        Adjugate matrices with shape ``(N, 3, 3)``.
    """
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(
            "matrices must have shape (N, 3, 3), "
            f"but received {matrices.shape}."
        )

    a00 = matrices[:, 0, 0]
    a01 = matrices[:, 0, 1]
    a02 = matrices[:, 0, 2]
    a10 = matrices[:, 1, 0]
    a11 = matrices[:, 1, 1]
    a12 = matrices[:, 1, 2]
    a20 = matrices[:, 2, 0]
    a21 = matrices[:, 2, 1]
    a22 = matrices[:, 2, 2]

    cofactor_00 = a11 * a22 - a12 * a21
    cofactor_01 = -(a10 * a22 - a12 * a20)
    cofactor_02 = a10 * a21 - a11 * a20

    cofactor_10 = -(a01 * a22 - a02 * a21)
    cofactor_11 = a00 * a22 - a02 * a20
    cofactor_12 = -(a00 * a21 - a01 * a20)

    cofactor_20 = a01 * a12 - a02 * a11
    cofactor_21 = -(a00 * a12 - a02 * a10)
    cofactor_22 = a00 * a11 - a01 * a10

    return jnp.stack(
        [
            jnp.stack(
                [cofactor_00, cofactor_10, cofactor_20],
                axis=1,
            ),
            jnp.stack(
                [cofactor_01, cofactor_11, cofactor_21],
                axis=1,
            ),
            jnp.stack(
                [cofactor_02, cofactor_12, cofactor_22],
                axis=1,
            ),
        ],
        axis=1,
    )


def make_point_geometry_functions(
    phi_fn,
    normal_eps: float = 1e-8,
    curvature_eps: float = 1e-8,
):
    """Create pointwise normal, mean-curvature, and Gaussian-curvature functions."""

    def normal_at_point(point: jax.Array) -> jax.Array:
        gradient = grad_phi(phi_fn, point[None, :])[0]
        gradient_norm = jnp.sqrt(
            jnp.dot(gradient, gradient) + normal_eps**2
        )
        return gradient / gradient_norm

    def kappa_at_point(point: jax.Array) -> jax.Array:
        gradient = grad_phi(phi_fn, point[None, :])[0]
        hessian = hessian_phi(phi_fn, point[None, :])[0]

        gradient_norm = jnp.sqrt(
            jnp.dot(gradient, gradient) + curvature_eps**2
        )
        laplacian = jnp.trace(hessian)
        gradient_hessian_gradient = jnp.dot(
            gradient,
            hessian @ gradient,
        )

        return (
            laplacian / gradient_norm
            - gradient_hessian_gradient / gradient_norm**3
        )

    def gaussian_curvature_at_point(point: jax.Array) -> jax.Array:
        gradient = grad_phi(phi_fn, point[None, :])[0]
        hessian = hessian_phi(phi_fn, point[None, :])[0]

        gradient_norm = jnp.sqrt(
            jnp.dot(gradient, gradient) + curvature_eps**2
        )
        normal = gradient / jnp.sqrt(
            jnp.dot(gradient, gradient) + normal_eps**2
        )

        adjugate = adjugate_3x3_batched(
            hessian[None, :, :]
        )[0]

        return (
            normal @ adjugate @ normal
        ) / gradient_norm**2

    return (
        normal_at_point,
        kappa_at_point,
        gaussian_curvature_at_point,
    )


def surface_laplacian_of_scalar(
    scalar_fn,
    normal_fn,
):
    """Create a function evaluating the surface Laplacian of a scalar field.

    The surface Laplacian is computed as

    ``div((I - n nᵀ) grad(f))``.
    """

    def surface_gradient(point: jax.Array) -> jax.Array:
        normal = normal_fn(point)
        projection = (
            jnp.eye(3, dtype=point.dtype)
            - jnp.outer(normal, normal)
        )
        gradient = jax.grad(scalar_fn)(point)
        return projection @ gradient

    def surface_laplacian(point: jax.Array) -> jax.Array:
        jacobian = jax.jacfwd(surface_gradient)(point)
        return jnp.trace(jacobian)

    return surface_laplacian


# ---------------------------------------------------------------------
# Surface analysis
# ---------------------------------------------------------------------


def calc_norm_curv_K_force(
    checkpoint: Mapping[str, Any],
    grid_size: int = 64,
    normal_eps: float = 1e-8,
    curvature_eps: float = 1e-8,
    curvature_kind: str = "H",
    transpose: bool = True,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    level: float = 0.0,
    kappa_b: float = 1.0,
    sigma: float = 0.0,
    compute_force: bool = True,
    batch_size: int = 4096,
    half_length: float = 1.0,
    hidden_dim: int = 128,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    """Compute mesh geometry and an optional Helfrich force.

    Returns
    -------
    vertices
        Mesh vertices with shape ``(N, 3)``.
    faces
        Triangle indices with shape ``(M, 3)``.
    normals
        Outward-oriented normal vectors with shape ``(N, 3)``.
    theta
        Angle between each normal and the positive x-axis.
    curvature
        Either ``kappa = div(n)`` or ``H = kappa / 2``.
    gaussian_curvature
        Gaussian curvature at each mesh vertex.
    force
        Force vectors with shape ``(N, 3)``, or ``None`` when disabled.
    """
    curvature_kind = curvature_kind.lower()

    if curvature_kind not in {"h", "kappa"}:
        raise ValueError(
            "curvature_kind must be 'H' or 'kappa'."
        )

    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, but received {batch_size}."
        )

    if half_length <= 0:
        raise ValueError(
            f"half_length must be positive, but received {half_length}."
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
    faces = faces.astype(np.int32)

    model = PINN(hidden_dim=hidden_dim)
    params = checkpoint["state"]["params"]

    def phi_fn(points: jax.Array) -> jax.Array:
        return model.apply(params, points)

    vertices_jax = jnp.asarray(vertices)

    gradients = grad_phi(phi_fn, vertices_jax)
    hessians = hessian_phi(phi_fn, vertices_jax)

    gradient_norm = jnp.sqrt(
        jnp.sum(gradients**2, axis=1)
        + normal_eps**2
    )
    normals_jax = gradients / gradient_norm[:, None]

    laplacian = jnp.trace(
        hessians,
        axis1=1,
        axis2=2,
    )
    gradient_hessian_gradient = jnp.einsum(
        "ni,nij,nj->n",
        gradients,
        hessians,
        gradients,
    )

    curvature_norm = jnp.sqrt(
        jnp.sum(gradients**2, axis=1)
        + curvature_eps**2
    )

    kappa = (
        laplacian / curvature_norm
        - gradient_hessian_gradient / curvature_norm**3
    )

    adjugates = adjugate_3x3_batched(hessians)
    gaussian_numerator = jnp.einsum(
        "ni,nij,nj->n",
        normals_jax,
        adjugates,
        normals_jax,
    )
    gaussian_curvature = (
        gaussian_numerator / curvature_norm**2
    )

    # Preserve the original orientation convention.
    normals = -np.asarray(normals_jax)

    if curvature_kind == "h":
        curvature_jax = -0.5 * kappa
    else:
        curvature_jax = -kappa

    force = None

    if compute_force:
        normal_fn, kappa_fn, _ = make_point_geometry_functions(
            phi_fn,
            normal_eps=normal_eps,
            curvature_eps=curvature_eps,
        )
        surface_laplacian_fn = surface_laplacian_of_scalar(
            kappa_fn,
            normal_fn,
        )
        batched_surface_laplacian = jax.vmap(
            surface_laplacian_fn
        )

        force_batches = []
        n_vertices = vertices_jax.shape[0]

        for start in range(0, n_vertices, batch_size):
            stop = min(start + batch_size, n_vertices)
            point_batch = vertices_jax[start:stop]

            surface_laplacian = batched_surface_laplacian(
                point_batch
            )

            kappa_batch = kappa[start:stop]
            gaussian_batch = gaussian_curvature[start:stop]
            normal_batch = normals_jax[start:stop]

            # Active formula preserved from the original implementation.
            force_normal = (
                kappa_b
                * (
                    surface_laplacian
                    - kappa_batch**3
                    + 2.0
                    * kappa_batch
                    * gaussian_batch
                )
                + sigma * kappa_batch
            )

            force_batches.append(
                force_normal[:, None] * normal_batch
            )

        force = np.asarray(
            jnp.concatenate(force_batches, axis=0)
        )

    theta = np.arccos(
        np.clip(
            normals @ np.array([1.0, 0.0, 0.0]),
            -1.0,
            1.0,
        )
    )

    scale = half_length
    vertices = vertices * scale
    curvature = np.asarray(curvature_jax) / scale
    gaussian_curvature_np = (
        np.asarray(gaussian_curvature) / scale**2
    )

    return (
        vertices,
        faces,
        normals,
        theta,
        curvature,
        gaussian_curvature_np,
        force,
    )


# ---------------------------------------------------------------------
# Mesh input/output
# ---------------------------------------------------------------------


def write_vtk_polydata(
    filename: str | Path,
    vertices: Any,
    faces: Any,
    point_vectors: Mapping[str, Any] | None = None,
    point_scalars: Mapping[str, Any] | None = None,
) -> None:
    """Write a triangular mesh as legacy ASCII VTK POLYDATA."""
    filename = Path(filename)

    vertices_array = np.asarray(
        vertices,
        dtype=np.float64,
    )
    faces_array = np.asarray(
        faces,
        dtype=np.int64,
    )

    if (
        vertices_array.ndim != 2
        or vertices_array.shape[1] != 3
    ):
        raise ValueError(
            "vertices must have shape (N, 3), "
            f"but received {vertices_array.shape}."
        )

    if (
        faces_array.ndim != 2
        or faces_array.shape[1] != 3
    ):
        raise ValueError(
            "faces must have shape (M, 3), "
            f"but received {faces_array.shape}."
        )

    vector_fields = (
        {}
        if point_vectors is None
        else dict(point_vectors)
    )
    scalar_fields = (
        {}
        if point_scalars is None
        else dict(point_scalars)
    )

    n_vertices = vertices_array.shape[0]
    n_faces = faces_array.shape[0]

    for name, vectors in vector_fields.items():
        vectors = np.asarray(vectors)

        if vectors.shape != vertices_array.shape:
            raise ValueError(
                f"Vector field {name!r} must have shape "
                f"{vertices_array.shape}, but received {vectors.shape}."
            )

    for name, scalars in scalar_fields.items():
        scalars = np.asarray(scalars).reshape(-1)

        if scalars.shape != (n_vertices,):
            raise ValueError(
                f"Scalar field {name!r} must have shape "
                f"({n_vertices},), but received {scalars.shape}."
            )

    with filename.open("w", encoding="utf-8") as vtk_file:
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("PINN surface mesh\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET POLYDATA\n")

        vtk_file.write(
            f"POINTS {n_vertices} float\n"
        )
        for x, y, z in vertices_array:
            vtk_file.write(
                f"{x:.9g} {y:.9g} {z:.9g}\n"
            )

        vtk_file.write(
            f"POLYGONS {n_faces} {4 * n_faces}\n"
        )
        for first, second, third in faces_array:
            vtk_file.write(
                f"3 {int(first)} {int(second)} {int(third)}\n"
            )

        if vector_fields or scalar_fields:
            vtk_file.write(
                f"\nPOINT_DATA {n_vertices}\n"
            )

        for name, vectors in vector_fields.items():
            vectors = np.asarray(
                vectors,
                dtype=np.float64,
            )

            vtk_file.write(
                f"VECTORS {name} float\n"
            )
            for x, y, z in vectors:
                vtk_file.write(
                    f"{x:.9g} {y:.9g} {z:.9g}\n"
                )

            vtk_file.write("\n")

        for name, scalars in scalar_fields.items():
            scalars = np.asarray(
                scalars,
                dtype=np.float64,
            ).reshape(-1)

            vtk_file.write(
                f"SCALARS {name} float 1\n"
            )
            vtk_file.write(
                "LOOKUP_TABLE default\n"
            )

            for value in scalars:
                vtk_file.write(f"{value:.9g}\n")

            vtk_file.write("\n")


def load_mesh_npz(
    npz_path: str | Path,
    keys: Sequence[str] = (
        "verts",
        "faces",
        "norms",
        "theta",
        "curvs",
        "gauss",
        "force",
    ),
) -> tuple[np.ndarray, ...]:
    """Load selected mesh arrays from a NumPy archive."""
    with np.load(npz_path, allow_pickle=True) as archive:
        missing_keys = [
            key for key in keys
            if key not in archive.files
        ]

        if missing_keys:
            raise KeyError(
                "Mesh archive is missing keys: "
                + ", ".join(missing_keys)
            )

        return tuple(
            np.asarray(archive[key])
            for key in keys
        )


# ---------------------------------------------------------------------
# Nearest-neighbor comparison
# ---------------------------------------------------------------------


def match_nearest_neighbor(
    vertices_a: Any,
    values_a: Any,
    vertices_b: Any,
    values_b: Any,
    *,
    max_dist: float | None = None,
    mutual: bool = False,
    workers: int = -1,
) -> dict[str, Any]:
    """Match values from dataset A to nearest vertices in dataset B."""
    vertices_a = np.asarray(
        vertices_a,
        dtype=float,
    )
    vertices_b = np.asarray(
        vertices_b,
        dtype=float,
    )
    values_a = np.asarray(
        values_a,
        dtype=float,
    ).reshape(-1)
    values_b = np.asarray(
        values_b,
        dtype=float,
    ).reshape(-1)

    if vertices_a.ndim != 2 or vertices_a.shape[1] != 3:
        raise ValueError(
            "vertices_a must have shape (N, 3), "
            f"but received {vertices_a.shape}."
        )

    if vertices_b.ndim != 2 or vertices_b.shape[1] != 3:
        raise ValueError(
            "vertices_b must have shape (M, 3), "
            f"but received {vertices_b.shape}."
        )

    if values_a.shape[0] != vertices_a.shape[0]:
        raise ValueError(
            "values_a and vertices_a must contain the same "
            "number of entries."
        )

    if values_b.shape[0] != vertices_b.shape[0]:
        raise ValueError(
            "values_b and vertices_b must contain the same "
            "number of entries."
        )

    tree_b = cKDTree(vertices_b)
    distances, indices_b = tree_b.query(
        vertices_a,
        k=1,
        workers=workers,
    )

    mask = np.ones(
        vertices_a.shape[0],
        dtype=bool,
    )

    if max_dist is not None:
        mask &= distances <= float(max_dist)

    if mutual:
        tree_a = cKDTree(vertices_a)
        _, indices_a = tree_a.query(
            vertices_b,
            k=1,
            workers=workers,
        )

        mask &= (
            indices_a[indices_b]
            == np.arange(vertices_a.shape[0])
        )

    values_a_kept = values_a[mask]
    values_b_kept = values_b[indices_b[mask]]

    if values_a_kept.shape[0] < 2:
        raise RuntimeError(
            "Too few matched points remained after filtering."
        )

    correlation = float(
        np.corrcoef(
            values_a_kept,
            values_b_kept,
        )[0, 1]
    )

    return {
        "indices_b": indices_b,
        "distances": distances,
        "mask": mask,
        "values_a": values_a_kept,
        "values_b": values_b_kept,
        "correlation": correlation,
    }


def match_nearest_and_plot(
    vertices_a: Any,
    values_a: Any,
    vertices_b: Any,
    values_b: Any,
    *,
    max_dist: float | None = None,
    mutual: bool = False,
    ax: Any | None = None,
    title: str = "A vs B (nearest-neighbor matched)",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    aspect: str = "auto",
) -> dict[str, Any]:
    """Match one dataset to another and create a correlation plot."""
    result = match_nearest_neighbor(
        vertices_a,
        values_a,
        vertices_b,
        values_b,
        max_dist=max_dist,
        mutual=mutual,
    )

    if ax is None:
        _, ax = plt.subplots(
            figsize=(4.2, 4.2)
        )

    matched_a = result["values_a"]
    matched_b = result["values_b"]
    correlation = result["correlation"]

    ax.scatter(
        matched_a,
        matched_b,
        s=8,
        alpha=0.6,
    )

    lower = float(
        np.nanmin(
            np.concatenate([matched_a, matched_b])
        )
    )
    upper = float(
        np.nanmax(
            np.concatenate([matched_a, matched_b])
        )
    )

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color="black",
        linewidth=1.5,
        zorder=3,
    )

    ax.set_xlabel("A values")
    ax.set_ylabel("B values (nearest-neighbor matched)")
    ax.set_title(
        f"{title}\n"
        f"Pearson r = {correlation:.3f} "
        f"(n={matched_a.shape[0]})"
    )

    ax.set_xlim(
        *(xlim if xlim is not None else (lower, upper))
    )
    ax.set_ylim(
        *(ylim if ylim is not None else (lower, upper))
    )
    ax.set_aspect(aspect)

    return result


def match_and_plot_multi_data(
    vertices_a: Any,
    values_a: Any,
    datasets_b: Sequence[tuple[Any, Any]],
    *,
    max_dist: float | None = None,
    mutual: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    aspect: str = "auto",
    box_aspect: float | None = None,
    labels: Sequence[str] | None = None,
    colors: Sequence[Any] | None = None,
    title: str | None = None,
    xlabel: str = "A values",
    ylabel: str = "B values (nearest-neighbor matched)",
    ax: Any | None = None,
    figsize: tuple[float, float] = (4.5, 4.5),
    marker_size: float = 2.0,
    alpha: float = 1.0,
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Match dataset A against multiple datasets and plot the comparisons."""
    n_datasets = len(datasets_b)

    if n_datasets == 0:
        raise ValueError(
            "datasets_b must contain at least one dataset."
        )

    if labels is None:
        labels = [
            f"B{index + 1}"
            for index in range(n_datasets)
        ]

    if colors is None:
        colors = [None] * n_datasets

    if len(labels) != n_datasets:
        raise ValueError(
            "labels must have the same length as datasets_b."
        )

    if len(colors) != n_datasets:
        raise ValueError(
            "colors must have the same length as datasets_b."
        )

    if ax is None:
        figure, ax = plt.subplots(
            figsize=figsize
        )
    else:
        figure = ax.figure

    results = {}
    plotted_values = []

    for (
        vertices_b,
        values_b,
    ), label, color in zip(
        datasets_b,
        labels,
        colors,
    ):
        result = match_nearest_neighbor(
            vertices_a,
            values_a,
            vertices_b,
            values_b,
            max_dist=max_dist,
            mutual=mutual,
        )
        results[label] = result

        matched_a = result["values_a"]
        matched_b = result["values_b"]

        ax.scatter(
            matched_a,
            matched_b,
            s=marker_size,
            alpha=alpha,
            color=color,
            label=(
                f"{label} "
                f"(r={result['correlation']:.3f})"
            ),
            zorder=2,
            edgecolors="none",
            rasterized=True,
        )

        plotted_values.extend(
            [matched_a, matched_b]
        )

    combined_values = np.concatenate(
        plotted_values
    )
    lower = float(np.nanmin(combined_values))
    upper = float(np.nanmax(combined_values))

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color="black",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_aspect(aspect)

    if box_aspect is not None:
        ax.set_box_aspect(box_aspect)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.45, -0.26),
        frameon=False,
        ncol=1,
        handletextpad=-0.5,
        columnspacing=0.0,
        labelspacing=0.0,
    )

    return figure, results


