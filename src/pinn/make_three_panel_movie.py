import sys
sys.path.append('../')

import os
import re
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import jax
import jax.numpy as jnp
from flax.training import checkpoints

from pinn.plot import _make_slice_points, _batch_apply
from pinn.model import PINN
from pinn.movie import build_phi_fn, predict_phase_slice


# ============================================================
# PARAMETERS
# ============================================================

lambda_1 = 100000
lambda_2 = 1
lambda_3 = 100000
lambda_4 = 0.01

step = 10000
hidden_dim = 128
phase = 2

# We explicitly sweep through z
axis = "z"

# Movie settings
fps = 20
dpi = 150

# Skip slices if desired.
# 1 = every slice
# 2 = every second slice
slice_step = 1

# If True:
#   z=0 -> z=max -> z=0
# If False:
#   z=0 -> z=max
back_and_forth = True

# Phase-field evaluation
batch = 4096
run_on_cpu = False

# Camera for the right panel
elev = 20
azim = -60

# Surface appearance
surface_alpha = 1.0

# ============================================================
# GOLGI
# ============================================================

data_id = "czii_27042022"
shape = "golgi_01"
expand_xy = None


# ============================================================
# PATHS
# ============================================================

checkpoint_dir = (
    f"../../outputs/logs/{data_id}/"
    f"{shape}_{hidden_dim}/"
    f"phase{phase}_{lambda_1}_{lambda_2}_{lambda_3}_{lambda_4}"
)

checkpoint_path = os.path.abspath(
    f"{checkpoint_dir}/checkpoint_{step}"
)

vtk_path = os.path.abspath(
    f"{checkpoint_dir}/verts_{step}.vtk"
)


if shape == "golgi_01":
    mrc_file_path = (
        f"../../data/experimental/vol/formatted/"
        f"{data_id}/{shape}.mrc"
    )

elif shape == "mito_01":
    mrc_file_path = (
        f"../../data/experimental/vol/raw/"
        f"{data_id}/{shape}.mrc"
    )

else:
    raise ValueError(f"Unknown shape: {shape}")


output_movie = f"../../outputs/movie/{shape}_three_panel_zscan.mp4"


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print(f"Loading checkpoint {checkpoint_path}")

checkpoint = checkpoints.restore_checkpoint(
    ckpt_dir=checkpoint_path,
    target=None,
)


# ============================================================
# BUILD PHASE-FIELD FUNCTION
# ============================================================

# Change this if your actual voxel_scale is not (1,1,1)
voxel_scale = (1.0, 1.0, 1.0)

phi_fn = build_phi_fn(
    checkpoint=checkpoint,
    hidden_dim=hidden_dim,
    voxel_scale=voxel_scale,
    run_on_cpu=run_on_cpu,
)


# ============================================================
# LOAD MRC
# ============================================================

print(f"Loading MRC {mrc_file_path}")

with mrcfile.open(mrc_file_path, permissive=True) as mrc:
    cryoET_data = np.asarray(mrc.data).copy()


print("MRC shape:", cryoET_data.shape)

# mrcfile convention is normally:
#
#     (z, y, x)
#
Nz, Ny, Nx = cryoET_data.shape


# ============================================================
# LEGACY ASCII VTK READER
# ============================================================

def read_legacy_vtk_polydata(filename):
    """
    Read POINTS and POLYGONS from a legacy ASCII VTK POLYDATA file.

    Returns
    -------
    vertices : (N, 3) ndarray
    faces : (M, 3) ndarray
    """

    with open(filename, "r") as f:
        lines = f.readlines()

    vertices = None
    faces = None

    # --------------------------
    # Read vertices
    # --------------------------

    for i, line in enumerate(lines):

        if line.startswith("POINTS"):
            parts = line.split()

            n_points = int(parts[1])

            numbers = []

            j = i + 1

            while len(numbers) < 3 * n_points:
                numbers.extend(
                    map(float, lines[j].split())
                )
                j += 1

            vertices = np.asarray(
                numbers[:3 * n_points],
                dtype=float,
            ).reshape(n_points, 3)

            break

    if vertices is None:
        raise RuntimeError(
            "Could not find POINTS section in VTK."
        )

    # --------------------------
    # Read polygons
    # --------------------------

    for i, line in enumerate(lines):

        if line.startswith("POLYGONS"):
            parts = line.split()

            n_polygons = int(parts[1])

            polygon_data = []

            j = i + 1

            while len(polygon_data) < n_polygons:
                vals = list(
                    map(int, lines[j].split())
                )

                if len(vals) == 0:
                    j += 1
                    continue

                n_vertices = vals[0]

                polygon_data.append(
                    vals[1:1 + n_vertices]
                )

                j += 1

            break

    if len(polygon_data) == 0:
        raise RuntimeError(
            "Could not find POLYGONS section in VTK."
        )

    # Your mesh is triangular.
    # Keep triangles directly.
    #
    # If some polygon contains >3 vertices,
    # triangulate it using a fan.

    triangles = []

    for polygon in polygon_data:

        if len(polygon) == 3:
            triangles.append(polygon)

        elif len(polygon) > 3:

            for k in range(1, len(polygon) - 1):
                triangles.append(
                    [
                        polygon[0],
                        polygon[k],
                        polygon[k + 1],
                    ]
                )

    faces = np.asarray(
        triangles,
        dtype=np.int64,
    )

    return vertices, faces


print(f"Loading VTK mesh {vtk_path}")

vertices, faces = read_legacy_vtk_polydata(
    vtk_path
)

print("Mesh vertices:", vertices.shape)
print("Mesh triangles:", faces.shape)


# ============================================================
# MESH EXTENT
# ============================================================

mesh_xmin, mesh_ymin, mesh_zmin = vertices.min(axis=0)
mesh_xmax, mesh_ymax, mesh_zmax = vertices.max(axis=0)

print()
print("Mesh bounds:")
print("x:", mesh_xmin, mesh_xmax)
print("y:", mesh_ymin, mesh_ymax)
print("z:", mesh_zmin, mesh_zmax)


# ============================================================
# TRUE TRIANGLE CLIPPING
# ============================================================

def interpolate_to_z_plane(p1, p2, z_cut):
    """
    Intersection of line segment p1--p2 with z=z_cut.
    """

    dz = p2[2] - p1[2]

    if abs(dz) < 1e-12:
        return p1.copy()

    t = (z_cut - p1[2]) / dz

    return p1 + t * (p2 - p1)


def clip_polygon_below_z(polygon, z_cut):
    """
    Clip one polygon against:

        z <= z_cut

    using Sutherland-Hodgman polygon clipping.
    """

    output = []

    n = len(polygon)

    for i in range(n):

        current = polygon[i]
        previous = polygon[i - 1]

        current_inside = current[2] <= z_cut
        previous_inside = previous[2] <= z_cut

        # entering
        if current_inside and not previous_inside:

            intersection = interpolate_to_z_plane(
                previous,
                current,
                z_cut,
            )

            output.append(intersection)
            output.append(current)

        # both inside
        elif current_inside and previous_inside:

            output.append(current)

        # leaving
        elif (
            not current_inside
            and previous_inside
        ):

            intersection = interpolate_to_z_plane(
                previous,
                current,
                z_cut,
            )

            output.append(intersection)

        # both outside -> nothing

    return output


def clipped_mesh_triangles(
    vertices,
    faces,
    z_cut,
):
    """
    Return renderable triangles representing the mesh
    clipped against z <= z_cut.

    The original mesh is not modified.
    """

    clipped_triangles = []

    for face in faces:

        triangle = [
            vertices[face[0]],
            vertices[face[1]],
            vertices[face[2]],
        ]

        polygon = clip_polygon_below_z(
            triangle,
            z_cut,
        )

        if len(polygon) < 3:
            continue

        # Triangulate polygon fan.
        # Clipped triangle can become triangle or quad.

        p0 = polygon[0]

        for i in range(1, len(polygon) - 1):

            clipped_triangles.append(
                [
                    p0,
                    polygon[i],
                    polygon[i + 1],
                ]
            )

    return clipped_triangles


# ============================================================
# PHASE-FIELD SLICES
# ============================================================

print()
print("Predicting phase-field slices...")


def evaluate_all_phase_z_slices(
    phi_fn,
    Nz,
    Nx,
    Ny,
    batch=4096,
    expand_xy=None,
    run_on_cpu=False,
):
    """
    Evaluate phase field on every z slice.

    NOTE:
    predict_phase_slice assumes one common grid_size.

    Your Golgi volume is currently expected to be cubic.
    If Nx != Ny != Nz, use a generalized evaluator instead.
    """

    if not (
        Nx == Ny == Nz
    ):
        raise ValueError(
            "Current predict_phase_slice() assumes a cubic grid. "
            f"MRC shape is {(Nz, Ny, Nx)}."
        )

    phase_volume = np.empty(
        (Nz, Ny, Nx),
        dtype=np.float32,
    )

    for k in range(Nz):

        print(
            f"\rPhase slice {k + 1}/{Nz}",
            end="",
            flush=True,
        )

        phase_slice = predict_phase_slice(
            phi_fn=phi_fn,
            grid_size=Nz,
            slice_index=k,
            axis="z",
            batch=batch,
            expand_xy=expand_xy,
            run_on_cpu=run_on_cpu,
        )

        # predict_phase_slice(axis="z")
        #
        # returns:
        #     shape = (x, y)
        #
        # while MRC slice is:
        #     shape = (y, x)
        #
        # therefore transpose.

        phase_volume[k] = phase_slice.T

    print()

    return phase_volume


phase_volume = evaluate_all_phase_z_slices(
    phi_fn=phi_fn,
    Nz=Nz,
    Nx=Nx,
    Ny=Ny,
    batch=batch,
    expand_xy=expand_xy,
    run_on_cpu=run_on_cpu,
)


# ============================================================
# FRAME SEQUENCE
# ============================================================

forward_indices = list(
    range(
        0,
        Nz,
        slice_step,
    )
)

# Guarantee final slice is included
if forward_indices[-1] != Nz - 1:
    forward_indices.append(Nz - 1)


if back_and_forth:

    backward_indices = forward_indices[-2:0:-1]

    frame_indices = (
        forward_indices
        + backward_indices
        + [0]
    )

else:

    frame_indices = forward_indices


print(
    "Total movie frames:",
    len(frame_indices),
)


# ============================================================
# INTENSITY LIMITS
# ============================================================

# Robust limits so a few extreme voxels do not dominate
mrc_vmin, mrc_vmax = np.percentile(
    cryoET_data,
    [1, 99],
)

# Phase field is expected around [-1, 1]
phase_absmax = max(
    abs(float(phase_volume.min())),
    abs(float(phase_volume.max())),
)

phase_vmin = -phase_absmax
phase_vmax = phase_absmax


# Custom grayscale from previous visualization
custom_gray = LinearSegmentedColormap.from_list(
    "custom_gray",
    ["#f0f0f0", "#111111"],
)


# ============================================================
# MAP MRC SLICE INDEX -> PHYSICAL VTK Z
# ============================================================

def slice_index_to_mesh_z(k):
    """
    Linear mapping:

        k = 0      -> mesh_zmin
        k = Nz - 1 -> mesh_zmax
    """

    fraction = k / (Nz - 1)

    return (
        mesh_zmin
        + fraction
        * (mesh_zmax - mesh_zmin)
    )


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(15, 5),
    constrained_layout=True,
)

gs = fig.add_gridspec(
    1,
    3,
    width_ratios=[1, 1, 1.2],
)


ax_mrc = fig.add_subplot(
    gs[0, 0]
)

ax_phase = fig.add_subplot(
    gs[0, 1]
)

ax_mesh = fig.add_subplot(
    gs[0, 2],
    projection="3d",
)


# ============================================================
# STATIC AXIS SETTINGS
# ============================================================

ax_mrc.set_title(
    "Tomogram"
)

ax_phase.set_title(
    "Phase field"
)

ax_mesh.set_title(
    "Reconstructed membrane"
)


# Raw MRC
mrc_image = ax_mrc.imshow(
    cryoET_data[0],
    cmap=custom_gray,
    vmin=mrc_vmin,
    vmax=mrc_vmax,
    origin="lower",
)

ax_mrc.set_axis_off()


# Phase field
phase_image = ax_phase.imshow(
    phase_volume[0],
    cmap="coolwarm",
    vmin=phase_vmin,
    vmax=phase_vmax,
    origin="lower",
)

ax_phase.set_axis_off()


# ============================================================
# 3D CAMERA + LIMITS
# ============================================================

ax_mesh.set_xlim(
    mesh_xmin,
    mesh_xmax,
)

ax_mesh.set_ylim(
    mesh_ymin,
    mesh_ymax,
)

ax_mesh.set_zlim(
    mesh_zmin,
    mesh_zmax,
)

ax_mesh.view_init(
    elev=elev,
    azim=azim,
)

# Equal-ish physical aspect
ax_mesh.set_box_aspect(
    (
        mesh_xmax - mesh_xmin,
        mesh_ymax - mesh_ymin,
        mesh_zmax - mesh_zmin,
    )
)

ax_mesh.set_axis_off()


# ============================================================
# CURRENT SLICE LABEL
# ============================================================

frame_text = fig.text(
    0.5,
    0.02,
    "",
    ha="center",
    va="bottom",
)


# ============================================================
# MOVIE WRITER
# ============================================================

writer = FFMpegWriter(
    fps=fps,
    metadata={
        "title": "Tomogram to membrane reconstruction",
    },
    bitrate=6000,
)


# ============================================================
# RENDER
# ============================================================

print()
print("Rendering movie...")


surface_collection = None


with writer.saving(
    fig,
    output_movie,
    dpi=dpi,
):

    for frame_number, k in enumerate(
        frame_indices
    ):

        print(
            f"\rFrame "
            f"{frame_number + 1}/"
            f"{len(frame_indices)} "
            f"(z slice {k})",
            end="",
            flush=True,
        )

        # ------------------------------------------
        # LEFT: MRC
        # ------------------------------------------

        mrc_image.set_data(
            cryoET_data[k]
        )

        # ------------------------------------------
        # MIDDLE: phase
        # ------------------------------------------

        phase_image.set_data(
            phase_volume[k]
        )

        # ------------------------------------------
        # RIGHT: progressive mesh
        # ------------------------------------------

        z_cut = slice_index_to_mesh_z(k)

        clipped_triangles = clipped_mesh_triangles(
            vertices=vertices,
            faces=faces,
            z_cut=z_cut,
        )

        # Remove previous surface
        if surface_collection is not None:
            surface_collection.remove()
            surface_collection = None

        if len(clipped_triangles) > 0:

            surface_collection = Poly3DCollection(
                clipped_triangles,
                alpha=surface_alpha,
            )

            # Let matplotlib choose default color.
            # You can change appearance later if desired.

            ax_mesh.add_collection3d(
                surface_collection
            )

        # Keep camera exactly fixed
        ax_mesh.set_xlim(
            mesh_xmin,
            mesh_xmax,
        )

        ax_mesh.set_ylim(
            mesh_ymin,
            mesh_ymax,
        )

        ax_mesh.set_zlim(
            mesh_zmin,
            mesh_zmax,
        )

        ax_mesh.view_init(
            elev=elev,
            azim=azim,
        )

        ax_mesh.set_box_aspect(
            (
                mesh_xmax - mesh_xmin,
                mesh_ymax - mesh_ymin,
                mesh_zmax - mesh_zmin,
            )
        )

        ax_mesh.set_axis_off()

        # ------------------------------------------
        # LABEL
        # ------------------------------------------

        frame_text.set_text(
            f"z slice {k + 1} / {Nz}"
        )

        writer.grab_frame()


print()
print()
print(
    f"Saved movie to:\n{output_movie}"
)