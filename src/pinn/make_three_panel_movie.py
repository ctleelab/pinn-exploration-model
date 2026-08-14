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

z_srt = 40
z_end = 81

# Movie settings
fps = 20
dpi = 150

# Skip slices if desired.
# 1 = every slice
# 2 = every second slice
slice_step = 4

# If True:
#   z=0 -> z=max -> z=0
# If False:
#   z=0 -> z=max
back_and_forth = True
# back_and_forth = False

# Phase-field evaluation
batch = 4096
run_on_cpu = False

# Camera for the right panel
elev = 0
azim = 0

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
# output_movie = f"../../outputs/movie/test.mp4"


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

def read_vtk_scalar(filename, scalar_name):
    """
    Read a POINT_DATA scalar field from legacy ASCII VTK.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith(f"SCALARS {scalar_name}"):

            # Next line is LOOKUP_TABLE default
            values = []
            j = i + 2

            while j < len(lines):
                line = lines[j].strip()

                # Stop at the next VTK data section
                if (
                    line.startswith("SCALARS")
                    or line.startswith("VECTORS")
                    or line.startswith("FIELD")
                ):
                    break

                if line:
                    values.extend(map(float, line.split()))

                j += 1

            return np.asarray(values)

    raise ValueError(f"{scalar_name} not found in {filename}")

def read_vtk_vectors(filename, vector_name):
    with open(filename, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith(f"VECTORS {vector_name}"):

            values = []
            j = i + 1

            while j < len(lines):
                line = lines[j].strip()

                if (
                    line.startswith("SCALARS")
                    or line.startswith("VECTORS")
                    or line.startswith("FIELD")
                ):
                    break

                if line:
                    values.extend(map(float, line.split()))

                j += 1

            return np.asarray(values).reshape(-1, 3)

    raise ValueError(f"{vector_name} not found")


print(f"Loading VTK mesh {vtk_path}")

vertices, faces = read_legacy_vtk_polydata(vtk_path)
# Undo axis swap from VTK generation
vertices = vertices[:, [0, 2, 1]]

print("Mesh vertices:", vertices.shape)
print("Mesh triangles:", faces.shape)

# curvature = read_vtk_scalar(vtk_path, "MeanCurvature")
curvature = read_vtk_scalar(vtk_path, "GaussianCurvature")
triangle_curvature = curvature[faces].mean(axis=1)

triangle_vertices = vertices[faces]
triangle_x = triangle_vertices[:, :, 0].mean(axis=1)
sort_order = np.argsort(triangle_x)
triangle_vertices_sorted = triangle_vertices[sort_order]
triangle_x_sorted = triangle_x[sort_order]
triangle_curvature_sorted = triangle_curvature[sort_order]

from matplotlib.colors import Normalize
import matplotlib.cm as cm

curv_max = np.percentile(np.abs(triangle_curvature),80)
curv_min = -curv_max
# curv_max = -0.005
# curv_min = 0.005
curv_norm = Normalize(vmin=curv_min, vmax=curv_max)
curv_cmap = cm.coolwarm

vertex_normals = read_vtk_vectors(vtk_path,"Normal")
triangle_normals = vertex_normals[faces].mean(axis=1)

triangle_normals /= np.linalg.norm(
    triangle_normals,
    axis=1,
    keepdims=True,
) + 1e-12
triangle_normals_sorted = triangle_normals[sort_order]


# ============================================================
# MESH EXTENT
# ============================================================

mesh_xmin, mesh_ymin, mesh_zmin = vertices.min(axis=0)
mesh_xmax, mesh_ymax, mesh_zmax = vertices.max(axis=0)

# Zoom the right panel
zoom = 0.8

cx = 0.5 * (mesh_xmin + mesh_xmax)
cy = 0.5 * (mesh_ymin + mesh_ymax)
cz = 0.5 * (mesh_zmin + mesh_zmax)

hx = 0.5 * (mesh_xmax - mesh_xmin) * zoom
hy = 0.5 * (mesh_ymax - mesh_ymin) * zoom
hz = 0.5 * (mesh_zmax - mesh_zmin) * zoom

view_xmin, view_xmax = cx - hx, cx + hx
view_ymin, view_ymax = cy - hy, cy + hy

# Move object downward in the panel:
# less margin below, more margin above
z_shift = 0.1 * (mesh_zmax - mesh_zmin)
view_zmin = cz - hz + z_shift
view_zmax = cz + hz + z_shift

print()
print("Mesh bounds:")
print("x:", mesh_xmin, mesh_xmax)
print("y:", mesh_ymin, mesh_ymax)
print("z:", mesh_zmin, mesh_zmax)


# ============================================================
# TRUE TRIANGLE CLIPPING
# ============================================================

def interpolate_to_x_plane(p1, p2, x_cut):
    dx = p2[0] - p1[0]

    if abs(dx) < 1e-12:
        return p1.copy()

    t = (x_cut - p1[0]) / dx
    return p1 + t * (p2 - p1)


def clip_polygon_below_x(polygon, x_cut):
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

        current_inside = current[0] <= x_cut
        previous_inside = previous[0] <= x_cut

        if current_inside and not previous_inside:
            intersection = interpolate_to_x_plane(
                previous, current, x_cut
            )
            output.append(intersection)
            output.append(current)

        elif current_inside and previous_inside:
            output.append(current)

        elif not current_inside and previous_inside:
            intersection = interpolate_to_x_plane(
                previous, current, x_cut
            )
            output.append(intersection)

    return output


def clipped_mesh_triangles(
    vertices,
    faces,
    x_cut,
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

        polygon = clip_polygon_below_x(
            triangle,
            x_cut,
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
            axis="x",
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

        # phase_volume[k] = phase_slice.T
        phase_volume[k] = phase_slice

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

def slice_index_to_mesh_x(k):
    """
    Linear mapping:

        k = 0      -> mesh_zmin
        k = Nz - 1 -> mesh_zmax
    """

    fraction = k / (Nz - 1)

    return (
        mesh_xmin
        + fraction * (mesh_xmax - mesh_xmin)
    )


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(12, 4),
    constrained_layout=True,
)

gs = fig.add_gridspec(
    1,
    3,
    width_ratios=[1, 1, 1],
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

# pos = ax_mesh.get_position()
# ax_mesh.set_position([
#     pos.x0 - 0.01,   # shift left
#     pos.y0,
#     pos.width,
#     pos.height,
# ])


# ============================================================
# STATIC AXIS SETTINGS
# ============================================================

# ax_mrc.set_title(
#     "Tomogram"
# )

# ax_phase.set_title(
#     "Phase Field"
# )

# ax_mesh.set_title(
#     "Membrane with Curvature"
# )


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

ax_mesh.set_xlim(view_xmin, view_xmax)
ax_mesh.set_ylim(view_ymin, view_ymax)
ax_mesh.set_zlim(view_zmin, view_zmax)

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


light_dir = np.array([
    -0.5,
    -0.5,
    1.0,
])

light_dir = light_dir / np.linalg.norm(light_dir)

# ambient = 0.45
# diffuse = 0.55

ambient = 0.3
diffuse = 0.7


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

        fraction = k / (Nz - 1)

        k_lm = int(round(
            z_srt
            + fraction * (z_end - z_srt)
        ))

        # ------------------------------------------
        # LEFT: MRC
        # ------------------------------------------

        mrc_image.set_data(
            cryoET_data[k_lm]
        )

        # ------------------------------------------
        # MIDDLE: phase
        # ------------------------------------------

        phase_image.set_data(
            phase_volume[k_lm]
        )

        # ------------------------------------------
        # RIGHT: progressive mesh
        # ------------------------------------------

        x_cut = slice_index_to_mesh_x(k)

        # clipped_triangles = clipped_mesh_triangles(
        #     vertices=vertices,
        #     faces=faces,
        #     x_cut=x_cut,
        # )

        n_visible = np.searchsorted(
            triangle_x_sorted,
            x_cut,
            side="right",
        )
        visible_triangles = triangle_vertices_sorted[:n_visible]
        visible_curvature = (triangle_curvature_sorted[:n_visible])
        facecolors = curv_cmap(curv_norm(visible_curvature))        

        # Lighting
        visible_normals = triangle_normals_sorted[:n_visible]
        lighting = visible_normals @ light_dir
        lighting = np.clip(lighting, 0.0, 1.0)   # Only illuminate surfaces facing the light
        lighting = ambient + diffuse * lighting  # Ambient + diffuse lighting
        facecolors[:, :3] *= lighting[:, None]
        facecolors[:, :3] = np.clip(facecolors[:, :3], 0.0, 1.0)

        # Remove previous surface
        if surface_collection is not None:
            surface_collection.remove()
            surface_collection = None

        # if len(clipped_triangles) > 0:
        if len(visible_triangles) > 0:

            # surface_collection = Poly3DCollection(
            #     clipped_triangles,
            #     alpha=surface_alpha,
            # )

            # surface_collection = Poly3DCollection(
            #     visible_triangles,
            #     facecolors=facecolors,
            #     edgecolors="none",
            #     alpha=surface_alpha,
            # )

            surface_collection = Poly3DCollection(
                visible_triangles,
                facecolors=facecolors,
                edgecolors="none",
                linewidths=0.0,
                alpha=1.0,
            )

            # Let matplotlib choose default color.
            # You can change appearance later if desired.
            ax_mesh.add_collection3d(surface_collection)

        # Keep camera exactly fixed
        ax_mesh.set_xlim(view_xmin, view_xmax)
        ax_mesh.set_ylim(view_ymin, view_ymax)
        ax_mesh.set_zlim(view_zmin, view_zmax)

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

        # frame_text.set_text(
        #     f"z slice {k + 1} / {Nz}"
        # )

        writer.grab_frame()


print()
print()
print(
    f"Saved movie to:\n{output_movie}"
)