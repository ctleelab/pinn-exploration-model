import numpy as np
from collections import deque
import jax
import jax.numpy as jnp
from pinn.model import PINN, grad_phi, hessian_phi
from pinn.normal_curves import adjugate_3x3_batched

def build_vertex_adjacency(n_verts, faces):
    """
    Build vertex adjacency list from triangular faces.

    Parameters
    ----------
    n_verts : int
        Number of vertices.
    faces : (M, 3) array
        Triangle indices.

    Returns
    -------
    adjacency : list[list[int]]
        adjacency[i] contains neighboring vertex indices of vertex i.
    """
    adjacency = [set() for _ in range(n_verts)]

    for tri in faces:
        i, j, k = map(int, tri)
        adjacency[i].update([j, k])
        adjacency[j].update([i, k])
        adjacency[k].update([i, j])

    return [list(neigh) for neigh in adjacency]


def connected_components_from_mask(mask, adjacency):
    """
    Find connected components among vertices where mask == True.

    Parameters
    ----------
    mask : (N,) bool array
        Boolean mask over vertices.
    adjacency : list[list[int]]
        Vertex adjacency list.

    Returns
    -------
    components : list[np.ndarray]
        List of connected components. Each component is an array of vertex indices.
    """
    visited = np.zeros(len(mask), dtype=bool)
    components = []

    for seed in np.where(mask)[0]:
        if visited[seed]:
            continue

        queue = deque([seed])
        visited[seed] = True
        comp = []

        while queue:
            v = queue.popleft()
            comp.append(v)

            for nb in adjacency[v]:
                if mask[nb] and not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

        components.append(np.asarray(comp, dtype=int))

    return components


def pca_frame(points):
    """
    Compute PCA frame of a point cloud.

    Parameters
    ----------
    points : (N, 3) array

    Returns
    -------
    center : (3,) array
        Centroid.
    eigvecs : (3, 3) array
        Principal axes as columns, ordered from largest variance to smallest.
    eigvals : (3,) array
        Eigenvalues ordered from largest to smallest.
    """
    center = points.mean(axis=0)
    X = points - center

    cov = (X.T @ X) / max(len(points) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    return center, eigvecs, eigvals


def extract_neck_band_geometry(
    npz_path,
    percentile=5.0,
    gaussian_threshold=None,
    min_component_size=30,
    return_all_components=False,
):
    """
    Extract a neck band from Gaussian curvature and estimate:
      1) neck-band vertices
      2) neck center
      3) neck plane and vertical axis

    The neck band is defined as a connected component of vertices having
    strongly negative Gaussian curvature.

    Parameters
    ----------
    npz_path : str
        Path to NPZ file containing at least:
          - 'verts' : (N, 3)
          - 'faces' : (M, 3)
          - 'gauss' : (N,)
    percentile : float, default=5.0
        If gaussian_threshold is None, use this percentile of gauss as threshold.
        Example: percentile=5 means keep vertices with gauss below the 5th percentile.
    gaussian_threshold : float or None, default=None
        Explicit threshold for Gaussian curvature. If given, overrides percentile.
    min_component_size : int, default=30
        Ignore connected components smaller than this size.
    return_all_components : bool, default=False
        If True, also return all candidate connected components.

    Returns
    -------
    result : dict
        Dictionary containing:
          - verts : (N, 3) array
          - faces : (M, 3) array
          - gauss : (N,) array
          - threshold : float
          - neck_mask_raw : (N,) bool array
          - neck_mask : (N,) bool array
          - neck_indices : (K,) int array
          - neck_points : (K, 3) array
          - neck_center : (3,) array
          - plane_normal : (3,) array
          - plane_basis_u : (3,) array
          - plane_basis_v : (3,) array
          - pca_eigvals : (3,) array

        Interpretation:
          - plane_normal: estimated neck-plane normal
          - plane_basis_u, plane_basis_v: orthonormal basis spanning neck plane
    """
    data = np.load(npz_path)
    verts = np.asarray(data["verts"], dtype=float)
    faces = np.asarray(data["faces"], dtype=int)
    gauss = np.asarray(data["gauss"], dtype=float).reshape(-1)

    if len(verts) != len(gauss):
        raise ValueError("verts and gauss must have the same length.")

    # ---- 1. Raw thresholding on Gaussian curvature ----
    if gaussian_threshold is None:
        threshold = np.percentile(gauss, percentile)
    else:
        threshold = float(gaussian_threshold)

    neck_mask_raw = gauss < threshold

    # ---- 2. Connected components on thresholded vertices ----
    adjacency = build_vertex_adjacency(len(verts), faces)
    components = connected_components_from_mask(neck_mask_raw, adjacency)

    # Filter by size
    components = [comp for comp in components if len(comp) >= min_component_size]

    if len(components) == 0:
        raise ValueError(
            "No connected component found in the thresholded Gaussian-curvature band. "
            "Try relaxing the threshold or lowering min_component_size."
        )

    # Choose the largest component as the neck band
    neck_indices = max(components, key=len)
    neck_mask = np.zeros(len(verts), dtype=bool)
    neck_mask[neck_indices] = True
    neck_points = verts[neck_indices]

    # ---- 3. Neck center and PCA frame ----
    neck_center, eigvecs, eigvals = pca_frame(neck_points)

    # For a ring-like band:
    #   eigvecs[:, 0], eigvecs[:, 1] span the neck plane
    #   eigvecs[:, 2] is the smallest-variance direction ~ plane normal
    plane_basis_u = eigvecs[:, 0]
    plane_basis_v = eigvecs[:, 1]
    plane_normal = eigvecs[:, 2]

    result = {
        "verts": verts,
        "faces": faces,
        "gauss": gauss,
        "threshold": threshold,
        "neck_mask_raw": neck_mask_raw,
        "neck_mask": neck_mask,
        "neck_indices": neck_indices,
        "neck_points": neck_points,
        "neck_center": neck_center,
        "plane_normal": plane_normal,
        "plane_basis_u": plane_basis_u,
        "plane_basis_v": plane_basis_v,
        "pca_eigvals": eigvals,
    }

    if return_all_components:
        result["components"] = components

    return result


import numpy as np
import jax.numpy as jnp
from skimage import measure


# def project_points_to_plane(points, center, basis_u, basis_v):
#     """
#     Project 3D points onto a 2D coordinate system on a plane.

#     Parameters
#     ----------
#     points : (N, 3) array
#     center : (3,) array
#     basis_u : (3,) array
#     basis_v : (3,) array

#     Returns
#     -------
#     st : (N, 2) array
#         Plane coordinates [s, t] of each point.
#     """
#     rel = points - center[None, :]
#     s = rel @ basis_u
#     t = rel @ basis_v
#     return np.column_stack([s, t])


def lift_plane_points_to_3d(st, center, basis_u, basis_v):
    """
    Convert plane coordinates (s, t) back to 3D points.

    Parameters
    ----------
    st : (N, 2) array
    center : (3,) array
    basis_u : (3,) array
    basis_v : (3,) array

    Returns
    -------
    pts3d : (N, 3) array
    """
    s = st[:, 0]
    t = st[:, 1]
    return center[None, :] + s[:, None] * basis_u[None, :] + t[:, None] * basis_v[None, :]


# def sample_phase_field_on_plane(
#     phi_batched,
#     center,
#     basis_u,
#     basis_v,
#     neck_points=None,
#     margin_ratio=0.3,
#     grid_size=256,
#     s_range=None,
#     t_range=None,
# ):
#     """
#     Sample phase field phi on a 2D grid embedded in the neck plane.

#     The plane is parameterized as:
#         x(s, t) = center + s * basis_u + t * basis_v

#     Parameters
#     ----------
#     phi_batched : callable
#         Function taking (N, 3) array and returning (N,) phase values.
#     center : (3,) array
#     basis_u : (3,) array
#     basis_v : (3,) array
#     neck_points : (K, 3) array or None
#         If s_range/t_range are not given, use these points to determine the window.
#     margin_ratio : float, default=0.3
#         Expand the plane window by this fraction.
#     grid_size : int, default=256
#         Number of grid points in each direction.
#     s_range : tuple or None
#         Explicit (smin, smax).
#     t_range : tuple or None
#         Explicit (tmin, tmax).

#     Returns
#     -------
#     result : dict
#         Contains:
#           - s_vals, t_vals
#           - S, T : meshgrid arrays
#           - phi_plane : (nt, ns) array
#           - center, basis_u, basis_v
#           - s_range, t_range
#     """
#     center = np.asarray(center, dtype=float)
#     basis_u = np.asarray(basis_u, dtype=float)
#     basis_v = np.asarray(basis_v, dtype=float)

#     if s_range is None or t_range is None:
#         if neck_points is None:
#             raise ValueError("Provide neck_points or explicit s_range/t_range.")

#         neck_st = project_points_to_plane(
#             np.asarray(neck_points, dtype=float), center, basis_u, basis_v
#         )
#         s0 = neck_st[:, 0]
#         t0 = neck_st[:, 1]

#         smin, smax = s0.min(), s0.max()
#         tmin, tmax = t0.min(), t0.max()

#         ds = smax - smin
#         dt = tmax - tmin

#         # avoid degenerate ranges
#         if ds == 0:
#             ds = 1e-3
#         if dt == 0:
#             dt = 1e-3

#         s_pad = margin_ratio * ds
#         t_pad = margin_ratio * dt

#         s_range = (smin - s_pad, smax + s_pad)
#         t_range = (tmin - t_pad, tmax + t_pad)
#     else:
#         s_range = tuple(map(float, s_range))
#         t_range = tuple(map(float, t_range))

#     s_vals = np.linspace(s_range[0], s_range[1], grid_size)
#     t_vals = np.linspace(t_range[0], t_range[1], grid_size)

#     # Note:
#     # S, T shape = (nt, ns) when indexing="xy"
#     S, T = np.meshgrid(s_vals, t_vals, indexing="xy")

#     pts3d = (
#         center[None, None, :]
#         + S[..., None] * basis_u[None, None, :]
#         + T[..., None] * basis_v[None, None, :]
#     )
#     pts3d_flat = pts3d.reshape(-1, 3)

#     phi_vals = np.asarray(phi_batched(jnp.asarray(pts3d_flat)))
#     phi_plane = phi_vals.reshape(S.shape)

#     return {
#         "s_vals": s_vals,
#         "t_vals": t_vals,
#         "S": S,
#         "T": T,
#         "phi_plane": phi_plane,
#         "center": center,
#         "basis_u": basis_u,
#         "basis_v": basis_v,
#         "s_range": s_range,
#         "t_range": t_range,
#     }


# def find_zero_contours_on_plane(phi_plane_data, level=0.0, min_points=20):
#     """
#     Extract zero contours from the sampled phase field on the neck plane.

#     Parameters
#     ----------
#     phi_plane_data : dict
#         Output of sample_phase_field_on_plane().
#     level : float, default=0.0
#         Contour level.
#     min_points : int, default=20
#         Ignore very short contours.

#     Returns
#     -------
#     contours_st : list of (N_i, 2) arrays
#         Each contour is in plane coordinates [s, t].
#     """
#     phi_plane = phi_plane_data["phi_plane"]
#     s_vals = phi_plane_data["s_vals"]
#     t_vals = phi_plane_data["t_vals"]

#     raw_contours = measure.find_contours(phi_plane, level=level)
#     contours_st = []

#     # skimage returns contour coordinates in (row, col) = (t_index, s_index)
#     for c in raw_contours:
#         if len(c) < min_points:
#             continue

#         row = c[:, 0]
#         col = c[:, 1]

#         # convert fractional indices -> physical plane coordinates
#         s = np.interp(col, np.arange(len(s_vals)), s_vals)
#         t = np.interp(row, np.arange(len(t_vals)), t_vals)

#         contours_st.append(np.column_stack([s, t]))

#     return contours_st


def is_closed_contour(contour_st, closure_tol=None):
    """
    Check whether a 2D contour is closed.

    Parameters
    ----------
    contour_st : (N, 2) array
    closure_tol : float or None
        If None, use 5% of contour bounding-box diagonal.

    Returns
    -------
    closed : bool
    """
    p0 = contour_st[0]
    p1 = contour_st[-1]

    if closure_tol is None:
        mins = contour_st.min(axis=0)
        maxs = contour_st.max(axis=0)
        diag = np.linalg.norm(maxs - mins)
        closure_tol = max(0.05 * diag, 1e-6)

    return np.linalg.norm(p1 - p0) < closure_tol


def polygon_area_2d(contour_st):
    """
    Signed area of a 2D closed polygon.
    """
    x = contour_st[:, 0]
    y = contour_st[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def select_neck_ring_contour(
    contours_st,
    neck_points_st=None,
    require_closed=True,
    min_area=1e-8,
):
    """
    Select the most likely neck-ring contour from candidate contours.

    Strategy:
      - prefer closed contours
      - prefer contours near the origin (0,0) in plane coordinates
      - prefer contours near projected neck-band points

    Parameters
    ----------
    contours_st : list of (N_i, 2) arrays
    neck_points_st : (K, 2) array or None
        Projected neck-band points.
    require_closed : bool, default=True
    min_area : float, default=1e-8
        Ignore tiny contours.

    Returns
    -------
    best_contour : (N, 2) array
    info : dict
        Selection info.
    """
    if len(contours_st) == 0:
        raise ValueError("No contours found.")

    candidates = []

    for i, c in enumerate(contours_st):
        closed = is_closed_contour(c)
        if require_closed and not closed:
            continue

        area = abs(polygon_area_2d(c))
        if area < min_area:
            continue

        centroid = c.mean(axis=0)
        dist_to_origin = np.linalg.norm(centroid)

        if neck_points_st is not None and len(neck_points_st) > 0:
            # mean minimum distance from contour points to neck-band points
            # simple O(NM), okay for moderate sizes
            dmat = np.linalg.norm(c[:, None, :] - neck_points_st[None, :, :], axis=2)
            overlap_score = dmat.min(axis=1).mean()
        else:
            overlap_score = 0.0

        score = dist_to_origin + overlap_score

        candidates.append({
            "index": i,
            "contour": c,
            "closed": closed,
            "area": area,
            "centroid": centroid,
            "dist_to_origin": dist_to_origin,
            "overlap_score": overlap_score,
            "score": score,
        })

    if len(candidates) == 0:
        raise ValueError("No valid contour remained after filtering.")

    best = min(candidates, key=lambda d: d["score"])
    return best["contour"], best


def extract_neck_ring_from_phase_field(
    checkpoint,
    neck_center,
    plane_basis_u,
    plane_basis_v,
    neck_points,
    half_length=1.0,
    margin_ratio=0.3,
    grid_size=256,
    level=0.0,
    min_contour_points=20,
    require_closed=True,
):
    """
    Main function:
      1. sample phase field on neck plane
      2. extract phi=0 contours
      3. select the neck ring
      4. map selected contour back to 3D

    Parameters
    ----------
    checkpoint : checkpoint data of phase field
    neck_center : (3,) array
    plane_basis_u : (3,) array
    plane_basis_v : (3,) array
    neck_points : (K,3) array
        Neck-band vertices used to define local search window and scoring.
    half_length : re-scaled half length of computational domain
    margin_ratio : float, default=0.3
    grid_size : int, default=256
    level : float, default=0.0
        Usually 0 for membrane surface.
    min_contour_points : int, default=20
    require_closed : bool, default=True

    Returns
    -------
    result : dict
        Contains:
          - ring_st : (N,2) selected contour in plane coordinates
          - ring_3d : (N,3) selected contour in 3D
          - contours_st : list of candidate contours
          - phi_plane_data : dict from sampling
          - selection_info : dict
          - neck_points_st : projected neck points
    """
    neck_center = np.asarray(neck_center, dtype=float)
    plane_basis_u = np.asarray(plane_basis_u, dtype=float)
    plane_basis_v = np.asarray(plane_basis_v, dtype=float)
    neck_points = np.asarray(neck_points, dtype=float)

    state = checkpoint["state"]
    params = state["params"]
    model = PINN()
    scale_factor = 1.0 / half_length
    voxel_scale = jnp.array([scale_factor, scale_factor, scale_factor], dtype=jnp.float32)
    @jax.jit
    def phi_batched(x):  # (N,3) -> (N,) or (N,1)
        x_scaled = x * voxel_scale[None, :]
        out = model.apply(params, x_scaled)
        return out.reshape(-1)


    phi_plane_data = sample_phase_field_on_plane(
        phi_batched=phi_batched,
        center=neck_center,
        basis_u=plane_basis_u,
        basis_v=plane_basis_v,
        # neck_points=neck_points,
        ref_points=neck_points,
        margin_ratio=margin_ratio,
        grid_size=grid_size,
    )

    phi_plane = phi_plane_data["phi_plane"]
    print(phi_plane.min(), phi_plane.max())

    contours_st = find_zero_contours_on_plane(
        phi_plane_data,
        level=level,
        min_points=min_contour_points,
    )

    neck_points_st = project_points_to_plane(
        neck_points, neck_center, plane_basis_u, plane_basis_v
    )

    ring_st, selection_info = select_neck_ring_contour(
        contours_st,
        neck_points_st=neck_points_st,
        require_closed=require_closed,
    )

    ring_3d = lift_plane_points_to_3d(
        ring_st, neck_center, plane_basis_u, plane_basis_v
    )

    return {
        "ring_st": ring_st,
        "ring_3d": ring_3d,
        "contours_st": contours_st,
        "phi_plane_data": phi_plane_data,
        "selection_info": selection_info,
        "neck_points_st": neck_points_st,
    }



def compute_ring_geometry_from_phase_field(
    checkpoint,
    ring_3d,
    half_length=1.0,
    normal_eps=1e-12,
    curvature_eps=1e-12,
):
    """
    Compute normals, mean curvature, and Gaussian curvature at points on a ring
    using derivatives of the phase field.

    Parameters
    ----------
    checkpoint : dict
        Checkpoint data containing checkpoint["state"]["params"].
    ring_3d : (N, 3) array
        3D coordinates of ring points.
    half_length : float, default=1.0
        Physical half-length used to scale coordinates before model evaluation.
        This matches the convention:
            x_scaled = x / half_length
    curvature_kind : {"H", "kappa"}, default="H"
        Controls the returned "curv" field:
          - "H"     : return mean curvature H
          - "kappa" : return kappa = 2H
    normal_eps : float, default=1e-12
        Small regularization for normal normalization.
    curvature_eps : float, default=1e-12
        Small regularization for curvature denominator.

    Returns
    -------
    result : dict
        Contains:
          - points   : (N, 3) array, copied from input
          - phi      : (N,) array
          - grads    : (N, 3) array
          - normals  : (N, 3) array
          - mean_curvature : (N,) array
          - kappa    : (N,) array   where kappa = 2H
          - gauss    : (N,) array
          - curv     : (N,) array   chosen by curvature_kind
          - gnorm    : (N,) array   |grad phi|
    """
    state = checkpoint["state"]
    params = state["params"]
    model = PINN()

    scale_factor = 1.0 / half_length
    voxel_scale = jnp.array([scale_factor, scale_factor, scale_factor], dtype=jnp.float32)

    @jax.jit
    def phi_fn_pts(x):
        x = jnp.asarray(x, dtype=jnp.float32)
        x_scaled = x * voxel_scale[None, :]
        out = model.apply(params, x_scaled)
        return out.reshape(-1)

    pts_j = jnp.asarray(ring_3d, dtype=jnp.float32)

    # ---- scalar field ----
    phi_vals = phi_fn_pts(pts_j)                                 # (N,)

    # ---- gradient / normal ----
    grads = grad_phi(phi_fn_pts, pts_j)                          # (N,3)
    gnorm = jnp.linalg.norm(grads, axis=1, keepdims=True)        # (N,1)

    normals_j = grads / (gnorm + normal_eps)                     # (N,3)

    # Match your sign convention
    normals = -np.asarray(normals_j)

    # ---- Hessian ----
    Hphi = hessian_phi(phi_fn_pts, pts_j)                        # (N,3,3)
    trH = jnp.trace(Hphi, axis1=1, axis2=2)                      # (N,)

    Hg = jnp.einsum("nij,nj->ni", Hphi, grads)                   # (N,3)
    gHg = jnp.einsum("ni,ni->n", grads, Hg)                      # (N,)

    G = jnp.squeeze(gnorm, axis=1)                               # (N,)
    Gsafe = jnp.sqrt(G * G + curvature_eps**2)                   # (N,)

    # kappa = div(n) = 2H
    kappa_j = trH / Gsafe - gHg / (Gsafe**3)                     # (N,)

    # ---- Gaussian curvature ----
    adjH = adjugate_3x3_batched(Hphi)                            # (N,3,3)
    num = jnp.einsum("ni,nij,nj->n", normals_j, adjH, normals_j) # (N,)
    K_j = num / (Gsafe**2)                                       # (N,)

    # ---- Output sign convention ----
    # Keep the same convention as your existing code
    mean_curvature_j = -0.5 * kappa_j

    result = {
        "points": np.asarray(ring_3d, dtype=float),
        "phi": np.asarray(phi_vals),
        "grads": np.asarray(grads),
        "normals": normals,
        "mean_curvature": np.asarray(mean_curvature_j),
        "kappa": np.asarray(kappa_j),
        "gauss": np.asarray(K_j),
        "gnorm": np.asarray(G),
    }

    return result

def save_ring_geom_to_vtk(
    ring_geom,
    filepath,
    include_closing_segment=True,
):
    """
    Save ring geometry and curvature data to a VTK PolyData file.

    Parameters
    ----------
    ring_geom : dict
        Output of compute_ring_geometry_from_phase_field()
    filepath : str
        Output .vtk file path
    include_closing_segment : bool, default=True
        Whether to close the ring (last point connects to first)
    """
    pts = np.asarray(ring_geom["points"], dtype=float)
    N = pts.shape[0]

    normals = np.asarray(ring_geom.get("normals", np.zeros_like(pts)))
    H = np.asarray(ring_geom.get("mean_curvature", np.zeros(N)))
    K = np.asarray(ring_geom.get("gauss", np.zeros(N)))
    kappa = np.asarray(ring_geom.get("kappa", np.zeros(N)))
    curv = np.asarray(ring_geom.get("curv", np.zeros(N)))

    # ---- connectivity (polyline) ----
    if include_closing_segment:
        line_indices = list(range(N)) + [0]
    else:
        line_indices = list(range(N))

    n_ids = len(line_indices)

    with open(filepath, "w") as f:
        # header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Ring geometry\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # points
        f.write(f"POINTS {N} float\n")
        for p in pts:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

        # polyline
        f.write(f"LINES 1 {n_ids + 1}\n")
        f.write(f"{n_ids} " + " ".join(map(str, line_indices)) + "\n")

        # point data
        f.write(f"POINT_DATA {N}\n")

        # scalars: mean curvature
        f.write("SCALARS mean_curvature float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in H:
            f.write(f"{val}\n")

        # scalars: Gaussian curvature
        f.write("SCALARS gaussian_curvature float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in K:
            f.write(f"{val}\n")

        # scalars: kappa
        f.write("SCALARS kappa float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in kappa:
            f.write(f"{val}\n")

        # scalars: curv (your chosen output)
        f.write("SCALARS curv float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in curv:
            f.write(f"{val}\n")

        # vectors: normals
        f.write("VECTORS normals float\n")
        for n in normals:
            f.write(f"{n[0]} {n[1]} {n[2]}\n")

    print(f"Saved VTK to: {filepath}")



import numpy as np
import matplotlib.pyplot as plt


def compute_ring_arclength(points, closed=True):
    """
    Compute cumulative arclength along an ordered 3D ring/polyline.

    Parameters
    ----------
    points : (N, 3) array
    closed : bool, default=True
        If True, include the closing segment in total length.

    Returns
    -------
    s : (N,) array
        Cumulative arclength starting from 0.
    total_length : float
        Total length of the ring.
    """
    points = np.asarray(points, dtype=float)

    diffs = np.diff(points, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seglen)])

    if closed:
        total_length = s[-1] + np.linalg.norm(points[0] - points[-1])
    else:
        total_length = s[-1]

    return s, total_length


def smooth_periodic(y, window=7, closed=True):
    """
    Moving-average smoothing for 1D data on a ring.
    """
    y = np.asarray(y, dtype=float)

    if window is None or window <= 1:
        return y.copy()

    window = int(window)
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=float) / window
    pad = window // 2

    if closed:
        ypad = np.concatenate([y[-pad:], y, y[:pad]])
    else:
        ypad = np.pad(y, (pad, pad), mode="edge")

    ys = np.convolve(ypad, kernel, mode="valid")
    return ys


def plot_ring_curvatures_multi(
    data_dict,
    closed=True,
    normalize_arclength=False,
    smooth_window=None,
    figsize=(8, 6),
    save_path=None,
):
    """
    Plot arc length vs mean/Gaussian curvature for multiple shapes.

    Parameters
    ----------
    data_dict : dict
        Mapping like:
            {
                "bud_07": np.load(...),
                "bud_09": np.load(...),
                ...
            }
        Each npz must contain:
            - "points"
            - "mean_curvature"
            - "gauss"
    closed : bool, default=True
    normalize_arclength : bool, default=False
        If True, x-axis is s / total_length in [0, 1].
    smooth_window : int or None, default=None
        Moving-average window.
    figsize : tuple, default=(8, 6)
    save_path : str or None
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    for shape, data in data_dict.items():
        points = np.asarray(data["points"], dtype=float)
        H = np.asarray(data["mean_curvature"], dtype=float)
        K = np.asarray(data["gauss"], dtype=float)

        s, total_length = compute_ring_arclength(points, closed=closed)

        if normalize_arclength:
            x = s / total_length if total_length > 0 else s
            xlabel = "Normalized arclength"
        else:
            x = s
            xlabel = "Arclength"

        H_plot = smooth_periodic(H, window=smooth_window, closed=closed)
        K_plot = smooth_periodic(K, window=smooth_window, closed=closed)

        axes[0].plot(x, H_plot, label=shape)
        axes[1].plot(x, K_plot, label=shape)

    axes[0].set_ylabel("Mean curvature H")
    axes[0].set_title("Mean curvature along ring")

    axes[1].set_ylabel("Gaussian curvature K")
    axes[1].set_title("Gaussian curvature along ring")
    axes[1].set_xlabel(xlabel)

    axes[0].legend(frameon=False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    return fig, axes



import numpy as np
import matplotlib.pyplot as plt


def summarize_ring_curvatures(data_dict):
    """
    Compute mean and std of mean/Gaussian curvature for each shape.

    Parameters
    ----------
    data_dict : dict
        Mapping:
            {
                "bud_07": np.load(...),
                "bud_09": np.load(...),
                ...
            }
        Each item must contain:
            - "mean_curvature"
            - "gauss"

    Returns
    -------
    summary : dict
        {
            shape: {
                "H_mean": ...,
                "H_std": ...,
                "K_mean": ...,
                "K_std": ...,
            },
            ...
        }
    """
    summary = {}

    for shape, data in data_dict.items():
        H = np.asarray(data["mean_curvature"], dtype=float)
        K = np.asarray(data["gauss"], dtype=float)

        summary[shape] = {
            "H_mean": np.mean(H),
            "H_std": np.std(H),
            "K_mean": np.mean(K),
            "K_std": np.std(K),
        }

    return summary


def plot_ring_curvature_summary(
    data_dict,
    shape_order=None,
    figsize=(8, 5),
    save_path=None,
):
    """
    Plot mean ± std of mean/Gaussian curvature for multiple shapes.

    Parameters
    ----------
    data_dict : dict
        Mapping shape -> npz data
    shape_order : list[str] or None
        Order of shapes on x-axis
    figsize : tuple
    save_path : str or None
    """
    summary = summarize_ring_curvatures(data_dict)

    if shape_order is None:
        shape_order = list(summary.keys())

    x = np.arange(len(shape_order))

    H_mean = [summary[s]["H_mean"] for s in shape_order]
    H_std  = [summary[s]["H_std"]  for s in shape_order]
    K_mean = [summary[s]["K_mean"] for s in shape_order]
    K_std  = [summary[s]["K_std"]  for s in shape_order]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].errorbar(x, H_mean, yerr=H_std, fmt='o', capsize=4)
    axes[0].set_ylabel("Mean curvature H")
    axes[0].set_title("Mean ± std of mean curvature")

    axes[1].errorbar(x, K_mean, yerr=K_std, fmt='o', capsize=4)
    axes[1].set_ylabel("Gaussian curvature K")
    axes[1].set_title("Mean ± std of Gaussian curvature")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(shape_order, rotation=45, ha="right")
    axes[1].set_xlabel("Shape")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    return fig, axes, summary
















import numpy as np
import jax
import jax.numpy as jnp
from skimage import measure


# =========================
# Basic plane/contour utils
# =========================

def project_points_to_plane(points, center, basis_u, basis_v):
    rel = np.asarray(points, dtype=float) - np.asarray(center, dtype=float)[None, :]
    s = rel @ np.asarray(basis_u, dtype=float)
    t = rel @ np.asarray(basis_v, dtype=float)
    return np.column_stack([s, t])


def lift_plane_points_to_3d(st, center, basis_u, basis_v):
    st = np.asarray(st, dtype=float)
    center = np.asarray(center, dtype=float)
    basis_u = np.asarray(basis_u, dtype=float)
    basis_v = np.asarray(basis_v, dtype=float)
    return center[None, :] + st[:, 0, None] * basis_u[None, :] + st[:, 1, None] * basis_v[None, :]


def compute_polyline_arclength(points, closed=True):
    points = np.asarray(points, dtype=float)
    diffs = np.diff(points, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1] + (np.linalg.norm(points[0] - points[-1]) if closed else 0.0)
    return s, total


def polygon_area_2d(contour_st):
    x = contour_st[:, 0]
    y = contour_st[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def polygon_perimeter_2d(contour_st, closed=True):
    pts = np.asarray(contour_st, dtype=float)
    diffs = np.diff(pts, axis=0)
    perim = np.linalg.norm(diffs, axis=1).sum()
    if closed:
        perim += np.linalg.norm(pts[0] - pts[-1])
    return perim


def is_closed_contour(contour_st, closure_tol=None):
    contour_st = np.asarray(contour_st, dtype=float)
    if len(contour_st) < 3:
        return False
    if closure_tol is None:
        mins = contour_st.min(axis=0)
        maxs = contour_st.max(axis=0)
        diag = np.linalg.norm(maxs - mins)
        closure_tol = max(0.05 * diag, 1e-8)
    return np.linalg.norm(contour_st[0] - contour_st[-1]) < closure_tol


def fit_circle_2d(points_st):
    """
    Algebraic least-squares circle fit:
        (x-a)^2 + (y-b)^2 = r^2
    """
    pts = np.asarray(points_st, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]

    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)

    a, b0, c = sol
    r = np.sqrt(max(c + a*a + b0*b0, 0.0))

    center = np.array([a, b0], dtype=float)
    radial = np.linalg.norm(pts - center[None, :], axis=1)
    residual_rms = np.sqrt(np.mean((radial - r)**2))

    return {
        "center_st": center,
        "radius": float(r),
        "residual_rms": float(residual_rms),
        "radial_values": radial,
    }


def resample_closed_curve_2d(contour_st, n_samples=200):
    contour_st = np.asarray(contour_st, dtype=float)

    # ensure explicitly closed for resampling
    if np.linalg.norm(contour_st[0] - contour_st[-1]) > 1e-12:
        pts = np.vstack([contour_st, contour_st[0]])
    else:
        pts = contour_st.copy()

    diffs = np.diff(pts, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = s[-1]

    if total <= 0:
        return contour_st.copy()

    s_new = np.linspace(0.0, total, n_samples + 1)[:-1]
    x_new = np.interp(s_new, s, pts[:, 0])
    y_new = np.interp(s_new, s, pts[:, 1])
    return np.column_stack([x_new, y_new])


# =========================
# Phase-field slicing
# =========================

def sample_phase_field_on_plane(
    phi_batched,
    center,
    basis_u,
    basis_v,
    ref_points=None,
    margin_ratio=0.5,
    grid_size=256,
    s_range=None,
    t_range=None,
):
    center = np.asarray(center, dtype=float)
    basis_u = np.asarray(basis_u, dtype=float)
    basis_v = np.asarray(basis_v, dtype=float)

    if s_range is None or t_range is None:
        if ref_points is None:
            raise ValueError("Provide ref_points or explicit s_range/t_range.")
        st = project_points_to_plane(ref_points, center, basis_u, basis_v)
        s0, t0 = st[:, 0], st[:, 1]
        smin, smax = s0.min(), s0.max()
        tmin, tmax = t0.min(), t0.max()
        ds = max(smax - smin, 1e-3)
        dt = max(tmax - tmin, 1e-3)
        s_pad = margin_ratio * ds
        t_pad = margin_ratio * dt
        s_range = (smin - s_pad, smax + s_pad)
        t_range = (tmin - t_pad, tmax + t_pad)

    s_vals = np.linspace(s_range[0], s_range[1], grid_size)
    t_vals = np.linspace(t_range[0], t_range[1], grid_size)
    S, T = np.meshgrid(s_vals, t_vals, indexing="xy")

    pts3d = (
        center[None, None, :]
        + S[..., None] * basis_u[None, None, :]
        + T[..., None] * basis_v[None, None, :]
    )

    vals = np.asarray(phi_batched(jnp.asarray(pts3d.reshape(-1, 3), dtype=jnp.float32)))
    phi_plane = vals.reshape(S.shape)

    return {
        "center": center,
        "basis_u": basis_u,
        "basis_v": basis_v,
        "s_vals": s_vals,
        "t_vals": t_vals,
        "S": S,
        "T": T,
        "phi_plane": phi_plane,
        "s_range": s_range,
        "t_range": t_range,
    }


def find_zero_contours_on_plane(phi_plane_data, level=0.0, min_points=20):
    phi_plane = phi_plane_data["phi_plane"]
    s_vals = phi_plane_data["s_vals"]
    t_vals = phi_plane_data["t_vals"]

    phi_min = float(phi_plane.min())
    phi_max = float(phi_plane.max())
    if not (phi_min <= level <= phi_max):
        return []

    raw = measure.find_contours(phi_plane, level=level)
    contours = []

    for c in raw:
        if len(c) < min_points:
            continue
        row = c[:, 0]
        col = c[:, 1]
        s = np.interp(col, np.arange(len(s_vals)), s_vals)
        t = np.interp(row, np.arange(len(t_vals)), t_vals)
        contours.append(np.column_stack([s, t]))

    return contours


def select_contour_near_reference(
    contours_st,
    ref_points_st=None,
    require_closed=True,
    min_area=1e-8,
):
    if len(contours_st) == 0:
        return None, None

    candidates = []
    for i, c in enumerate(contours_st):
        closed = is_closed_contour(c)
        if require_closed and not closed:
            continue

        area = abs(polygon_area_2d(c))
        if area < min_area:
            continue

        centroid = c.mean(axis=0)
        dist_to_origin = np.linalg.norm(centroid)

        if ref_points_st is not None and len(ref_points_st) > 0:
            dmat = np.linalg.norm(c[:, None, :] - ref_points_st[None, :, :], axis=2)
            ref_score = dmat.min(axis=1).mean()
        else:
            ref_score = 0.0

        score = ref_score + 0.25 * dist_to_origin
        candidates.append({
            "index": i,
            "contour": c,
            "closed": closed,
            "area": area,
            "centroid_st": centroid,
            "ref_score": ref_score,
            "score": score,
        })

    if not candidates:
        return None, None

    best = min(candidates, key=lambda d: d["score"])
    return best["contour"], best


# =========================
# Phase-field differential geometry
# Assumes these exist in your environment:
#   PINN, grad_phi, hessian_phi, adjugate_3x3_batched
# =========================

def make_phi_functions_from_checkpoint(checkpoint, half_length=1.0):
    state = checkpoint["state"]
    params = state["params"]
    model = PINN()

    scale_factor = 1.0 / half_length
    voxel_scale = jnp.array([scale_factor, scale_factor, scale_factor], dtype=jnp.float32)

    @jax.jit
    def phi_fn_pts(x):
        x = jnp.asarray(x, dtype=jnp.float32)
        x_scaled = x * voxel_scale[None, :]
        out = model.apply(params, x_scaled)
        return out.reshape(-1)

    return phi_fn_pts


def compute_contour_geometry_from_phase_field(
    checkpoint,
    points_3d,
    half_length=1.0,
    curvature_kind="H",
    normal_eps=1e-12,
    curvature_eps=1e-12,
):
    phi_fn_pts = make_phi_functions_from_checkpoint(checkpoint, half_length=half_length)
    pts_j = jnp.asarray(points_3d, dtype=jnp.float32)

    phi_vals = phi_fn_pts(pts_j)
    grads = grad_phi(phi_fn_pts, pts_j)
    gnorm = jnp.linalg.norm(grads, axis=1, keepdims=True)

    normals_j = grads / (gnorm + normal_eps)
    normals = -np.asarray(normals_j)

    Hphi = hessian_phi(phi_fn_pts, pts_j)
    trH = jnp.trace(Hphi, axis1=1, axis2=2)

    Hg = jnp.einsum("nij,nj->ni", Hphi, grads)
    gHg = jnp.einsum("ni,ni->n", grads, Hg)

    G = jnp.squeeze(gnorm, axis=1)
    Gsafe = jnp.sqrt(G * G + curvature_eps**2)

    kappa_j = trH / Gsafe - gHg / (Gsafe**3)

    adjH = adjugate_3x3_batched(Hphi)
    num = jnp.einsum("ni,nij,nj->n", normals_j, adjH, normals_j)
    K_j = num / (Gsafe**2)

    mean_curvature_j = -0.5 * kappa_j

    if curvature_kind.lower() == "h":
        curv_j = mean_curvature_j
    elif curvature_kind.lower() == "kappa":
        curv_j = -kappa_j
    else:
        raise ValueError("curvature_kind must be 'H' or 'kappa'")

    return {
        "points": np.asarray(points_3d, dtype=float),
        "phi": np.asarray(phi_vals),
        "grads": np.asarray(grads),
        "normals": normals,
        "mean_curvature": np.asarray(mean_curvature_j),
        "kappa": np.asarray(kappa_j),
        "gauss": np.asarray(K_j),
        "curv": np.asarray(curv_j),
        "gnorm": np.asarray(G),
    }


# =========================
# Slice summary metrics
# =========================

def summarize_contour_slice(
    contour_st,
    contour_3d,
    slice_center,
    axis_direction,
    basis_u,
    basis_v,
    contour_geom=None,
):
    contour_st = np.asarray(contour_st, dtype=float)
    contour_3d = np.asarray(contour_3d, dtype=float)
    slice_center = np.asarray(slice_center, dtype=float)
    axis_direction = np.asarray(axis_direction, dtype=float)
    axis_direction = axis_direction / np.linalg.norm(axis_direction)

    contour_centroid_st = contour_st.mean(axis=0)
    contour_centroid_3d = contour_3d.mean(axis=0)

    circle_fit = fit_circle_2d(contour_st)
    area = abs(polygon_area_2d(contour_st))
    perimeter = polygon_perimeter_2d(contour_st, closed=True)

    # axis point closest to contour centroid
    dz = np.dot(contour_centroid_3d - slice_center, axis_direction)
    axis_foot = slice_center + dz * axis_direction
    center_offset_from_axis = np.linalg.norm(contour_centroid_3d - axis_foot)

    eq_radius_area = np.sqrt(area / np.pi) if area > 0 else np.nan
    eq_radius_perimeter = perimeter / (2 * np.pi) if perimeter > 0 else np.nan
    circularity = (4 * np.pi * area / (perimeter**2)) if perimeter > 0 else np.nan

    out = {
        "contour_centroid_st": contour_centroid_st,
        "contour_centroid_3d": contour_centroid_3d,
        "fitted_circle_center_st": circle_fit["center_st"],
        "fitted_radius": circle_fit["radius"],
        "circle_fit_residual_rms": circle_fit["residual_rms"],
        "area": area,
        "perimeter": perimeter,
        "equivalent_radius_area": eq_radius_area,
        "equivalent_radius_perimeter": eq_radius_perimeter,
        "circularity": circularity,
        "axis_foot_3d": axis_foot,
        "center_offset_from_axis": center_offset_from_axis,
        "slice_center_3d": slice_center,
        "axis_direction": axis_direction,
        "basis_u": np.asarray(basis_u, dtype=float),
        "basis_v": np.asarray(basis_v, dtype=float),
    }

    if contour_geom is not None:
        H = np.asarray(contour_geom["mean_curvature"], dtype=float)
        K = np.asarray(contour_geom["gauss"], dtype=float)
        out["H_mean"] = float(np.mean(H))
        out["H_std"] = float(np.std(H))
        out["K_mean"] = float(np.mean(K))
        out["K_std"] = float(np.std(K))

    return out


# =========================
# Main stack extraction
# =========================

def extract_contour_stack_along_axis(
    checkpoint,
    neck_center,
    vertical_axis,
    plane_basis_u,
    plane_basis_v,
    ref_points,
    z_offsets,
    half_length=1.0,
    margin_ratio=1.0,
    grid_size=256,
    level=0.0,
    min_contour_points=20,
    require_closed=True,
    contour_resample_n=200,
    curvature_kind="H",
    normal_eps=1e-12,
    curvature_eps=1e-12,
    discard_negative_offaxis=True,
    offaxis_radius_factor=1.0,
):
    """
    Extract contour datasets for a stack of planes parallel to the neck plane.

    Parameters
    ----------
    checkpoint : dict
        Phase-field checkpoint.
    neck_center : (3,) array
        Reference neck center.
    vertical_axis : (3,) array
        Axis normal to the neck plane.
    plane_basis_u, plane_basis_v : (3,) arrays
        Basis spanning the neck plane.
    ref_points : (K,3) array
        Reference points used to define plane window and contour selection.
        Usually neck points or ring_3d from the neck slice.
    z_offsets : array-like
        Offsets along vertical_axis, in the same units as the geometry.
    half_length : float
    margin_ratio : float
    grid_size : int
    level : float
    min_contour_points : int
    require_closed : bool
    contour_resample_n : int
        Number of points in each contour after resampling.
    curvature_kind : str
    normal_eps, curvature_eps : float

    Returns
    -------
    stack : dict
        {
            "z_offsets": ...,
            "slices": [slice_result_or_None, ...]
        }

    Each valid slice_result contains:
        - z_offset
        - plane_center
        - contour_st
        - contour_3d
        - contour_geom
        - slice_summary
        - phi_plane_data
        - selection_info
    """
    neck_center = np.asarray(neck_center, dtype=float)
    axis = np.asarray(vertical_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    u = np.asarray(plane_basis_u, dtype=float)
    v = np.asarray(plane_basis_v, dtype=float)
    ref_points = np.asarray(ref_points, dtype=float)
    z_offsets = np.asarray(z_offsets, dtype=float)

    phi_fn_pts = make_phi_functions_from_checkpoint(checkpoint, half_length=half_length)

    # keep the same plane window for all z slices based on reference points projected
    ref_st_at_neck = project_points_to_plane(ref_points, neck_center, u, v)
    s0 = ref_st_at_neck[:, 0]
    t0 = ref_st_at_neck[:, 1]
    smin, smax = s0.min(), s0.max()
    tmin, tmax = t0.min(), t0.max()
    ds = max(smax - smin, 1e-3)
    dt = max(tmax - tmin, 1e-3)
    s_range = (smin - margin_ratio * ds, smax + margin_ratio * ds)
    t_range = (tmin - margin_ratio * dt, tmax + margin_ratio * dt)

    slices = []

    prev_contour_st = ref_st_at_neck if len(ref_st_at_neck) > 0 else None

    for z in z_offsets:
        plane_center = neck_center + z * axis

        phi_plane_data = sample_phase_field_on_plane(
            phi_batched=phi_fn_pts,
            center=plane_center,
            basis_u=u,
            basis_v=v,
            s_range=s_range,
            t_range=t_range,
            grid_size=grid_size,
        )

        contours_st = find_zero_contours_on_plane(
            phi_plane_data,
            level=level,
            min_points=min_contour_points,
        )

        # use previous slice contour if available; otherwise use reference projection
        ref_for_selection = prev_contour_st if prev_contour_st is not None else ref_st_at_neck

        contour_st, selection_info = select_contour_near_reference(
            contours_st,
            ref_points_st=ref_for_selection,
            require_closed=require_closed,
        )

        if contour_st is None:
            slices.append(None)
            prev_contour_st = None
            continue

        contour_st = resample_closed_curve_2d(contour_st, n_samples=contour_resample_n)
        contour_3d = lift_plane_points_to_3d(contour_st, plane_center, u, v)

        contour_geom = compute_contour_geometry_from_phase_field(
            checkpoint=checkpoint,
            points_3d=contour_3d,
            half_length=half_length,
            curvature_kind=curvature_kind,
            normal_eps=normal_eps,
            curvature_eps=curvature_eps,
        )

        slice_summary = summarize_contour_slice(
            contour_st=contour_st,
            contour_3d=contour_3d,
            slice_center=plane_center,
            axis_direction=axis,
            basis_u=u,
            basis_v=v,
            contour_geom=contour_geom,
        )

        # Optional discard rule:
        # discard mother-side slices whose contour center is too far from axis
        if discard_negative_offaxis:
            if (
                z < 0
                and slice_summary["center_offset_from_axis"]
                > offaxis_radius_factor * slice_summary["fitted_radius"]
            ):
                slices.append(None)
                prev_contour_st = None
                continue

        slice_result = {
            "z_offset": float(z),
            "plane_center": plane_center,
            "contour_st": contour_st,
            "contour_3d": contour_3d,
            "contour_geom": contour_geom,
            "slice_summary": slice_summary,
            "phi_plane_data": phi_plane_data,
            "selection_info": selection_info,
            "candidate_contours_st": contours_st,
        }
        slices.append(slice_result)
        prev_contour_st = contour_st

    return {
        "z_offsets": z_offsets,
        "neck_center": neck_center,
        "vertical_axis": axis,
        "basis_u": u,
        "basis_v": v,
        "slices": slices,
    }



def save_contour_stack_to_vtk(
    stack,
    filepath,
    close_each_contour=True,
):
    """
    Save contour stack to a VTK PolyData file.

    Parameters
    ----------
    stack : dict
        Output of extract_contour_stack_along_axis().
    filepath : str
        Output .vtk file path.
    close_each_contour : bool, default=True
        If True, each contour polyline closes back to its first point.
    """
    valid_slices = [sl for sl in stack["slices"] if sl is not None]
    if len(valid_slices) == 0:
        raise ValueError("No valid slices found in stack.")

    # -------------------------
    # Collect all points / data
    # -------------------------
    all_points = []

    # point data
    all_z = []
    all_H = []
    all_K = []
    all_kappa = []
    all_curv = []
    all_phi = []
    all_gnorm = []
    all_normals = []

    # repeated slice-summary data per point
    all_fitted_radius = []
    all_center_offset = []
    all_circularity = []
    all_circle_fit_residual = []
    all_area = []
    all_perimeter = []
    all_H_mean = []
    all_H_std = []
    all_K_mean = []
    all_K_std = []

    lines = []
    point_offset = 0

    for sl in valid_slices:
        pts = np.asarray(sl["contour_3d"], dtype=float)
        geom = sl["contour_geom"]
        summary = sl["slice_summary"]

        n = len(pts)
        if n == 0:
            continue

        all_points.append(pts)

        # pointwise data
        all_z.append(np.full(n, sl["z_offset"], dtype=float))
        all_H.append(np.asarray(geom["mean_curvature"], dtype=float))
        all_K.append(np.asarray(geom["gauss"], dtype=float))
        all_kappa.append(np.asarray(geom["kappa"], dtype=float))
        all_curv.append(np.asarray(geom["curv"], dtype=float))
        all_phi.append(np.asarray(geom["phi"], dtype=float))
        all_gnorm.append(np.asarray(geom["gnorm"], dtype=float))
        all_normals.append(np.asarray(geom["normals"], dtype=float))

        # slice summary repeated per point
        all_fitted_radius.append(np.full(n, summary["fitted_radius"], dtype=float))
        all_center_offset.append(np.full(n, summary["center_offset_from_axis"], dtype=float))
        all_circularity.append(np.full(n, summary["circularity"], dtype=float))
        all_circle_fit_residual.append(np.full(n, summary["circle_fit_residual_rms"], dtype=float))
        all_area.append(np.full(n, summary["area"], dtype=float))
        all_perimeter.append(np.full(n, summary["perimeter"], dtype=float))
        all_H_mean.append(np.full(n, summary.get("H_mean", np.nan), dtype=float))
        all_H_std.append(np.full(n, summary.get("H_std", np.nan), dtype=float))
        all_K_mean.append(np.full(n, summary.get("K_mean", np.nan), dtype=float))
        all_K_std.append(np.full(n, summary.get("K_std", np.nan), dtype=float))

        # connectivity for one polyline
        if close_each_contour:
            line_ids = list(range(point_offset, point_offset + n)) + [point_offset]
        else:
            line_ids = list(range(point_offset, point_offset + n))

        lines.append(line_ids)
        point_offset += n

    # concatenate all
    points = np.vstack(all_points)
    z_offset = np.concatenate(all_z)
    H = np.concatenate(all_H)
    K = np.concatenate(all_K)
    kappa = np.concatenate(all_kappa)
    curv = np.concatenate(all_curv)
    phi = np.concatenate(all_phi)
    gnorm = np.concatenate(all_gnorm)
    normals = np.vstack(all_normals)

    fitted_radius = np.concatenate(all_fitted_radius)
    center_offset = np.concatenate(all_center_offset)
    circularity = np.concatenate(all_circularity)
    circle_fit_residual = np.concatenate(all_circle_fit_residual)
    area = np.concatenate(all_area)
    perimeter = np.concatenate(all_perimeter)
    H_mean = np.concatenate(all_H_mean)
    H_std = np.concatenate(all_H_std)
    K_mean = np.concatenate(all_K_mean)
    K_std = np.concatenate(all_K_std)

    n_points = len(points)
    n_lines = len(lines)
    line_size_total = sum(len(ids) + 1 for ids in lines)

    # -------------------------
    # Write VTK legacy POLYDATA
    # -------------------------
    with open(filepath, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Contour stack\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # points
        f.write(f"POINTS {n_points} float\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

        # lines
        f.write(f"LINES {n_lines} {line_size_total}\n")
        for ids in lines:
            f.write(str(len(ids)) + " " + " ".join(map(str, ids)) + "\n")

        # point data
        f.write(f"POINT_DATA {n_points}\n")

        def write_scalar(name, arr):
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for val in arr:
                f.write(f"{float(val)}\n")

        def write_vector(name, arr):
            f.write(f"VECTORS {name} float\n")
            for v in arr:
                f.write(f"{float(v[0])} {float(v[1])} {float(v[2])}\n")

        # pointwise fields
        write_scalar("z_offset", z_offset)
        write_scalar("mean_curvature", H)
        write_scalar("gaussian_curvature", K)
        write_scalar("kappa", kappa)
        write_scalar("curv", curv)
        write_scalar("phi", phi)
        write_scalar("gnorm", gnorm)
        write_vector("normals", normals)

        # slice summary fields repeated on each point
        write_scalar("fitted_radius", fitted_radius)
        write_scalar("center_offset_from_axis", center_offset)
        write_scalar("circularity", circularity)
        write_scalar("circle_fit_residual_rms", circle_fit_residual)
        write_scalar("area", area)
        write_scalar("perimeter", perimeter)
        write_scalar("H_mean_slice", H_mean)
        write_scalar("H_std_slice", H_std)
        write_scalar("K_mean_slice", K_mean)
        write_scalar("K_std_slice", K_std)

    print(f"Saved contour stack to: {filepath}")



import numpy as np
import matplotlib.pyplot as plt


def get_neck_radius_from_ring(ring_data, key="fitted_radius"):
    """
    Extract neck radius from saved ring data.

    Priority:
      1. explicit fitted_radius if present
      2. area-equivalent radius if present
      3. estimate from 2D or 3D ring points
    """
    if key in ring_data:
        return float(np.asarray(ring_data[key]))

    if "equivalent_radius_area" in ring_data:
        return float(np.asarray(ring_data["equivalent_radius_area"]))

    if "ring_st" in ring_data:
        ring_st = np.asarray(ring_data["ring_st"], dtype=float)
        center = ring_st.mean(axis=0)
        r = np.linalg.norm(ring_st - center[None, :], axis=1)
        return float(np.mean(r))

    if "points" in ring_data:
        pts = np.asarray(ring_data["points"], dtype=float)
        center = pts.mean(axis=0)
        r = np.linalg.norm(pts - center[None, :], axis=1)
        return float(np.mean(r))

    raise KeyError("Could not determine neck radius from ring data.")


def extract_valid_slice_summaries(stack_dict):
    """
    Return list of slice_summary for valid slices only.
    """
    return [sl["slice_summary"] for sl in stack_dict["slices"] if sl is not None]


# def largest_fitted_radius_from_stack(stack_dict):
#     summaries = extract_valid_slice_summaries(stack_dict)
#     if len(summaries) == 0:
#         return np.nan
#     return max(s["fitted_radius"] for s in summaries)

def largest_fitted_radius_from_stack(stack_dict, positive_only=True):
    valid_slices = [sl for sl in stack_dict["slices"] if sl is not None]

    if positive_only:
        valid_slices = [sl for sl in valid_slices if sl["z_offset"] > 0]

    if len(valid_slices) == 0:
        return np.nan

    return max(sl["slice_summary"]["fitted_radius"] for sl in valid_slices)    


def curvature_std_from_ring(ring_data):
    """
    Standard deviation of mean and Gaussian curvature along the neck ring.
    """
    H_std = float(np.std(np.asarray(ring_data["mean_curvature"], dtype=float)))
    K_std = float(np.std(np.asarray(ring_data["gauss"], dtype=float)))
    return H_std, K_std


def curvature_std_from_stack(stack_dict, mode="all_points"):
    """
    Compute std of mean/Gaussian curvature using the contour stack.

    mode:
      - "all_points": pool all contour points from all valid slices
      - "slice_means": std of per-slice mean curvature summaries
    """
    valid_slices = [sl for sl in stack_dict["slices"] if sl is not None]
    if len(valid_slices) == 0:
        return np.nan, np.nan

    if mode == "all_points":
        H = np.concatenate([
            np.asarray(sl["contour_geom"]["mean_curvature"], dtype=float)
            for sl in valid_slices
        ])
        K = np.concatenate([
            np.asarray(sl["contour_geom"]["gauss"], dtype=float)
            for sl in valid_slices
        ])
        return float(np.std(H)), float(np.std(K))

    elif mode == "slice_means":
        Hm = np.array([sl["slice_summary"]["H_mean"] for sl in valid_slices], dtype=float)
        Km = np.array([sl["slice_summary"]["K_mean"] for sl in valid_slices], dtype=float)
        return float(np.std(Hm)), float(np.std(Km))

    else:
        raise ValueError("mode must be 'all_points' or 'slice_means'")


def summarize_shapes_radius_ratio_vs_curvature_std(
    shape_list,
    ring_dict,
    stack_dict,
    curvature_source="ring",   # "ring" or "stack"
    stack_std_mode="all_points",
):
    """
    For each shape compute:
      x = (largest fitted radius in stack) / (neck radius)
      y = std of mean curvature
      y = std of Gaussian curvature
    """
    rows = []

    for shape in shape_list:
        neck_radius = get_neck_radius_from_ring(ring_dict[shape])
        max_radius = largest_fitted_radius_from_stack(stack_dict[shape])
        radius_ratio = max_radius / neck_radius if neck_radius > 0 else np.nan

        if curvature_source == "ring":
            H_std, K_std = curvature_std_from_ring(ring_dict[shape])
        elif curvature_source == "stack":
            H_std, K_std = curvature_std_from_stack(
                stack_dict[shape],
                mode=stack_std_mode,
            )
        else:
            raise ValueError("curvature_source must be 'ring' or 'stack'")

        rows.append({
            "shape": shape,
            "neck_radius": neck_radius,
            "max_fitted_radius": max_radius,
            "radius_ratio": radius_ratio,
            "H_std": H_std,
            "K_std": K_std,
        })

    return rows

def plot_radius_ratio_vs_curvature_std(rows, figsize=(8, 4), save_path=None):
    """
    Make two scatter plots:
      left : radius_ratio vs H_std
      right: radius_ratio vs K_std
    """
    x = np.array([r["radius_ratio"] for r in rows], dtype=float)
    H_std = np.array([r["H_std"] for r in rows], dtype=float)
    K_std = np.array([r["K_std"] for r in rows], dtype=float)
    labels = [r["shape"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].scatter(x, H_std)
    for xi, yi, lab in zip(x, H_std, labels):
        axes[0].annotate(lab, (xi, yi), xytext=(4, 4), textcoords="offset points")
    axes[0].set_xlabel("Radius ratio (max fitted radius / neck radius)")
    axes[0].set_ylabel("Std of mean curvature")
    axes[0].set_title("Radius ratio vs mean-curvature std")

    axes[1].scatter(x, K_std)
    for xi, yi, lab in zip(x, K_std, labels):
        axes[1].annotate(lab, (xi, yi), xytext=(4, 4), textcoords="offset points")
    axes[1].set_xlabel("Radius ratio (max fitted radius / neck radius)")
    axes[1].set_ylabel("Std of Gaussian curvature")
    axes[1].set_title("Radius ratio vs Gaussian-curvature std")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    return fig, axes


import numpy as np
import matplotlib.pyplot as plt


def largest_center_offset_from_stack(stack_dict):
    summaries = [sl["slice_summary"] for sl in stack_dict["slices"] if sl is not None]
    if len(summaries) == 0:
        return np.nan
    return max(s["center_offset_from_axis"] for s in summaries)


def summarize_shapes_offset_vs_curvature_std(
    shape_list,
    ring_dict,
    stack_dict,
    curvature_source="ring",   # "ring" or "stack"
    stack_std_mode="all_points",
):
    """
    For each shape compute:
      x = largest center-to-axis distance in stack
      y = std of mean curvature
      y = std of Gaussian curvature
    """
    rows = []

    for shape in shape_list:
        max_center_offset = largest_center_offset_from_stack(stack_dict[shape])

        if curvature_source == "ring":
            H = np.asarray(ring_dict[shape]["mean_curvature"], dtype=float)
            K = np.asarray(ring_dict[shape]["gauss"], dtype=float)
            H_std = float(np.std(H))
            K_std = float(np.std(K))

        elif curvature_source == "stack":
            valid_slices = [sl for sl in stack_dict[shape]["slices"] if sl is not None]
            if len(valid_slices) == 0:
                H_std, K_std = np.nan, np.nan
            elif stack_std_mode == "all_points":
                H = np.concatenate([
                    np.asarray(sl["contour_geom"]["mean_curvature"], dtype=float)
                    for sl in valid_slices
                ])
                K = np.concatenate([
                    np.asarray(sl["contour_geom"]["gauss"], dtype=float)
                    for sl in valid_slices
                ])
                H_std = float(np.std(H))
                K_std = float(np.std(K))
            elif stack_std_mode == "slice_means":
                Hm = np.array([sl["slice_summary"]["H_mean"] for sl in valid_slices], dtype=float)
                Km = np.array([sl["slice_summary"]["K_mean"] for sl in valid_slices], dtype=float)
                H_std = float(np.std(Hm))
                K_std = float(np.std(Km))
            else:
                raise ValueError("stack_std_mode must be 'all_points' or 'slice_means'")

        else:
            raise ValueError("curvature_source must be 'ring' or 'stack'")

        rows.append({
            "shape": shape,
            "max_center_offset": max_center_offset,
            "H_std": H_std,
            "K_std": K_std,
        })

    return rows


def plot_offset_vs_curvature_std(rows, figsize=(8, 4), save_path=None):
    """
    Make two scatter plots:
      left : max center-to-axis distance vs H_std
      right: max center-to-axis distance vs K_std
    """
    x = np.array([r["max_center_offset"] for r in rows], dtype=float)
    H_std = np.array([r["H_std"] for r in rows], dtype=float)
    K_std = np.array([r["K_std"] for r in rows], dtype=float)
    labels = [r["shape"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].scatter(x, H_std)
    for xi, yi, lab in zip(x, H_std, labels):
        axes[0].annotate(lab, (xi, yi), xytext=(4, 4), textcoords="offset points")
    axes[0].set_xlabel("Largest center-to-axis distance")
    axes[0].set_ylabel("Std of mean curvature")
    axes[0].set_title("Offset vs mean-curvature std")

    axes[1].scatter(x, K_std)
    for xi, yi, lab in zip(x, K_std, labels):
        axes[1].annotate(lab, (xi, yi), xytext=(4, 4), textcoords="offset points")
    axes[1].set_xlabel("Largest center-to-axis distance")
    axes[1].set_ylabel("Std of Gaussian curvature")
    axes[1].set_title("Offset vs Gaussian-curvature std")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    return fig, axes


import matplotlib.ticker as ticker

def plot_slice_summary_bar(
    stack,
    var="fitted_radius",
    reducer="max",
    shape_list=None,
    only_positive_z=False,
    ax=None,
    show_std=True,
    custom_labels=None,
    show_title=True,
    show_y_label=True,
):
    if shape_list is None:
        shape_list = list(stack.keys())

    reducer_map = {
        "max": np.max,
        "min": np.min,
        "mean": np.mean,
        "median": np.median,
    }
    if isinstance(reducer, str):
        reducer_fn = reducer_map[reducer]
        reducer_name = reducer
    else:
        reducer_fn = reducer
        reducer_name = getattr(reducer, "__name__", "custom")

    values_by_shape = {}
    vals_per_shape = {} 

    for shape in shape_list:
        slices = stack[shape]["slices"]
        vals = []

        for s in slices:
            if s is None:
                continue
            if only_positive_z and s["z_offset"] <= 0:
                continue
            if "slice_summary" not in s or s["slice_summary"] is None:
                continue

            summary = s["slice_summary"]
            if var not in summary:
                continue

            val = summary[var]
            if np.isscalar(val):
                vals.append(float(val))

        vals_per_shape[shape] = vals  # <-- save raw values
        values_by_shape[shape] = reducer_fn(vals) if vals else np.nan

    x = np.arange(len(shape_list))
    y = [values_by_shape[s] for s in shape_list]


    if ax is None:
        fig, ax = plt.subplots(figsize=(0.9, 1.5))
    else:
        fig = ax.figure

    if show_std:
        y_std = [np.std(vals_per_shape[s]) if len(vals_per_shape[s]) > 0 else 0 for s in shape_list]
        ax.bar(x, y, yerr=y_std, capsize=1.5, error_kw={"elinewidth": 0.7, "capthick": 0.7})
    else:
        ax.bar(x, y)



    # optional: annotate values
    # for xi, yi in zip(x, y):
    #     ax.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)

    if custom_labels is not None:
        ax.set_xticklabels(custom_labels, rotation=45, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels(shape_list, rotation=45)
    # ax.set_xlabel("Shape")

    if show_y_label:
        ax.set_ylabel(f"{reducer_name}({var})")

    if show_title:
        title = f"{reducer_name} {var}"
        if only_positive_z:
            title += " (z > 0)"
        ax.set_title(title)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)

    fig.tight_layout()

    # return values_by_shape
    return fig, ax


def plot_contour_property_vs_angle(
    stack_npz,
    shape=None,
    prop="mean_curvature",
    cmap="viridis",
    zmin=None,
    zmax=None,
    angle_unit="deg",
    ylim=None,
    show_legend=True,
    show_title=True,
    show_y_label=True,
):
    slices = stack_npz["slices"]
    basis_u = np.asarray(stack_npz["basis_u"])
    basis_v = np.asarray(stack_npz["basis_v"])

    fig, ax = plt.subplots(figsize=(0.6, 0.7))

    valid_entries = []
    for i, s in enumerate(slices):
        if s is None:
            continue
        if "contour_geom" not in s or s["contour_geom"] is None:
            continue
        if prop not in s["contour_geom"]:
            continue
        valid_entries.append((i, s))

    if len(valid_entries) == 0:
        raise ValueError(f"No valid slices with contour_geom['{prop}'] found.")

    zvals = np.array([s["z_offset"] for _, s in valid_entries], dtype=float)

    if zmin is None:
        zmin = zvals.min()
    if zmax is None:
        zmax = zvals.max()

    norm = plt.Normalize(vmin=zmin, vmax=zmax)
    cmap_obj = plt.get_cmap(cmap)

    for i, s in valid_entries:
        geom = s["contour_geom"]
        pts = np.asarray(geom["points"])       # (N,3)
        values = np.asarray(geom[prop])        # (N,)
        center = np.asarray(s["plane_center"]) # (3,)

        if pts.ndim != 2 or len(pts) != len(values):
            print(f"Skipping slice {i}: inconsistent shapes {pts.shape}, {values.shape}")
            continue

        rel = pts - center[None, :]
        coord_u = rel @ basis_u
        coord_v = rel @ basis_v

        theta = np.arctan2(coord_v, coord_u)
        order = np.argsort(theta)

        theta_sorted = theta[order]
        values_sorted = values[order]

        if angle_unit == "deg":
            theta_sorted = np.degrees(theta_sorted)
            xlabel = "Angle (deg)"
        elif angle_unit == "rad":
            xlabel = "Angle (rad)"
        else:
            raise ValueError("angle_unit must be 'rad' or 'deg'")

        color = cmap_obj(norm(s["z_offset"]))
        ax.plot(theta_sorted, values_sorted, color=color, lw=0.5)

    ax.set_xlabel(xlabel)
    if show_y_label:
        ax.set_ylabel(prop.replace("_", " "))

    if show_title:
        ax.set_title(f"{shape}: {prop} vs angle" if shape else f"{prop} vs angle")

    if ylim is not None: 
        ax.set_ylim(ylim)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])

    if show_legend:
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("z_offset")

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)

    # plt.tight_layout()
    # plt.show()

    return fig, ax



