import numpy as np
import jax
import jax.numpy as jnp
import vtk
from vtk.util import numpy_support


def evaluate_phase_field_for_vtk(
    phi_fn,
    grid_size=64,
    x_range=None,
    y_range=None,
    z_range=None,
    transpose=True,
    batch_size=100_000,
):
    """
    Evaluate the phase field using the same coordinate convention as the
    existing isosurface-extraction workflow.

    Returns
    -------
    phi_volume : np.ndarray
        Scalar volume in VTK-compatible array order (nz, ny, nx).
    coordinates : tuple
        Physical/model coordinate vectors (x, y, z).
    """

    if x_range is None:
        x_range = (-1.0, 1.0)
    if y_range is None:
        y_range = (-1.0, 1.0)
    if z_range is None:
        z_range = (-1.0, 1.0)

    # Allow either one grid size or separate dimensions
    if np.isscalar(grid_size):
        nx = ny = nz = int(grid_size)
    else:
        nx, ny, nz = map(int, grid_size)

    x = np.linspace(x_range[0], x_range[1], nx, dtype=np.float32)
    y = np.linspace(y_range[0], y_range[1], ny, dtype=np.float32)
    z = np.linspace(z_range[0], z_range[1], nz, dtype=np.float32)

    # Array axes are (z, y, x)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    if transpose:
        # Common convention when the original image is stored as (z, y, x)
        # but the neural network was trained with coordinates (x, y, z).
        model_points = np.column_stack(
            [
                xx.ravel(),
                yy.ravel(),
                zz.ravel(),
            ]
        )
    else:
        # Use this only if the model itself expects array-axis order (z, y, x).
        model_points = np.column_stack(
            [
                zz.ravel(),
                yy.ravel(),
                xx.ravel(),
            ]
        )

    model_points = model_points.astype(np.float32)

    phi_batches = []

    for start in range(0, len(model_points), batch_size):
        stop = min(start + batch_size, len(model_points))

        values = phi_fn(jnp.asarray(model_points[start:stop]))
        values = np.asarray(jax.device_get(values)).reshape(-1)

        phi_batches.append(values)

    phi_flat = np.concatenate(phi_batches)

    # VTK expects x to vary fastest, corresponding to NumPy shape (z, y, x).
    phi_volume = phi_flat.reshape(nz, ny, nx)

    return phi_volume, (x, y, z)


def numpy_phase_field_to_vtk_image(
    phi_volume,
    coordinates,
    physical_scale=1.0,
):
    """
    Convert a phase-field volume of shape (nz, ny, nx) into vtkImageData.
    """
    phi_volume = np.asarray(phi_volume, dtype=np.float32)

    if phi_volume.ndim != 3:
        raise ValueError(
            f"phi_volume must have shape (nz, ny, nx), got {phi_volume.shape}"
        )

    x, y, z = coordinates
    nz, ny, nx = phi_volume.shape

    if (len(x), len(y), len(z)) != (nx, ny, nz):
        raise ValueError(
            "Coordinate dimensions do not match volume: "
            f"volume={phi_volume.shape}, "
            f"coordinates={(len(x), len(y), len(z))}"
        )

    x = np.asarray(x, dtype=float) * physical_scale
    y = np.asarray(y, dtype=float) * physical_scale
    z = np.asarray(z, dtype=float) * physical_scale

    dx = float(x[1] - x[0]) if nx > 1 else 1.0
    dy = float(y[1] - y[0]) if ny > 1 else 1.0
    dz = float(z[1] - z[0]) if nz > 1 else 1.0

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(nx, ny, nz)
    image_data.SetOrigin(float(x[0]), float(y[0]), float(z[0]))
    image_data.SetSpacing(dx, dy, dz)

    vtk_array = numpy_support.numpy_to_vtk(
        phi_volume.ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )
    vtk_array.SetName("phase_field")

    image_data.GetPointData().SetScalars(vtk_array)

    return image_data

def render_zero_level_set(
    image_data,
    iso_value=0.0,
    window_size=(1000, 800),
):
    """
    Render the phi = iso_value isosurface.
    """
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image_data)
    contour.SetValue(0, iso_value)
    contour.ComputeNormalsOn()
    contour.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(contour.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetColor(0.85, 0.85, 0.9)
    actor.GetProperty().SetSpecular(0.3)
    actor.GetProperty().SetSpecularPower(20)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.08, 0.08, 0.1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(*window_size)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    renderer.ResetCamera()
    render_window.Render()
    interactor.Start()


def render_phase_field_volume(
    image_data,
    window_size=(1000, 800),
):
    """
    Volume-render a phase field whose values are approximately between -1 and 1.
    """
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(-1.0, 0.1, 0.2, 0.9)
    color.AddRGBPoint(0.0, 1.0, 1.0, 1.0)
    color.AddRGBPoint(1.0, 0.9, 0.15, 0.1)

    opacity = vtk.vtkPiecewiseFunction()

    # Mostly transparent bulk phases, opaque near phi = 0.
    opacity.AddPoint(-1.0, 0.01)
    opacity.AddPoint(-0.2, 0.03)
    opacity.AddPoint(-0.05, 0.65)
    opacity.AddPoint(0.0, 0.95)
    opacity.AddPoint(0.05, 0.65)
    opacity.AddPoint(0.2, 0.03)
    opacity.AddPoint(1.0, 0.01)

    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color)
    volume_property.SetScalarOpacity(opacity)
    volume_property.SetInterpolationTypeToLinear()
    volume_property.ShadeOn()
    volume_property.SetAmbient(0.3)
    volume_property.SetDiffuse(0.7)
    volume_property.SetSpecular(0.2)

    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(image_data)

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volume_property)

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.08, 0.08, 0.1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(*window_size)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    renderer.ResetCamera()
    render_window.Render()
    interactor.Start()

def save_vti(image_data, filename):
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image_data)
    writer.Write()

import os
import vtk


def save_phase_field_and_surface(
    image_data,
    output_prefix,
    iso_value=0.0,
):
    """
    Save:
      1. the complete 3D phase field as a .vti file
      2. the extracted phi = iso_value surface as a .vtp file

    Parameters
    ----------
    image_data : vtk.vtkImageData
        VTK image containing the 3D phase-field scalar values.
    output_prefix : str
        Output path without an extension.

        Example:
            output_prefix="outputs/golgi_01"

        creates:
            outputs/golgi_01_phase_field.vti
            outputs/golgi_01_surface.vtp

    iso_value : float, default=0.0
        Phase-field value used to extract the surface.
        For the membrane, this is normally phi = 0.
    """
    output_prefix = os.path.abspath(output_prefix)

    output_directory = os.path.dirname(output_prefix)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    vti_filename = f"{output_prefix}_phase_field.vti"
    vtp_filename = f"{output_prefix}_surface.vtp"

    # ---------------------------------------------------------
    # 1. Save the complete volumetric phase field
    # ---------------------------------------------------------
    volume_writer = vtk.vtkXMLImageDataWriter()
    volume_writer.SetFileName(vti_filename)
    volume_writer.SetInputData(image_data)

    if volume_writer.Write() != 1:
        raise RuntimeError(f"Failed to write phase field: {vti_filename}")

    # ---------------------------------------------------------
    # 2. Extract the phi = iso_value surface
    # ---------------------------------------------------------
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image_data)
    contour.SetValue(0, iso_value)
    contour.ComputeNormalsOn()
    contour.ComputeGradientsOff()
    contour.Update()

    surface = contour.GetOutput()

    if surface.GetNumberOfPoints() == 0:
        raise ValueError(
            f"No surface was found at phi = {iso_value}. "
            "Check the scalar range of image_data."
        )

    # ---------------------------------------------------------
    # 3. Save the extracted surface
    # ---------------------------------------------------------
    surface_writer = vtk.vtkXMLPolyDataWriter()
    surface_writer.SetFileName(vtp_filename)
    surface_writer.SetInputData(surface)

    if surface_writer.Write() != 1:
        raise RuntimeError(f"Failed to write surface: {vtp_filename}")

    print(f"Saved phase field: {vti_filename}")
    print(f"Saved surface:     {vtp_filename}")
    print(
        f"Surface contains {surface.GetNumberOfPoints()} points and "
        f"{surface.GetNumberOfCells()} triangles."
    )

    return vti_filename, vtp_filename


from pathlib import Path

import mrcfile
import numpy as np
import vtk
from vtk.util import numpy_support


# def mrc_to_vti_matching_phase_field(
#     mrc_path,
#     vti_path,
#     x_range,
#     y_range,
#     z_range,
#     half_length=1.0,
#     transpose=True,
#     scalar_name="cryoET_intensity",
# ):
#     """
#     Convert an MRC volume to VTI using the same coordinate convention and
#     physical scaling as the phase-field surface workflow.

#     Parameters
#     ----------
#     mrc_path : str or Path
#         Input MRC file.

#     vti_path : str or Path
#         Output VTI file.

#     x_range, y_range, z_range : tuple
#         Coordinate ranges used for the neural-network evaluation.

#     half_length : float
#         Same physical scaling used in:
#             verts = verts * half_length

#     transpose : bool
#         Apply the same transpose convention used by the existing membrane
#         surface extraction.

#         True:
#             raw MRC (z, y, x) -> display volume (x, y, z) through
#             np.transpose(volume, (2, 1, 0))

#         False:
#             keep the raw MRC array unchanged.

#     scalar_name : str
#         VTK scalar-array name.
#     """

#     mrc_path = Path(mrc_path)
#     vti_path = Path(vti_path)

#     with mrcfile.open(mrc_path, permissive=True) as mrc:
#         volume = np.asarray(mrc.data, dtype=np.float32).copy()

#     print(f"Raw MRC shape: {volume.shape}")

#     if volume.ndim != 3:
#         raise ValueError(f"Expected a 3D volume, got {volume.shape}")

#     if transpose:
#         # Raw MRC is normally indexed as (z, y, x).
#         # This matches the common transpose=True marching-cubes workflow.
#         volume = np.transpose(volume, (2, 1, 0))
#         print(f"Transposed shape: {volume.shape}")

#     # The current array must be interpreted as (z, y, x) for VTK export.
#     nz, ny, nx = volume.shape

#     # Apply the same physical scaling as the exported membrane vertices.
#     x_min = float(x_range[0]) * half_length
#     x_max = float(x_range[1]) * half_length

#     y_min = float(y_range[0]) * half_length
#     y_max = float(y_range[1]) * half_length

#     z_min = float(z_range[0]) * half_length
#     z_max = float(z_range[1]) * half_length

#     dx = (x_max - x_min) / max(nx - 1, 1)
#     dy = (y_max - y_min) / max(ny - 1, 1)
#     dz = (z_max - z_min) / max(nz - 1, 1)

#     image_data = vtk.vtkImageData()
#     image_data.SetDimensions(nx, ny, nz)
#     image_data.SetOrigin(x_min, y_min, z_min)
#     image_data.SetSpacing(dx, dy, dz)

#     vtk_array = numpy_support.numpy_to_vtk(
#         volume.ravel(order="C"),
#         deep=True,
#         array_type=vtk.VTK_FLOAT,
#     )
#     vtk_array.SetName(scalar_name)

#     image_data.GetPointData().SetScalars(vtk_array)

#     vti_path.parent.mkdir(parents=True, exist_ok=True)

#     writer = vtk.vtkXMLImageDataWriter()
#     writer.SetFileName(str(vti_path))
#     writer.SetInputData(image_data)

#     if writer.Write() != 1:
#         raise RuntimeError(f"Failed to write {vti_path}")

#     print(f"Saved:       {vti_path}")
#     print(f"Dimensions:  {(nx, ny, nz)}")
#     print(f"Origin:      {(x_min, y_min, z_min)}")
#     print(f"Spacing:     {(dx, dy, dz)}")
#     print(f"VTK bounds:  {image_data.GetBounds()}")

#     return image_data


# def mrc_to_vti_like_phase_field(
#     mrc_path,
#     vti_path,
#     x_range=None,
#     y_range=None,
#     z_range=None,
#     transpose=True,
#     half_length=1.0,
#     scalar_name="cryoET_intensity",
# ):
#     """
#     Export MRC data using the same spatial convention as
#     evaluate_phase_field_for_vtk.

#     The output NumPy volume is always stored as (z, y, x), matching the
#     phi_volume returned by evaluate_phase_field_for_vtk.
#     """
#     from pathlib import Path

#     import mrcfile
#     import numpy as np
#     import vtk
#     from vtk.util import numpy_support

#     mrc_path = Path(mrc_path)
#     vti_path = Path(vti_path)

#     with mrcfile.open(mrc_path, permissive=True) as mrc:
#         raw_volume = np.asarray(mrc.data, dtype=np.float32).copy()

#     if raw_volume.ndim != 3:
#         raise ValueError(f"Expected a 3D MRC volume, got {raw_volume.shape}")

#     print("Raw MRC shape:", raw_volume.shape)

#     if transpose:
#         # Match the intended transpose=True convention:
#         # convert the input orientation, then store the result as (z, y, x).
#         volume = np.transpose(raw_volume, (2, 1, 0))
#     else:
#         volume = raw_volume

#     # The array passed to VTK is interpreted as (z, y, x).
#     nz, ny, nx = volume.shape

#     if x_range is None:
#         x_range = (-1.0, 1.0)
#     if y_range is None:
#         y_range = (-1.0, 1.0)
#     if z_range is None:
#         z_range = (-1.0, 1.0)

#     # Use exactly the same linspace convention as evaluate_phase_field_for_vtk.
#     x = np.linspace(
#         x_range[0],
#         x_range[1],
#         nx,
#         dtype=np.float32,
#     )
#     y = np.linspace(
#         y_range[0],
#         y_range[1],
#         ny,
#         dtype=np.float32,
#     )
#     z = np.linspace(
#         z_range[0],
#         z_range[1],
#         nz,
#         dtype=np.float32,
#     )

#     # Apply the same physical scaling used for the membrane vertices.
#     x = x * half_length
#     y = y * half_length
#     z = z * half_length

#     dx = float(x[1] - x[0]) if nx > 1 else 1.0
#     dy = float(y[1] - y[0]) if ny > 1 else 1.0
#     dz = float(z[1] - z[0]) if nz > 1 else 1.0

#     vtk_image = vtk.vtkImageData()
#     vtk_image.SetDimensions(nx, ny, nz)
#     vtk_image.SetOrigin(
#         float(x[0]),
#         float(y[0]),
#         float(z[0]),
#     )
#     vtk_image.SetSpacing(dx, dy, dz)

#     vtk_array = numpy_support.numpy_to_vtk(
#         volume.ravel(order="C"),
#         deep=True,
#         array_type=vtk.VTK_FLOAT,
#     )
#     vtk_array.SetName(scalar_name)

#     vtk_image.GetPointData().SetScalars(vtk_array)

#     vti_path.parent.mkdir(parents=True, exist_ok=True)

#     writer = vtk.vtkXMLImageDataWriter()
#     writer.SetFileName(str(vti_path))
#     writer.SetInputData(vtk_image)

#     if writer.Write() != 1:
#         raise RuntimeError(f"Failed to write {vti_path}")

#     print("Exported volume shape (z,y,x):", volume.shape)
#     print("VTK dimensions (x,y,z):", vtk_image.GetDimensions())
#     print("VTK origin:", vtk_image.GetOrigin())
#     print("VTK spacing:", vtk_image.GetSpacing())
#     print("VTK bounds:", vtk_image.GetBounds())
#     print("Saved:", vti_path)

#     return vtk_image


from pathlib import Path

import mrcfile
import numpy as np
import vtk
from vtk.util import numpy_support


def mrc_to_vti_like_phase_field(
    mrc_path,
    vti_path,
    x_range=None,
    y_range=None,
    z_range=None,
    full_x_range=(-1.0, 1.0),
    full_y_range=(-1.0, 1.0),
    full_z_range=(-1.0, 1.0),
    transpose=True,
    half_length=1.0,
    scalar_name="cryoET_intensity",
):
    """
    Crop an MRC volume in normalized coordinates and save it as VTI.

    This preserves the orientation convention of the previous
    mrc_to_vti_like_phase_field function, but x_range, y_range, and z_range
    now crop the voxel data instead of compressing the complete volume.

    Parameters
    ----------
    mrc_path : str or Path
        Input MRC file.

    vti_path : str or Path
        Output VTI file.

    x_range, y_range, z_range : tuple or None
        Requested crop ranges in normalized coordinates.

        If None, the complete extent along that axis is retained.

    full_x_range, full_y_range, full_z_range : tuple
        Normalized-coordinate bounds of the complete oriented MRC volume.

        These describe the full MRC before cropping.

    transpose : bool
        Apply the same orientation convention as the previous successful
        version:

            volume = np.transpose(raw_volume, (2, 1, 0))

    half_length : float
        Converts normalized coordinates into physical coordinates:

            physical_coordinate = normalized_coordinate * half_length

    scalar_name : str
        Name of the intensity scalar in ParaView.

    Returns
    -------
    vtk_image : vtk.vtkImageData
        Exported cropped volume.

    cropped_volume : np.ndarray
        Cropped NumPy array in (z, y, x) storage order.

    crop_coordinates : tuple
        Physical coordinate arrays (x, y, z).
    """
    mrc_path = Path(mrc_path)
    vti_path = Path(vti_path)

    # ---------------------------------------------------------
    # 1. Load MRC
    # ---------------------------------------------------------
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        raw_volume = np.asarray(mrc.data, dtype=np.float32).copy()

    if raw_volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D MRC volume, got shape {raw_volume.shape}"
        )

    print(f"Raw MRC shape: {raw_volume.shape}")

    # ---------------------------------------------------------
    # 2. Apply the established orientation convention
    # ---------------------------------------------------------
    if transpose:
        volume = np.transpose(raw_volume, (2, 1, 0))
    else:
        volume = raw_volume

    # From here onward, the array is interpreted as (z, y, x).
    nz_full, ny_full, nx_full = volume.shape

    print(f"Oriented shape (z,y,x): {volume.shape}")

    # ---------------------------------------------------------
    # 3. Construct coordinates of the complete MRC
    # ---------------------------------------------------------
    x_full = np.linspace(
        full_x_range[0],
        full_x_range[1],
        nx_full,
        dtype=np.float64,
    )
    y_full = np.linspace(
        full_y_range[0],
        full_y_range[1],
        ny_full,
        dtype=np.float64,
    )
    z_full = np.linspace(
        full_z_range[0],
        full_z_range[1],
        nz_full,
        dtype=np.float64,
    )

    # None means retain the entire axis.
    if x_range is None:
        x_range = full_x_range

    if y_range is None:
        y_range = full_y_range

    if z_range is None:
        z_range = full_z_range

    # ---------------------------------------------------------
    # 4. Validate crop ranges
    # ---------------------------------------------------------
    def validate_crop_range(name, crop_range, full_range):
        if crop_range[0] >= crop_range[1]:
            raise ValueError(
                f"{name}_range must be increasing, got {crop_range}"
            )

        if crop_range[0] < full_range[0] or crop_range[1] > full_range[1]:
            raise ValueError(
                f"{name}_range={crop_range} lies outside the complete "
                f"MRC range {full_range}."
            )

    validate_crop_range("x", x_range, full_x_range)
    validate_crop_range("y", y_range, full_y_range)
    validate_crop_range("z", z_range, full_z_range)

    # ---------------------------------------------------------
    # 5. Convert coordinate ranges into voxel slices
    # ---------------------------------------------------------
    def coordinate_range_to_slice(coordinates, crop_range):
        """
        Return a Python slice containing voxel centers within crop_range.
        """
        start = int(
            np.searchsorted(
                coordinates,
                crop_range[0],
                side="left",
            )
        )

        stop = int(
            np.searchsorted(
                coordinates,
                crop_range[1],
                side="right",
            )
        )

        start = max(0, min(start, len(coordinates)))
        stop = max(start, min(stop, len(coordinates)))

        if stop <= start:
            raise ValueError(
                f"Crop range {crop_range} contains no voxel centers."
            )

        return slice(start, stop)

    x_slice = coordinate_range_to_slice(x_full, x_range)
    y_slice = coordinate_range_to_slice(y_full, y_range)
    z_slice = coordinate_range_to_slice(z_full, z_range)

    # Array order is (z, y, x).
    cropped_volume = volume[
        z_slice,
        y_slice,
        x_slice,
    ]

    # Retain the actual voxel-center coordinates.
    x_crop = x_full[x_slice]
    y_crop = y_full[y_slice]
    z_crop = z_full[z_slice]

    nz, ny, nx = cropped_volume.shape

    # ---------------------------------------------------------
    # 6. Convert normalized coordinates to physical coordinates
    # ---------------------------------------------------------
    x_physical = x_crop * half_length
    y_physical = y_crop * half_length
    z_physical = z_crop * half_length

    dx = (
        float(x_physical[1] - x_physical[0])
        if nx > 1
        else 1.0
    )
    dy = (
        float(y_physical[1] - y_physical[0])
        if ny > 1
        else 1.0
    )
    dz = (
        float(z_physical[1] - z_physical[0])
        if nz > 1
        else 1.0
    )

    # ---------------------------------------------------------
    # 7. Create vtkImageData
    # ---------------------------------------------------------
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(nx, ny, nz)

    vtk_image.SetOrigin(
        float(x_physical[0]),
        float(y_physical[0]),
        float(z_physical[0]),
    )

    vtk_image.SetSpacing(dx, dy, dz)

    vtk_array = numpy_support.numpy_to_vtk(
        cropped_volume.ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )
    vtk_array.SetName(scalar_name)

    vtk_image.GetPointData().SetScalars(vtk_array)

    # ---------------------------------------------------------
    # 8. Save VTI
    # ---------------------------------------------------------
    vti_path.parent.mkdir(parents=True, exist_ok=True)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(vti_path))
    writer.SetInputData(vtk_image)

    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTI file: {vti_path}")

    print("Crop ranges:")
    print(f"  x: {x_range}")
    print(f"  y: {y_range}")
    print(f"  z: {z_range}")

    print("Crop slices:")
    print(f"  x: [{x_slice.start}:{x_slice.stop}]")
    print(f"  y: [{y_slice.start}:{y_slice.stop}]")
    print(f"  z: [{z_slice.start}:{z_slice.stop}]")

    print(f"Cropped shape (z,y,x): {cropped_volume.shape}")
    print(f"VTK dimensions (x,y,z): {vtk_image.GetDimensions()}")
    print(f"VTK origin: {vtk_image.GetOrigin()}")
    print(f"VTK spacing: {vtk_image.GetSpacing()}")
    print(f"VTK bounds: {vtk_image.GetBounds()}")
    print(f"Saved: {vti_path}")

    return vtk_image, cropped_volume, (
        x_physical,
        y_physical,
        z_physical,
    )
