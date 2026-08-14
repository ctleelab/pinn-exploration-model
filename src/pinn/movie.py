import os
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pinn.model import PINN
from pinn.plot import _make_slice_points, _batch_apply
from matplotlib.colors import LinearSegmentedColormap

custom_gray = LinearSegmentedColormap.from_list(
    "custom_gray",
    ["#f0f0f0", "#111111"],
)

def build_phi_fn(
    checkpoint,
    hidden_dim=128,
    voxel_scale=(1.0, 1.0, 1.0),
    run_on_cpu=False,
):
    """
    Construct the trained phase-field function from a checkpoint.
    """
    state = checkpoint["state"]
    params = state["params"]

    model = PINN(hidden_dim=hidden_dim)
    voxel_scale = jnp.asarray(voxel_scale)

    if run_on_cpu:
        device = jax.devices("cpu")[0]
        params = jax.device_put(params, device)
        voxel_scale = jax.device_put(voxel_scale, device)

    def phi_fn(points):
        scaled_points = points * voxel_scale[None, :]
        return model.apply(params, scaled_points)

    return phi_fn


def predict_phase_slice(
    phi_fn,
    grid_size,
    slice_index,
    axis="x",
    batch=4096,
    expand_xy=None,
    run_on_cpu=False,
):
    """
    Evaluate the trained phase field on one 2D slice.

    Returns
    -------
    phase_slice : ndarray
        Predicted phase-field slice.
    """
    x = jnp.linspace(-1, 1, grid_size)

    if expand_xy is None:
        y = jnp.linspace(-1, 1, grid_size)
        z = jnp.linspace(-1, 1, grid_size)
    else:
        y = jnp.linspace(-expand_xy, expand_xy, grid_size)
        z = jnp.linspace(-expand_xy, expand_xy, grid_size)

    points, shape_2d = _make_slice_points(
        x=x,
        y=y,
        z=z,
        axis=axis,
        slice_index=slice_index,
    )

    if run_on_cpu:
        cpu_device = jax.devices("cpu")[0]
        points = jax.device_put(points, cpu_device)

    def evaluate(points_batch):
        return phi_fn(points_batch).squeeze()

    values = _batch_apply(
        evaluate,
        points,
        batch=batch,
    )

    phase_slice = values.reshape(shape_2d)
    return np.asarray(jax.device_get(phase_slice))


def extract_cryoet_slice(volume, slice_index, axis):
    """
    Extract a cryoET slice using the same axis convention as the phase field.

    The volume dimensions are assumed to correspond to (x, y, z).
    """
    axis_to_dimension = {
        "x": 0,
        "y": 1,
        "z": 2,
    }

    if axis not in axis_to_dimension:
        raise ValueError(f"axis must be x, y, or z; received {axis!r}")

    dimension = axis_to_dimension[axis]

    return np.take(
        np.asarray(volume),
        slice_index,
        axis=dimension,
    )


def figure_to_rgb_array(fig):
    """
    Convert a Matplotlib figure into an RGB image array.
    """
    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba()).reshape(height, width, 4)

    return rgba[:, :, :3].copy()


def render_movie_frame(
    cryo_slice,
    phase_slice=None,
    raw_vmin=None,
    raw_vmax=None,
    slice_index=None,
    total_slices=None,
    figure_size=(6, 6),
    dpi=120,
    phase_alpha=0.35,
    show_text=True,
):
    """
    Render one movie frame.

    When phase_slice is supplied, the frame includes:
      1. a translucent membrane band where |phi| < membrane_band;
      2. a contour marking the phi = 0 surface.
    """
    fig, ax = plt.subplots(
        figsize=figure_size,
        dpi=dpi,
        facecolor="black",
    )

    ax.set_facecolor("black")

    ax.imshow(
        cryo_slice,
        cmap=custom_gray,
        origin="lower",
        extent=(-1, 1, -1, 1),
        vmin=raw_vmin,
        vmax=raw_vmax,
        interpolation="nearest",
        # interpolation="bilinear",
    )

    if phase_slice is not None:
        ax.imshow(
            phase_slice,
            cmap="coolwarm",
            origin="lower",
            extent=(-1, 1, -1, 1),
            vmin=-1,
            vmax=1,
            alpha=phase_alpha,
            interpolation="bilinear",
        )

    if show_text:
        if phase_slice is None:
            scene_label = "Raw cryoET"
        else:
            scene_label = "PINN reconstruction"

        ax.text(
            0.04,
            0.94,
            scene_label,
            transform=ax.transAxes,
            color="white",
            fontsize=15,
            horizontalalignment="left",
            verticalalignment="top",
            bbox={
                "facecolor": "black",
                "alpha": 0.40,
                "edgecolor": "none",
                "pad": 5,
            },
        )

        if slice_index is not None and total_slices is not None:
            ax.text(
                0.96,
                0.06,
                f"Slice {slice_index + 1}/{total_slices}",
                transform=ax.transAxes,
                color="white",
                fontsize=10,
                horizontalalignment="right",
                verticalalignment="bottom",
                alpha=0.85,
            )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
    )

    frame = figure_to_rgb_array(fig)
    plt.close(fig)

    return frame


def make_cryoet_phase_gif(
    cryoet,
    checkpoint,
    output_path,
    axis="x",
    hidden_dim=128,
    expand_xy=None,
    voxel_scale=(1.0, 1.0, 1.0),
    batch=4096,
    start_slice=0,
    end_slice=None,
    slice_step=1,
    fps=15,
    transition_pause_seconds=0.5,
    ending_pause_seconds=1.0,
    phase_alpha=0.35,
    percentile_range=(1.0, 99.5),
    figure_size=(6, 6),
    dpi=120,
    run_on_cpu=False,
    show_text=True,
):
    """
    Create a GIF with two scenes:

    Scene 1
        Raw cryoET slices, moving from the first slice to the last.

    Scene 2
        Raw cryoET slices plus phase-field reconstruction, moving backward.

    Parameters
    ----------
    cryoet : ndarray
        Three-dimensional cryoET volume.

    checkpoint : dict
        Restored Flax checkpoint.

    output_path : str
        Destination GIF path.

    slice_step : int
        Spacing between displayed slices. Use a larger value for a shorter GIF.

    fps : int
        Playback frame rate.

    percentile_range : tuple
        Percentiles used to determine the fixed cryoET intensity range.
    """
    cryoet = np.asarray(cryoet)

    axis_to_dimension = {
        "x": 0,
        "y": 1,
        "z": 2,
    }

    if axis not in axis_to_dimension:
        raise ValueError(f"axis must be x, y, or z; received {axis!r}")

    if slice_step < 1:
        raise ValueError("slice_step must be at least 1")

    dimension = axis_to_dimension[axis]
    number_of_slices = cryoet.shape[dimension]

    # Use a fixed intensity range throughout the movie to prevent flickering.
    raw_vmin, raw_vmax = np.percentile(
        cryoet[np.isfinite(cryoet)],
        percentile_range,
    )

    if end_slice is None:
        end_slice = number_of_slices - 1

    forward_indices = list(
        range(
            start_slice,
            end_slice + 1,
            slice_step,
        )
    )

    if forward_indices[-1] != end_slice:
        forward_indices.append(end_slice)

    reverse_indices = forward_indices[::-1]

    phi_fn = build_phi_fn(
        checkpoint=checkpoint,
        hidden_dim=hidden_dim,
        voxel_scale=voxel_scale,
        run_on_cpu=run_on_cpu,
    )

    # ------------------------------------------------------------
    # Precompute phase slices used in Scene 2.
    # ------------------------------------------------------------
    phase_slices = {}

    print(f"Computing {len(reverse_indices)} phase-field slices...")

    for frame_number, slice_index in enumerate(reverse_indices, start=1):
        phase_slices[slice_index] = predict_phase_slice(
            phi_fn=phi_fn,
            grid_size=number_of_slices,
            slice_index=slice_index,
            axis=axis,
            batch=batch,
            expand_xy=expand_xy,
            run_on_cpu=run_on_cpu,
        )

        print(
            f"\rPhase slice {frame_number}/{len(reverse_indices)}",
            end="",
            flush=True,
        )

    print("\nRendering GIF frames...")

    frames = []

    # ------------------------------------------------------------
    # Scene 1: raw cryoET, first slice to last slice.
    # ------------------------------------------------------------
    for frame_number, slice_index in enumerate(forward_indices, start=1):
        cryo_slice = extract_cryoet_slice(
            volume=cryoet,
            slice_index=slice_index,
            axis=axis,
        )

        frame = render_movie_frame(
            cryo_slice=cryo_slice,
            phase_slice=None,
            raw_vmin=raw_vmin,
            raw_vmax=raw_vmax,
            slice_index=slice_index,
            total_slices=number_of_slices,
            figure_size=figure_size,
            dpi=dpi,
            show_text=show_text,
        )

        frames.append(frame)

        print(
            f"\rRaw frame {frame_number}/{len(forward_indices)}",
            end="",
            flush=True,
        )

    # Pause at the bottom before beginning the reconstruction pass.
    transition_pause_frames = max(
        1,
        int(round(transition_pause_seconds * fps)),
    )

    frames.extend([frames[-1].copy()] * transition_pause_frames)

    # ------------------------------------------------------------
    # Scene 2: reconstruction overlay, last slice to first slice.
    # ------------------------------------------------------------
    for frame_number, slice_index in enumerate(reverse_indices, start=1):
        cryo_slice = extract_cryoet_slice(
            volume=cryoet,
            slice_index=slice_index,
            axis=axis,
        )

        frame = render_movie_frame(
            cryo_slice=cryo_slice,
            phase_slice=phase_slices[slice_index],
            raw_vmin=raw_vmin,
            raw_vmax=raw_vmax,
            slice_index=slice_index,
            total_slices=number_of_slices,
            figure_size=figure_size,
            dpi=dpi,
            phase_alpha=phase_alpha,
            show_text=show_text,
        )

        frames.append(frame)

        print(
            f"\rOverlay frame {frame_number}/{len(reverse_indices)}",
            end="",
            flush=True,
        )

    ending_pause_frames = max(
        1,
        int(round(ending_pause_seconds * fps)),
    )

    frames.extend([frames[-1].copy()] * ending_pause_frames)

    output_directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_directory, exist_ok=True)

    imageio.mimsave(
        output_path,
        frames,
        format="GIF",
        duration=1000 / fps,
        loop=0,
    )

    print(f"\nSaved GIF to:\n{os.path.abspath(output_path)}")

