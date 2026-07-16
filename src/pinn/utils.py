from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from pinn.model import PINN


LOSS_KEYS = (
    "step",
    "total_loss",
    "data_loss",
    "phys_loss",
    "sign_loss",
    "curv_loss",
)


# ---------------------------------------------------------------------
# Point sampling
# ---------------------------------------------------------------------


def sample_surface_points(
    key: jax.Array,
    checkpoint: Mapping[str, Any],
    n_points: int,
    *,
    bounds: tuple[float, float] = (-1.0, 1.0),
    oversample: int = 50,
    phi_band: float = 0.02,
    max_rounds: int = 10,
    perm: tuple[int, int, int] = (2, 1, 0),
) -> dict[str, jax.Array]:
    """Sample points near the zero level set of a trained phase field.

    Candidate points are drawn uniformly from the specified cubic domain.
    Points satisfying ``abs(phi) < phi_band`` are retained until at least
    ``n_points`` have been collected.

    Parameters
    ----------
    key
        JAX random key.
    checkpoint
        Restored checkpoint containing ``checkpoint["state"]["params"]``.
    n_points
        Number of points to return.
    bounds
        Lower and upper coordinate bounds used for all three axes.
    oversample
        Number of candidates drawn per remaining requested point in each round.
    phi_band
        Maximum absolute phase-field value for accepting a point.
    max_rounds
        Maximum number of candidate-sampling rounds.
    perm
        Axis permutation applied before returning the sampled points. The
        default converts model coordinates from ``(z, y, x)`` to ``(x, y, z)``.

    Returns
    -------
    dict
        Dictionary containing ``"points"`` with shape ``(n_points, 3)``.

    Raises
    ------
    ValueError
        If the arguments are invalid or insufficient points are collected.
    """
    if n_points <= 0:
        raise ValueError(f"n_points must be positive, but received {n_points}.")

    if oversample <= 0:
        raise ValueError(
            f"oversample must be positive, but received {oversample}."
        )

    if max_rounds <= 0:
        raise ValueError(
            f"max_rounds must be positive, but received {max_rounds}."
        )

    if phi_band <= 0:
        raise ValueError(
            f"phi_band must be positive, but received {phi_band}."
        )

    lower, upper = bounds
    if lower >= upper:
        raise ValueError(
            f"Expected bounds[0] < bounds[1], but received {bounds}."
        )

    params = checkpoint["state"]["params"]
    model = PINN()

    def phi_fn(points: jax.Array) -> jax.Array:
        return model.apply(params, points).reshape(-1)

    collected_points = []
    n_collected = 0

    for _ in range(max_rounds):
        key, candidate_key = jax.random.split(key)

        n_remaining = max(n_points - n_collected, 1)
        n_candidates = oversample * n_remaining

        candidates = jax.random.uniform(
            candidate_key,
            shape=(n_candidates, 3),
            minval=lower,
            maxval=upper,
        )

        phase_values = phi_fn(candidates)
        near_surface = jnp.abs(phase_values) < phi_band

        accepted = candidates[near_surface]
        collected_points.append(accepted)
        n_collected += accepted.shape[0]

        if n_collected >= n_points:
            break

    points = jnp.concatenate(collected_points, axis=0)

    if points.shape[0] < n_points:
        raise ValueError(
            "Could not collect enough near-surface points. "
            f"Collected {points.shape[0]} of {n_points} points with "
            f"|phi| < {phi_band}. Increase oversample, phi_band, "
            "or max_rounds."
        )

    key, selection_key = jax.random.split(key)
    selected_indices = jax.random.choice(
        selection_key,
        points.shape[0],
        shape=(n_points,),
        replace=False,
    )
    points = points[selected_indices]

    key, shuffle_key = jax.random.split(key)
    points = jax.random.permutation(
        shuffle_key,
        points,
        axis=0,
    )

    points = points[:, perm]

    return {"points": points}


# ---------------------------------------------------------------------
# Point-data input/output
# ---------------------------------------------------------------------


def save_pts_data(
    data: Mapping[str, Any],
    path: str | PathLike[str],
    meta: Mapping[str, Any] | None = None,
) -> None:
    """Save point data and optional metadata as a compressed NumPy archive.

    Parameters
    ----------
    data
        Mapping containing ``"points"`` with shape ``(N, 3)``. It may also
        contain a ``"label"`` array.
    path
        Output ``.npz`` path.
    meta
        Optional metadata dictionary. Metadata is stored as a pickled object.
    """
    output = {
        "points": np.asarray(data["points"]),
    }

    if "label" in data and data["label"] is not None:
        output["label"] = np.asarray(data["label"])

    if meta is not None:
        output["meta"] = np.asarray(meta, dtype=object)

    np.savez(path, **output)


def load_pts_data(
    path: str | PathLike[str],
    perm: tuple[int, int, int] = (0, 1, 2),
    scale: float | Sequence[float] | None = None,
) -> dict[str, Any]:
    """Load point data from a NumPy archive.

    Parameters
    ----------
    path
        Input ``.npz`` path.
    perm
        Axis permutation applied to the points.
    scale
        Optional uniform scale or three per-axis scale factors.

    Returns
    -------
    dict
        Dictionary containing ``"points"``, ``"label"``, and ``"meta"``.
        Missing labels or metadata are returned as ``None``.
    """
    with np.load(path, allow_pickle=True) as archive:
        points = np.asarray(archive["points"])
        points = points[:, perm]

        if scale is not None:
            scale_array = np.asarray(scale)

            if scale_array.ndim == 0:
                points = points * scale_array
            else:
                if scale_array.shape != (3,):
                    raise ValueError(
                        "Per-axis scale must contain exactly three values, "
                        f"but received shape {scale_array.shape}."
                    )
                points = points * scale_array[None, :]

        label = (
            np.asarray(archive["label"])
            if "label" in archive.files
            else None
        )
        meta = (
            archive["meta"].item()
            if "meta" in archive.files
            else None
        )

    return {
        "points": points,
        "label": label,
        "meta": meta,
    }


def strip_meta(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the point and label arrays from a data dictionary."""
    return {
        "points": data["points"],
        "label": data.get("label"),
    }


# ---------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------


def to_numpy(tree: Any) -> Any:
    """Move JAX arrays to the host and convert them to NumPy arrays."""

    def convert(value):
        if isinstance(value, (jax.Array, np.ndarray)):
            return np.asarray(jax.device_get(value))
        return value

    return jax.tree_util.tree_map(convert, tree)


def as_f32_scalar(value: Any) -> float:
    """Convert a scalar-like value to a Python float through float32."""
    return np.asarray(value, dtype=np.float32).reshape(()).item()


def pick(
    mapping: Mapping[str, Any],
    *keys: str,
    default: Any = 0.0,
) -> Any:
    """Return the value associated with the first existing key."""
    for key in keys:
        if key in mapping:
            return mapping[key]

    return default


def save_ckpt(
    checkpoint_dir: str | PathLike[str],
    step: int,
    state: Any,
    loss_batch: Mapping[str, Any],
    keep: int,
) -> str:
    """Save a training checkpoint.

    The checkpoint contains a host-side copy of the training state and the
    associated loss history.
    """
    payload = {
        "state": to_numpy(state),
        "loss": loss_batch,
    }

    return checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=payload,
        step=step,
        overwrite=False,
        keep=keep,
    )


# ---------------------------------------------------------------------
# Loss-history utilities
# ---------------------------------------------------------------------


def _as_1d_numpy(value: Any) -> np.ndarray:
    """Convert an array-like value to a one-dimensional NumPy array."""
    return np.asarray(value).reshape(-1)


def load_loss_from_checkpoint(
    checkpoint_path: str | PathLike[str],
) -> dict[str, np.ndarray]:
    """Load loss arrays from one checkpoint."""
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_path,
        target=None,
    )

    if restored is None or "loss" not in restored:
        raise ValueError(
            f"No loss history found in checkpoint: {checkpoint_path}"
        )

    loss = restored["loss"]

    missing_keys = [key for key in LOSS_KEYS if key not in loss]
    if missing_keys:
        raise KeyError(
            "Checkpoint loss history is missing keys: "
            + ", ".join(missing_keys)
        )

    return {
        key: _as_1d_numpy(loss[key])
        for key in LOSS_KEYS
    }


def load_loss_history_dir(
    checkpoint_dir: str | PathLike[str],
    checkpoint_steps: Sequence[int],
    step_offset: int = 0,
) -> dict[str, np.ndarray]:
    """Load and combine loss histories from selected checkpoints.

    Parameters
    ----------
    checkpoint_dir
        Directory containing files named ``checkpoint_<step>``.
    checkpoint_steps
        Checkpoint step numbers to load.
    step_offset
        Offset added to every stored training step.

    Returns
    -------
    dict
        Sorted loss-history arrays. When duplicate steps occur, the last
        occurrence is retained.
    """
    checkpoint_dir = Path(checkpoint_dir)
    histories = {key: [] for key in LOSS_KEYS}

    for checkpoint_step in checkpoint_steps:
        checkpoint_path = (
            checkpoint_dir / f"checkpoint_{checkpoint_step}"
        )
        loss = load_loss_from_checkpoint(checkpoint_path)

        histories["step"].append(loss["step"] + step_offset)

        for key in LOSS_KEYS:
            if key != "step":
                histories[key].append(loss[key])

    history = {
        key: (
            np.concatenate(histories[key])
            if histories[key]
            else np.array([])
        )
        for key in LOSS_KEYS
    }

    return _sort_and_deduplicate_history(history)


def concat_histories(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Concatenate two loss histories and resolve duplicate steps."""
    if first["step"].size == 0:
        return {key: np.asarray(second[key]) for key in LOSS_KEYS}

    if second["step"].size == 0:
        return {key: np.asarray(first[key]) for key in LOSS_KEYS}

    combined = {
        key: np.concatenate([first[key], second[key]])
        for key in LOSS_KEYS
    }

    return _sort_and_deduplicate_history(combined)


def _sort_and_deduplicate_history(
    history: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Sort a loss history and keep the last value for duplicate steps."""
    history = {
        key: np.asarray(history[key])
        for key in LOSS_KEYS
    }

    if history["step"].size == 0:
        return history

    order = np.argsort(history["step"])
    history = {
        key: history[key][order]
        for key in LOSS_KEYS
    }

    reversed_steps = history["step"][::-1]
    _, reversed_unique_indices = np.unique(
        reversed_steps,
        return_index=True,
    )

    keep_indices = (
        history["step"].size - 1 - reversed_unique_indices
    )
    keep_indices.sort()

    return {
        key: history[key][keep_indices]
        for key in LOSS_KEYS
    }


def assemble_loss_history(
    checkpoint_data: Mapping[Any, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Assemble loss histories from restored checkpoint dictionaries."""
    histories = {key: [] for key in LOSS_KEYS}

    for checkpoint in checkpoint_data.values():
        if "loss" not in checkpoint:
            continue

        loss = checkpoint["loss"]

        for key in LOSS_KEYS:
            histories[key].extend(
                _as_1d_numpy(loss[key]).tolist()
            )

    return {
        key: np.asarray(histories[key])
        for key in LOSS_KEYS
    }


def loss_dict_to_batched(
    loss_list: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Convert per-step loss dictionaries into batched NumPy arrays."""
    return {
        "step": np.asarray(
            [entry["step"] for entry in loss_list],
            dtype=np.int64,
        ),
        **{
            key: np.asarray(
                [
                    as_f32_scalar(entry[key])
                    for entry in loss_list
                ],
                dtype=np.float32,
            )
            for key in LOSS_KEYS
            if key != "step"
        },
    }

