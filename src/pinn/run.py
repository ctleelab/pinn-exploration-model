from __future__ import annotations

import argparse
from pathlib import Path

import absl.logging
import jax
import matplotlib.pyplot as plt
import tifffile as tiff
from flax.training import checkpoints
from tqdm.auto import trange

from pinn.plot import (
    plot_unnormalized_loss_history_ax,
    visualize_cryoET_with_contours,
    visualize_phase,
)
from pinn.train import (
    create_train_state, 
    make_train_step, 
    compute_initial_losses
)
from pinn.utils import (
    assemble_loss_history,
    load_pts_data,
    loss_dict_to_batched,
    pick,
    sample_surface_points,
    save_ckpt,
    save_pts_data,
    strip_meta,
)


absl.logging.set_verbosity(absl.logging.ERROR)

import jax.experimental.layout as _layout

if not hasattr(_layout, "DeviceLocalLayout"):
    _layout.DeviceLocalLayout = _layout.Layout


NUM_CHECKPOINTS_TO_KEEP = 1_000
DEFAULT_VISUALIZATION_STEPS = (0, 1_000, 5_000, 10_000)


def _format_number(value: float) -> str:
    """Format numeric values consistently for checkpoint directory names."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _checkpoint_dir(output_dir, stage, lambda_1, lambda_2, lambda_3, lambda_4):
    """Return the checkpoint directory for one training stage."""
    name = "stage{}_{}_{}_{}_{}".format(
        stage,
        _format_number(lambda_1),
        _format_number(lambda_2),
        _format_number(lambda_3),
        _format_number(lambda_4),
    )
    return Path(output_dir).expanduser().resolve() / name


def _restore_checkpoint(checkpoint_path):
    """Restore a checkpoint and raise a clear error when it cannot be found."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = checkpoints.restore_checkpoint(ckpt_dir=str(checkpoint_path), target=None)

    if checkpoint is None:
        raise RuntimeError(f"Unable to restore checkpoint: {checkpoint_path}")

    return checkpoint


def _load_checkpoint_series(checkpoint_dir, max_step, save_interval):
    """Load all available checkpoints up to max_step."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_data = {}

    for step in range(0, max_step + 1, save_interval):
        checkpoint_path = checkpoint_dir / f"checkpoint_{step}"
        if checkpoint_path.exists():
            checkpoint_data[step] = _restore_checkpoint(checkpoint_path)

    return checkpoint_data


def plot_result(
    data_path,
    checkpoint_dir,
    shape,
    steps_to_visualize=(0, 1000, 5000, 10000),
    save_interval=100,
    axis="x",
):
    """Plot CryoET contours, phase field, energy density, and loss history."""
    data_path = Path(data_path)
    checkpoint_dir = Path(checkpoint_dir)

    if not steps_to_visualize:
        raise ValueError("steps_to_visualize must contain at least one step.")

    checkpoint_data = _load_checkpoint_series(
        checkpoint_dir=checkpoint_dir,
        max_step=max(steps_to_visualize),
        save_interval=save_interval,
    )
    steps_to_visualize = [step for step in steps_to_visualize if step in checkpoint_data]

    if not steps_to_visualize:
        print(f"[plot_result] No requested checkpoints found in {checkpoint_dir}")
        return None

    file_path = data_path / f"{shape}.tif"

    if not file_path.exists():
        raise FileNotFoundError(f"CryoET volume not found: {file_path}")

    cryoet_data = tiff.imread(file_path).astype("float32")
    max_intensity = float(cryoet_data.max())

    if max_intensity > 0:
        cryoet_data /= max_intensity

    cryoet_data = 1.0 - cryoet_data
    grid_size = cryoet_data.shape[0]
    slice_index = grid_size // 2
    n_columns = len(steps_to_visualize)

    fig = plt.figure(figsize=(3 * n_columns, 12))

    for i, step in enumerate(steps_to_visualize):
        ax = fig.add_subplot(4, n_columns, i + 1)
        visualize_cryoET_with_contours(
            ax, step, checkpoint_data[step], cryoet_data,
            grid_size=grid_size, slice_index=slice_index,
            axis=axis, thresholding=False,
        )
        if i == 0:
            ax.set_ylabel("CryoET + contours")

    for i, step in enumerate(steps_to_visualize):
        ax = fig.add_subplot(4, n_columns, n_columns + i + 1)
        visualize_phase(
            ax, epsilon=0.05, checkpoint=checkpoint_data[step], component="phi",
            grid_size=grid_size, slice_index=slice_index, axis=axis,
            no_label=True, vmin=-1.0, vmax=1.0,
        )
        if i == 0:
            ax.set_ylabel("Phase field")

    for i, step in enumerate(steps_to_visualize):
        ax = fig.add_subplot(4, n_columns, 2 * n_columns + i + 1)
        visualize_phase(
            ax, epsilon=0.05, checkpoint=checkpoint_data[step], component="tension",
            grid_size=grid_size, slice_index=slice_index, axis=axis,
            no_label=True, vmin=0.0, vmax=1.0,
        )
        if i == 0:
            ax.set_ylabel("Energy")

    assembled_loss = assemble_loss_history(checkpoint_data)
    loss_indices = (1, 2, 3, 4)

    for i in range(n_columns):
        ax = fig.add_subplot(4, n_columns, 3 * n_columns + i + 1)

        if i < len(loss_indices):
            plot_unnormalized_loss_history_ax(ax, loss_indices[i], assembled_loss)
        else:
            ax.axis("off")

    fig.tight_layout()

    output_path = checkpoint_dir / "result.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")
    return output_path


def sample_curv_points(
    data_path: str,
    output_dir: str,
    shape: str,
    lambda_1: float = 100000,
    lambda_2: float = 10,
    lambda_3: float = 100000,
    num_curv: int = 5000,
    seed: int = 0,
    step: int = 10000,
):
    """Sample curvature-loss collocation points from a stage-1 checkpoint."""
    key = jax.random.PRNGKey(seed)
    checkpoint_dir = _checkpoint_dir(output_dir, 1, lambda_1, lambda_2, lambda_3, 0)
    checkpoint_path = checkpoint_dir / f"checkpoint_{step}"
    checkpoint_data = _restore_checkpoint(checkpoint_path)

    print(f"Sampling curvature points from: {checkpoint_path}")

    data_curv = sample_surface_points(key, checkpoint_data, num_curv, oversample=500)
    meta_curv = {
        "type": "curv",
        "n_sample": num_curv,
        "seed": seed,
        "phi_fn": str(checkpoint_path),
    }

    save_path = Path(data_path) / f"pt_c_{shape}.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_pts_data(data_curv, save_path, meta_curv)

    print(f"Saved: {save_path}")
    return save_path


def _load_training_data(data_path, shape, stage):
    """Load edge, sign, physics, and curvature point datasets."""
    data_path = Path(data_path)

    edge = load_pts_data(data_path / f"pt_e_{shape}.npz", perm=(2, 1, 0))
    sign = load_pts_data(data_path / f"pt_s_{shape}.npz", perm=(2, 1, 0))
    phys = load_pts_data(data_path / f"pt_p_{shape}.npz", perm=(2, 1, 0))

    curv_path = data_path / f"pt_c_{shape}.npz" if stage == 2 else data_path / "dummy.npz"
    curv = load_pts_data(curv_path, perm=(2, 1, 0))

    return (
        jax.device_put(strip_meta(edge)),
        jax.device_put(strip_meta(sign)),
        jax.device_put(strip_meta(phys)),
        jax.device_put(strip_meta(curv)),
    )


def _get_initial_checkpoint(
    output_dir,
    stage,
    lambda_1,
    lambda_2,
    lambda_3,
    num_steps,
    init_ckpt_path=None,
):
    """Load the explicitly supplied or automatically inferred initial checkpoint."""
    if init_ckpt_path is not None:
        checkpoint_path = Path(init_ckpt_path).expanduser().resolve()
    elif stage == 1:
        checkpoint_path = _checkpoint_dir(output_dir, 0, lambda_1, lambda_2, 0, 0)
        checkpoint_path = checkpoint_path / f"checkpoint_{num_steps}"
    elif stage == 2:
        checkpoint_path = _checkpoint_dir(output_dir, 1, lambda_1, lambda_2, lambda_3, 0)
        checkpoint_path = checkpoint_path / f"checkpoint_{num_steps}"
    else:
        return None, None

    checkpoint = _restore_checkpoint(checkpoint_path)
    print(f"Initial shape from: {checkpoint_path}")

    return checkpoint, checkpoint_path


def run_sim(
    data_path: str,
    output_dir: str,
    shape: str,
    sdf_pretrain: str,
    lambda_1: float = 100000,
    lambda_2: float = 100000,
    lambda_3: float = 1,
    lambda_4: float = 10,
    stage: int = 0,
    seed: int = 1,
    num_steps: int = 10000,
    save_interval: int = 1000,
    learning_rate: float = 1e-3,
    init_ckpt_path: str | None = None,
):
    """Run one training stage and save periodic checkpoints."""
    if stage not in {0, 1, 2}:
        raise ValueError("stage must be 0, 1, or 2.")

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    if save_interval <= 0:
        raise ValueError("save_interval must be positive.")

    key = jax.random.PRNGKey(seed)
    checkpoint_dir = _checkpoint_dir(
        output_dir, stage, lambda_1, lambda_2, lambda_3, lambda_4
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    init_checkpoint, _ = _get_initial_checkpoint(
        output_dir=output_dir,
        stage=stage,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
        num_steps=num_steps,
        init_ckpt_path=init_ckpt_path,
    )

    edge_d, sign_d, phys_d, curv_d = _load_training_data(data_path, shape, stage)

    state, _ = create_train_state(
        key=key,
        lambda_data=lambda_1,
        lambda_phys=lambda_3,
        lambda_sign=lambda_2,
        lambda_curv=lambda_4,
        learning_rate=learning_rate,
        sdf_pretrain=None if sdf_pretrain == "checkpoint" else sdf_pretrain,
        init_checkpoint=init_checkpoint,
    )

    loss_weights = (
        state.lambda_data,
        state.lambda_sign,
        state.lambda_phys,
        state.lambda_curv,
    )
    print(
        f"[{shape}] pretrain={sdf_pretrain}, lambdas={loss_weights} "
        f"-> {checkpoint_dir}"
    )

    train_step = make_train_step(use_curvature_loss=(lambda_4 != 0))

    initial_losses = compute_initial_losses(state, edge_d, sign_d, phys_d, curv_d)
    initial_record = {
        "step": 0,
        "total_loss": pick(initial_losses, "total_loss", "total", "loss", default=0.0),
        "data_loss": pick(initial_losses, "data_loss", "data", default=0.0),
        "phys_loss": pick(initial_losses, "phys_loss", "phys", default=0.0),
        "sign_loss": pick(initial_losses, "sign_loss", "sign", default=0.0),
        "curv_loss": pick(initial_losses, "curv_loss", "curv", default=0.0),
    }

    save_ckpt(
        checkpoint_dir, step=0, state=state,
        loss_batch=loss_dict_to_batched([initial_record]),
        keep=NUM_CHECKPOINTS_TO_KEEP,
    )

    loss_buffer = []

    for step in trange(1, num_steps + 1, desc=f"Optimizing {shape}"):
        state, losses = train_step(state, edge_d, sign_d, phys_d, curv_d)

        loss_buffer.append({
            "step": step,
            "total_loss": float(losses["total_loss"]),
            "data_loss": float(losses["data_loss"]),
            "phys_loss": float(losses["phys_loss"]),
            "sign_loss": float(losses["sign_loss"]),
            "curv_loss": float(losses["curv_loss"]),
        })

        if step % save_interval == 0:
            save_ckpt(
                checkpoint_dir, step=step, state=state,
                loss_batch=loss_dict_to_batched(loss_buffer),
                keep=NUM_CHECKPOINTS_TO_KEEP,
            )
            loss_buffer.clear()

    if loss_buffer:
        save_ckpt(
            checkpoint_dir, step=num_steps, state=state,
            loss_batch=loss_dict_to_batched(loss_buffer),
            keep=NUM_CHECKPOINTS_TO_KEEP,
        )

    visualization_steps = tuple(
        step for step in DEFAULT_VISUALIZATION_STEPS if step <= num_steps
    )

    plot_result(
        data_path=data_path,
        checkpoint_dir=checkpoint_dir,
        shape=shape,
        steps_to_visualize=visualization_steps,
        save_interval=save_interval,
    )

    return checkpoint_dir


def build_parser():
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train, sample, and visualize the phase-field PINN."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run one training stage.")
    train_parser.add_argument("--data-path", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--shape", required=True)
    train_parser.add_argument("--sdf-pretrain", default="uniform")
    train_parser.add_argument("--lambda-1", type=float, default=100000)
    train_parser.add_argument("--lambda-2", type=float, default=100000)
    train_parser.add_argument("--lambda-3", type=float, default=1)
    train_parser.add_argument("--lambda-4", type=float, default=10)
    train_parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=0)
    train_parser.add_argument("--seed", type=int, default=1)
    train_parser.add_argument("--num-steps", type=int, default=10000)
    train_parser.add_argument("--save-interval", type=int, default=100)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--init-ckpt-path", default=None)

    sample_parser = subparsers.add_parser(
        "sample-curvature", help="Sample curvature-loss collocation points."
    )
    sample_parser.add_argument("--data-path", required=True)
    sample_parser.add_argument("--output-dir", required=True)
    sample_parser.add_argument("--shape", required=True)
    sample_parser.add_argument("--lambda-1", type=float, default=100000)
    sample_parser.add_argument("--lambda-2", type=float, default=10)
    sample_parser.add_argument("--lambda-3", type=float, default=100000)
    sample_parser.add_argument("--num-curv", type=int, default=5000)
    sample_parser.add_argument("--seed", type=int, default=0)
    sample_parser.add_argument("--step", type=int, default=10000)

    plot_parser = subparsers.add_parser("plot", help="Plot saved checkpoints.")
    plot_parser.add_argument("--data-path", required=True)
    plot_parser.add_argument("--checkpoint-dir", required=True)
    plot_parser.add_argument("--shape", required=True)
    plot_parser.add_argument(
        "--steps", type=int, nargs="+", default=list(DEFAULT_VISUALIZATION_STEPS)
    )
    plot_parser.add_argument("--save-interval", type=int, default=100)
    plot_parser.add_argument("--axis", choices=("x", "y", "z"), default="x")

    return parser


def main():
    """Run the selected command."""
    args = build_parser().parse_args()

    if args.command == "train":
        run_sim(
            data_path=args.data_path,
            output_dir=args.output_dir,
            shape=args.shape,
            sdf_pretrain=args.sdf_pretrain,
            lambda_1=args.lambda_1,
            lambda_2=args.lambda_2,
            lambda_3=args.lambda_3,
            lambda_4=args.lambda_4,
            stage=args.stage,
            seed=args.seed,
            num_steps=args.num_steps,
            save_interval=args.save_interval,
            learning_rate=args.learning_rate,
            init_ckpt_path=args.init_ckpt_path,
        )

    elif args.command == "sample-curvature":
        sample_curv_points(
            data_path=args.data_path,
            output_dir=args.output_dir,
            shape=args.shape,
            lambda_1=args.lambda_1,
            lambda_2=args.lambda_2,
            lambda_3=args.lambda_3,
            num_curv=args.num_curv,
            seed=args.seed,
            step=args.step,
        )

    elif args.command == "plot":
        plot_result(
            data_path=args.data_path,
            checkpoint_dir=args.checkpoint_dir,
            shape=args.shape,
            steps_to_visualize=tuple(args.steps),
            save_interval=args.save_interval,
            axis=args.axis,
        )

if __name__ == "__main__":
    main()

