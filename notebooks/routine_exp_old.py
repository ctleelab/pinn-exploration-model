import sys
sys.path.append('../src')

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import os
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from flax.training import checkpoints
from tqdm.notebook import trange

from pinn.train import create_train_state, train_step_sched, train_step_nosched
from pinn.utils import (
    load_pts_data, initial_loss, strip_meta, loss_dict_to_batched,
    save_ckpt, pick
)

from pinn.cryoet_io import load_mrc_data
from pinn.plot import (
    visualize_cryoET_with_contours,
    visualize_phase,
    plot_loss_history_ax,
    plot_normalized_loss_history_ax,
)
from pinn.utils import assemble_loss_history

import jax.experimental.layout as _layout
if not hasattr(_layout, 'DeviceLocalLayout'):
    _layout.DeviceLocalLayout = _layout.Layout
    print('Patched jax.experimental.layout.DeviceLocalLayout ->', getattr(_layout, 'DeviceLocalLayout'))

NUM_CHECKPOINTS_TO_KEEP = 1000



def plot_result(
    checkpoint_dir, 
    shape, 
    grid_size=64, 
    steps_to_visualize=(0, 1000, 5000, 10000), 
    save_interval=100
):
    slice_index = grid_size // 2
    axis = "x"

    # Decide which checkpoints to load (based on your save interval)
    max_step = max(steps_to_visualize)
    steps_series = list(range(0, max_step + 1, save_interval))

    # ---- Load checkpoints ----
    checkpoint_data = {}
    for step in steps_series:
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
        if os.path.exists(ckpt_path):
            checkpoint_data[step] = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None)

    # Ensure the requested steps exist (otherwise plotting will crash / skip)
    steps_to_visualize = [s for s in steps_to_visualize if s in checkpoint_data]
    if len(steps_to_visualize) == 0:
        print(f"[plot_result] No checkpoints found in {checkpoint_dir}")
        return

    # ---- Load cryoET volume ----
    mrc_file_path = f"../data/experimental/masked/{shape}.mrc"
    cryoET_data = load_mrc_data(mrc_file_path, grid_size=grid_size)

    # ---- Make figure ----
    fig = plt.figure(figsize=(3 * 4, 12))

    # Row 1: CryoET + contours
    for i, step in enumerate(steps_to_visualize):
        ax = fig.add_subplot(4, 4, i + 1)
        visualize_cryoET_with_contours(
            ax, step, checkpoint_data[step], cryoET_data,
            grid_size=grid_size, slice_index=slice_index, axis=axis,
            thresholding=False
        )
        if i == 0:
            ax.set_ylabel("CryoET + Contours")

    # Row 2: phi
    for i, step in enumerate(steps_to_visualize):
        ax = fig.add_subplot(4, 4, 4 + i + 1)
        visualize_phase(
            ax, epsilon=0.05, checkpoint=checkpoint_data[step], component="phi",
            grid_size=grid_size, slice_index=slice_index, axis=axis,
            no_label=True, vmin=-1, vmax=1
        )
        if i == 0:
            ax.set_ylabel("Phi")

    # Row 3: tension
    for i, step in enumerate(steps_to_visualize):
        ax = fig.add_subplot(4, 4, 8 + i + 1)
        visualize_phase(
            ax, epsilon=0.05, checkpoint=checkpoint_data[step], component="tension",
            grid_size=grid_size, slice_index=slice_index, axis=axis,
            no_label=True, vmin=0.0, vmax=1.0
        )
        if i == 0:
            ax.set_ylabel("Tension energy")

    # Row 4: loss history
    assembled_loss = assemble_loss_history(checkpoint_data)
    ax = fig.add_subplot(4, 4, 12 + 1)
    plot_loss_history_ax(ax, assembled_loss)

    for i in [0, 1, 2]:
        ax = fig.add_subplot(4, 4, 12 + i + 2)
        j = 3 if i == 2 else i
        plot_normalized_loss_history_ax(ax, j, assembled_loss)

    plt.tight_layout()
    out_png = os.path.join(checkpoint_dir, "result.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    # plt.show()
    print("Saved:", out_png)



def run_one(
    shape: str,
    lambda_2: float,
    sdf_pretrain: str,
    lambda_1: float = 100000,
    lambda_3: float = 100000,
    seed: int = 1,
    num_colloc: int = 40000,
    num_steps: int = 10000,
    save_interval: int = 100,
    learning_rate: float = 1e-3,
    warmup_steps: int = 2000,
    schedule: bool = False,
    init_ckpt_path: str | None = None,
):
    key = jax.random.PRNGKey(seed)

    root_dir = f"../outputs/logs/jrc_cos7-1b/{shape}"

    # Where to SAVE this run
    if sdf_pretrain == "checkpoint":
        checkpoint_dir = os.path.abspath(f"{root_dir}/lambda_{lambda_1}_{lambda_2}_{lambda_3}_cont")
    else:
        checkpoint_dir = os.path.abspath(f"{root_dir}/lambda_{lambda_1}_{lambda_2}_{lambda_3}")

    # Optional: where to LOAD initial state
    init_ckpt = None
    if sdf_pretrain == "checkpoint":
        init_ckpt_path = f"{root_dir}/lambda_{lambda_1}_0_{lambda_3}/checkpoint_10000"
        init_ckpt_path = os.path.abspath(init_ckpt_path)
        init_ckpt = checkpoints.restore_checkpoint(ckpt_dir=init_ckpt_path, target=None)
        print("Initial shape from:", init_ckpt_path)

    # ===== LOAD POINT DATA =====
    pts_path = f"../data/experimental"
    edge = load_pts_data(f"{pts_path}/edge/jrc_cos7-1b/{shape}.npz", perm=(2,1,0))
    # sign = load_pts_data(f"{pts_path}/sign/{shape}/manual_signs.npz", perm=(2,1,0))
    sign = load_pts_data(f"{pts_path}/sign/jrc_cos7-1b/{shape}.npz", perm=(2,1,0))
    phys = load_pts_data(f"{pts_path}/phys/p_uniform_{num_colloc}.npz", perm=(2,1,0))

    edge_d = jax.device_put(strip_meta(edge))
    sign_d = jax.device_put(strip_meta(sign))
    phys_d = jax.device_put(strip_meta(phys))

    # ===== CREATE TRAIN STATE =====
    if sdf_pretrain == "checkpoint":
        state, model = create_train_state(
            key, lambda_1, lambda_2, lambda_3,
            learning_rate, warmup_steps,
            init_ckpt=init_ckpt
        )
    else:
        state, model = create_train_state(
            key, lambda_1, lambda_2, lambda_3,
            learning_rate, warmup_steps,
            sdf_pretrain=sdf_pretrain
        )

    print(f"[{shape}] pretrain={sdf_pretrain} lambdas=({state.lambda_1},{state.lambda_2},{state.lambda_3}) -> {checkpoint_dir}")

    if schedule:
        train_step_jit = jax.jit(train_step_sched, donate_argnums=(0,))
    else:
        train_step_jit = jax.jit(train_step_nosched, donate_argnums=(0,))

    # ===== INITIAL LOSS + CKPT(step=0) =====
    init_d = initial_loss(state, edge_d, sign_d, phys_d)
    init_record = {
        "step": 0,
        "total_loss": pick(init_d, "total_loss", "total", "loss", default=0.0),
        "data_loss":  pick(init_d, "data_loss", "data", default=0.0),
        "phys_loss":  pick(init_d, "phys_loss", "phys", default=0.0),
        "sign_loss":  pick(init_d, "sign_loss", "sign", default=0.0),
    }
    save_ckpt(
        checkpoint_dir, step=0, state=state,
        loss_batch=loss_dict_to_batched([init_record]),
        keep=NUM_CHECKPOINTS_TO_KEEP,
    )

    # ===== TRAIN LOOP =====
    buffer = []
    for step in trange(1, num_steps + 1):
        state, total, ld, lp, ls = train_step_jit(state, step, edge_d, sign_d, phys_d)
        buffer.append({"step": step, "total_loss": total, "data_loss": ld, "phys_loss": lp, "sign_loss": ls})

        if step % save_interval == 0:
            save_ckpt(
                checkpoint_dir, step=step, state=state,
                loss_batch=loss_dict_to_batched(buffer),
                keep=NUM_CHECKPOINTS_TO_KEEP,
            )
            buffer.clear()

    # Return the path to the final checkpoint for chaining
    final_ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{num_steps}")

    plot_result(
        checkpoint_dir=checkpoint_dir,
        shape=shape,
        grid_size=64,
        steps_to_visualize=(0, 1000, 5000, 10000),
        save_interval=save_interval,
    )

    return


# =========================
# Two-stage experiment plan
# =========================
for shape in ["bud-1"]:
    lambda_1 = 100000
    lambda_3 = 100000
    num_colloc =80000

    # Stage 1: lambda_2 = 0, uniform pretrain
    run_one(
        shape=shape,
        lambda_2=0,
        sdf_pretrain="uniform",
        lambda_1=lambda_1,
        lambda_3=lambda_3,
        num_colloc=num_colloc,
        num_steps=10000,
    )

    # Stage 2: lambda_2 in [0,1,10], checkpoint pretrain from stage1_final
    for lambda_2 in [10, 1, 0]:
        run_one(
            shape=shape,
            lambda_2=lambda_2,
            sdf_pretrain="checkpoint",
            lambda_1=lambda_1,
            lambda_3=lambda_3,
            num_colloc=num_colloc,
            num_steps=10000,
        )

