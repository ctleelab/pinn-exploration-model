import sys
sys.path.append('../src')

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import jax
import jax.numpy as jnp

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from flax.training import checkpoints
from pinn.train import create_train_state, train_step_sched, train_step_nosched
from pinn.utils import load_edge_data, load_pts_data, load_sign_data, initial_loss, strip_meta, loss_dict_to_batched, save_ckpt, pick
from tqdm.notebook import trange

# Configure JAX to use only the CPU
# jax.config.update('jax_platform_name', 'cpu') 

print('jax:', jax.__version__)      # 0.7.2
print('devices:', jax.devices())    # [CudaDevice(id=0)]

import jax.experimental.layout as _layout
if not hasattr(_layout, 'DeviceLocalLayout'):
        # Provide a minimal alias so orbax can import function annotations\n",
        _layout.DeviceLocalLayout = _layout.Layout
        print('Patched jax.experimental.layout.DeviceLocalLayout ->', getattr(_layout, 'DeviceLocalLayout'))

NUM_CHECKPOINTS_TO_KEEP = 1000   # Checkpoint retention count (older ones get removed)


for shape in ["czii_gl_1"]:
    # for lambda_2 in [10, 0]:
    for lambda_2 in [1]:

        ## ===== PARAMETERS ========
        lambda_1 = 100000       # Weight for boundary loss (data loss)
        # lambda_2 = 0            # Weight for physics loss
        lambda_3 = 100000       # Weight for sign loss
        # shape = "multi"
        seed = 1
        num_colloc = 40000

        # sdf_pretrain="uniform"
        sdf_pretrain="checkpoint"
        num_steps = 10000
        save_interval = 100
        learning_rate = 1e-3
        warmup_steps  = 2000
        # schedule = True
        schedule = False

        key = jax.random.PRNGKey(seed)
        root_dir = f"../outputs/logs/{shape}"
        if sdf_pretrain == "checkpoint":
            checkpoint_dir = os.path.abspath(f"{root_dir}/lambda_{lambda_1}_{lambda_2}_{lambda_3}_cont")
        else:
            checkpoint_dir = os.path.abspath(f"{root_dir}/lambda_{lambda_1}_{lambda_2}_{lambda_3}")

        ## ===== LOAD INITIAL SHAPE FROM CHECKPOINT ========
        if sdf_pretrain == "checkpoint":
            init_ckpt_path = os.path.abspath(f"{root_dir}/lambda_{lambda_1}_0_{lambda_3}/checkpoint_10000")
            # init_ckpt_path = os.path.abspath(f"{root_dir}_v1/lambda_{lambda_1}_0_{lambda_3}/checkpoint_10000")
            # init_ckpt_path = os.path.abspath(f"{root_dir}_v2/lambda_100000_0_100000/checkpoint_10000")
            init_ckpt = checkpoints.restore_checkpoint(ckpt_dir=init_ckpt_path, target=None)
            print("Ininitial shape: ", init_ckpt_path)

        ## ===== LOAD POINT DATA (EDGE, SIGN, and PHYS) ========
        pts_path = f"../data/experimental"
        edge = load_pts_data(f"{pts_path}/edge/{shape}.npz", perm=(2,1,0))
        sign = load_pts_data(f"{pts_path}/sign/{shape}/manual_signs.npz", perm=(2,1,0))
        phys = load_pts_data(f"{pts_path}/phys/p_uniform_{num_colloc}.npz", perm=(2,1,0))

        # Put constant inputs on device once (avoid host->device every step)
        phys_d = jax.device_put(strip_meta(phys))
        edge_d = jax.device_put(strip_meta(edge))
        sign_d = jax.device_put(strip_meta(sign))

        ## ===== CREATE TRAIN STATE ========
        if sdf_pretrain == "checkpoint":
            state, model = create_train_state(key, lambda_1, lambda_2, lambda_3, learning_rate, warmup_steps, init_ckpt=init_ckpt)
        else:
            state, model = create_train_state(key, lambda_1, lambda_2, lambda_3, learning_rate, warmup_steps, sdf_pretrain=sdf_pretrain)
        print(f"Using lambda_1: {state.lambda_1}, lambda_2: {state.lambda_2}, lambda_3: {state.lambda_3}, learning_rate: {state.learning_rate}, warmup_steps: {state.warmup_steps}")

        # Optionally: donate state to reduce peak memory
        if schedule:
            train_step_jit = jax.jit(train_step_sched, donate_argnums=(0,))
        else:
            train_step_jit = jax.jit(train_step_nosched, donate_argnums=(0,))

        # ===== INITIAL LOSS + CHECKPOINT (step=0) =====
        init_d = initial_loss(state, edge_d, sign_d, phys_d)
        init_record = {
            "step": 0,
            "total_loss":   pick(init_d, "total_loss", "total", "loss", default=0.0),
            "data_loss":    pick(init_d, "data_loss", "data", default=0.0),
            "phys_loss":    pick(init_d, "phys_loss", "phys", default=0.0),
            "sign_loss":    pick(init_d, "sign_loss", "sign", default=0.0),
        }
        save_ckpt(
            checkpoint_dir,
            step=0,
            state=state,
            loss_batch=loss_dict_to_batched([init_record]),
            keep=NUM_CHECKPOINTS_TO_KEEP,
        )

        # ===== TRAIN LOOP =====
        buffer = []  # accumulate losses until save_interval

        for step in trange(1, num_steps + 1):
            state, total, ld, lp, ls = train_step_jit(state, step, edge_d, sign_d, phys_d)

            buffer.append({
                "step": step,
                "total_loss": total,
                "data_loss": ld,
                "phys_loss": lp,
                "sign_loss": ls,
            })

            if step % save_interval == 0:
                save_ckpt(
                    checkpoint_dir,
                    step=step,
                    state=state,
                    loss_batch=loss_dict_to_batched(buffer),
                    keep=NUM_CHECKPOINTS_TO_KEEP,
                )
                buffer.clear()


        # ===== PLOT RESULTS =====
        import matplotlib.pyplot as plt
        from pinn.cryoet_io import load_mrc_data
        from pinn.plot import (
            visualize_cryoET_with_contours,
            visualize_phase,
            # plot_3d_isosurface,
            plot_loss_history_ax,
            plot_normalized_loss_history_ax,
        )
        from pinn.utils import assemble_loss_history

        GRID_SIZE = 64
        slice_index = GRID_SIZE //2
        axis = "x"

        steps_to_visualize = [0, 1000, 5000, 10000]
        steps_series = list(range(0, 10001, 100))

        ## ===== LOAD CHECKPOINT ========
        checkpoint_data = {}
        for step in steps_series:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
            checkpoint_data[step] = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)

        ## ===== LOAD CRYOET ========
        # mrc_file_path = f"../data/experimental/downsampled/{shape}.mrc"
        mrc_file_path = f"../data/experimental/masked/{shape}.mrc"
        cryoET_data = load_mrc_data(mrc_file_path, grid_size=GRID_SIZE)

        ## ===== MAKE FIGURE ========
        fig = plt.figure(figsize=(3*4, 12))

        # Row 1: visualize_cryoET_with_contours
        for i, step in enumerate(steps_to_visualize):
            ax = fig.add_subplot(4, 4, i + 1)
            visualize_cryoET_with_contours(ax, step, checkpoint_data[step], cryoET_data, grid_size=GRID_SIZE, slice_index=slice_index, axis=axis, thresholding=False)
            if i == 0:
                ax.set_ylabel("CryoET + Contours")

        # Row 2: visualize_checkpoint_result
        for i, step in enumerate(steps_to_visualize):
            ax = fig.add_subplot(4, 4, 4 + i + 1)
            visualize_phase(ax, epsilon=0.05, checkpoint=checkpoint_data[step], component="phi",
                            grid_size=GRID_SIZE, slice_index=slice_index, axis=axis, no_label=True, vmin=-1, vmax=1)
            if i == 0:
                ax.set_ylabel("Phi")

        # Row 3: plot_3d_isosurface 
        for i, step in enumerate(steps_to_visualize):

            # ax = fig.add_subplot(4, 4, 8 + i + 1, projection='3d')
            # plot_3d_isosurface(ax, step, checkpoint_data[step], grid_size=GRID_SIZE)
            # if i == 0:
            #     ax.set_zlabel("3D Isosurface", rotation=180)

            ax = fig.add_subplot(4, 4, 8 + i + 1)
            visualize_phase(ax, epsilon=0.05, checkpoint=checkpoint_data[step], component="tension",
                            grid_size=GRID_SIZE, slice_index=slice_index, axis=axis, no_label=True, vmin=0.0, vmax=1.0)
            if i == 0:
                ax.set_ylabel("Tension energy")

        # Row 4: loss history
        assembled_loss = assemble_loss_history(checkpoint_data)
        ax = fig.add_subplot(4, 4, 12+1)
        plot_loss_history_ax(ax, assembled_loss)
        for i in [0,1,2]:
            ax = fig.add_subplot(4, 4, 12+i+2)
            if i == 2:
                j = 3
            else:
                j=i
            plot_normalized_loss_history_ax(ax, j, assembled_loss)

        plt.tight_layout()
        plt.savefig(f"{checkpoint_dir}/result.png", dpi=300, bbox_inches="tight")
        plt.show()
