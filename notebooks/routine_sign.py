import sys
sys.path.append('../src')

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import jax
import jax.numpy as jnp

# Configure JAX to use only the CPU
# jax.config.update('jax_platform_name', 'cpu') 

print('jax:', jax.__version__)      # 0.7.2
print('devices:', jax.devices())    # [CudaDevice(id=0)]

import jax.experimental.layout as _layout
if not hasattr(_layout, 'DeviceLocalLayout'):
        # Provide a minimal alias so orbax can import function annotations\n",
        _layout.DeviceLocalLayout = _layout.Layout
        print('Patched jax.experimental.layout.DeviceLocalLayout ->', getattr(_layout, 'DeviceLocalLayout'))
import os
import numpy as np
import matplotlib.pyplot as plt
from pinn.train import create_train_state, train_step
from pinn.cryoet_io import load_mrc_data
import pickle
from flax.training import checkpoints
from pinn.utils import initial_loss
from tqdm.notebook import trange

from pinn.utils import sample_sign_points, save_sign_data, load_sign_data, load_edge_data

# Utility function to convert GPU/JAX arrays to CPU/NumPy arrays
def to_numpy(tree):
    return jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x)) if isinstance(x, (jnp.ndarray, np.ndarray)) else x,
        tree,
    )
def as_f32_scalar(x):
    # Make sure to extract as a scalar (float32) even if it's an array/DeviceArray
    return np.asarray(x, dtype=np.float32).reshape(()).item()



GRID_SIZE = 128                  # cryo-ET data will be resampled to GRID_SIZE^3 (64 * 64 * 64)
NUM_CHECKPOINTS_TO_KEEP = 1000  # Checkpoint retention count (older ones get removed)

# Parameters
lambda_1 = 1000000       # Weight for boundary loss (data loss)
lambda_2 = 0             # Weight for physics loss
lambda_3 = 0             # Weight for sign loss
shape = "chri_1"
sdf_pretrain="uniform"
num_collocation = 10000
num_steps = 10000  # 10000
save_interval = 100 # 100
threshold = 0.8
learning_rate = 1e-3  #1e-3
radius = 0.4

# Load initial checkpoint data
# init_ckpt_dir= os.path.abspath(f"../outputs/logs/{shape}/lambda_{lambda_1}_0_100000")
# init_ckpt_path = os.path.join(init_ckpt_dir, f"checkpoint_10000")
# init_ckpt = checkpoints.restore_checkpoint(ckpt_dir=init_ckpt_path, target=None)

# Initialize JAX random key
key = jax.random.PRNGKey(0)

### Load Edges ###
edge_data = f"../data/experimental/edge/{shape}.npz"
edges = load_edge_data(edge_data)

### Sample Signs ###
sign_data = f"../data/experimental/sign/{shape}.npz"
# sign_data = f"../data/experimental/sign/{shape}/manual_signs.npz"
signs = load_sign_data(sign_data)


### Create train state ###
state, model = create_train_state(key, lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3, threshold = threshold, learning_rate=learning_rate, sdf_pretrain=sdf_pretrain)
# state, model = create_train_state(key, lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3, init_ckpt=init_ckpt, learning_rate=learning_rate)
print(f"Using lambda_1: {state.lambda_1}, lambda_2: {state.lambda_2}, lambda_3: {state.lambda_3}, threshold: {state.threshold}")


# x_train = (jax.random.uniform(key, (num_collocation, 3), minval=-1, maxval=1)) # Sampled from [-1, 1]^3

# For thin volume in z direction
Z = 256
slice_min = 134
slice_max = 164
zmin_norm = 2 * slice_min / (Z - 1) - 1
zmax_norm = 2 * slice_max / (Z - 1) - 1
key_xy, key_z = jax.random.split(key)
xy = jax.random.uniform(key_xy, (num_collocation, 2), minval=-1.0, maxval=1.0) # x,y ∈ [-1, 1]
z = jax.random.uniform(key_z, (num_collocation, 1), minval=zmin_norm, maxval=zmax_norm) # z ∈ [zmin_norm, zmax_norm]
x_train = jnp.concatenate([xy, z], axis=1)

checkpoint_dir = os.path.abspath(f"../outputs/logs/{shape}/lambda_{lambda_1}_{lambda_2}_{lambda_3}")
# checkpoint_dir = os.path.abspath(f"../outputs/logs/{shape}/lambda_{lambda_1}_{lambda_2}_{lambda_3}_cont")

# --- keep the same schema for step 0 ---
# init_d = initial_loss(state, x_train, cryoET_data, signs)     # Get the dictionary of initial losses
init_d = initial_loss(state, x_train, edges, signs)     # Get the dictionary of initial losses
init_total = init_d.get("total_loss", init_d.get("total", init_d.get("loss", 0.0)))
init_data  = init_d.get("data_loss",  init_d.get("data",  0.0))
init_phys  = init_d.get("physics_loss", init_d.get("phys", init_d.get("physics", 0.0)))
init_sign  = init_d.get("sign_loss",  init_d.get("sign",  0.0))

payload0 = {
    "state": to_numpy(state),
    "loss": {
        "step":         np.asarray([0], dtype=np.int64),
        "total_loss":   np.asarray([as_f32_scalar(init_total)], dtype=np.float32),
        "data_loss":    np.asarray([as_f32_scalar(init_data)],  dtype=np.float32),
        "physics_loss": np.asarray([as_f32_scalar(init_phys)],  dtype=np.float32),
        "sign_loss":    np.asarray([as_f32_scalar(init_sign)],  dtype=np.float32),
    }
}

checkpoints.save_checkpoint(
    ckpt_dir=checkpoint_dir,
    target=payload0,
    step=0,
    overwrite=False,
    keep=NUM_CHECKPOINTS_TO_KEEP,
)

loss_history = []

# for step in range(num_steps):
for step in trange(num_steps):
    # state, loss_val, loss_data_val, loss_physics_val, loss_sign_val = train_step(state, x_train, cryoET_data, signs)
    state, loss_val, loss_data_val, loss_physics_val, loss_sign_val = train_step(state, x_train, edges, signs)

    loss_history.append({
    "step":  step + 1,
    "total_loss":   loss_val,
    "data_loss":    loss_data_val,
    "physics_loss": loss_physics_val,
    "sign_loss": loss_sign_val,
    })

    if (step + 1) % save_interval == 0:

        # print("step = ", step)

        batched_loss = {
        "step":         np.asarray([e["step"] for e in loss_history], dtype=np.int64),
        "total_loss":   np.asarray([as_f32_scalar(e["total_loss"])   for e in loss_history], dtype=np.float32),
        "data_loss":    np.asarray([as_f32_scalar(e["data_loss"])    for e in loss_history], dtype=np.float32),
        "physics_loss": np.asarray([as_f32_scalar(e["physics_loss"]) for e in loss_history], dtype=np.float32),
        "sign_loss":    np.asarray([as_f32_scalar(e["sign_loss"])    for e in loss_history], dtype=np.float32),
        }

        payload = {
            "state": to_numpy(state),  # 항상 CPU NumPy로
            "loss":  batched_loss
        }

        checkpoints.save_checkpoint(
            ckpt_dir=checkpoint_dir,
            target=payload,
            step=step + 1,
            overwrite=False,
            keep=NUM_CHECKPOINTS_TO_KEEP,
        )
        # print(f"Checkpoint saved at step {step+1}")
        loss_history = []



# ### PLOT AND SAVE FIGURES ###
# ## make one combined image file

# slice_index = 32

# import matplotlib.pyplot as plt
# from flax.training import checkpoints
# from pinn.cryoet_io import load_mrc_data
# from pinn.model import PINN
# from pinn.plot import (
#     visualize_checkpoint_result,
#     visualize_cryoET_with_contours,
#     plot_3d_isosurface,
#     plot_loss_history_ax,
#     plot_normalized_loss_history_ax,
#     visualize_physics_loss
# )
# from pinn.utils import assemble_loss_history

# steps_to_visualize = [0, 1000, 5000, 10000]
# steps_series = list(range(0, 10001, 100))

# # Load checkpoint data
# # checkpoint_dir = os.path.abspath(f"../outputs/logs/{shape}/lambda_{lambda_1}_{lambda_2}")
# checkpoint_dir = os.path.abspath(f"../outputs/logs/{shape}/lambda_{lambda_1}_{lambda_2}_{lambda_3}")
# print(checkpoint_dir)
# checkpoint_data = {}
# for step in steps_series:
#     checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
#     checkpoint_data[step] = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)

# # Load CryoET data
# mrc_file_path = f"../data/experimental/masked/{shape}.mrc"
# cryoET_data = load_mrc_data(mrc_file_path, grid_size=GRID_SIZE)

# fig = plt.figure(figsize=(3*4, 12))

# # Row 1: visualize_cryoET_with_contours
# for i, step in enumerate(steps_to_visualize):
#     ax = fig.add_subplot(4, 4, i + 1)
#     # visualize_cryoET_with_contours(ax, step, checkpoint_data[step], cryoET_data, grid_size=GRID_SIZE, slice_index=GRID_SIZE//2, axis="z")
#     visualize_cryoET_with_contours(ax, step, checkpoint_data[step], cryoET_data, grid_size=GRID_SIZE, slice_index=slice_index, axis="x", thresholding=False)
#     if i == 0:
#         ax.set_ylabel("CryoET + Contours")

# # Row 2: visualize_checkpoint_result
# for i, step in enumerate(steps_to_visualize):
#     ax = fig.add_subplot(4, 4, 4 + i + 1)
#     # visualize_checkpoint_result(ax, step, checkpoint_data[step], grid_size=GRID_SIZE, slice_index=GRID_SIZE//2, axis="z", show_contour=False)
#     # visualize_checkpoint_result(ax, step, checkpoint_data[step], grid_size=GRID_SIZE, slice_index=36, axis="x", show_contour=False)
#     visualize_physics_loss(ax, epsilon=0.05, checkpoint=checkpoint_data[step], component="phi",
#                                    # grid_size=GRID_SIZE, slice_index=GRID_SIZE//2, axis="z", no_label=True, vmin=-1, vmax=1)
#                                grid_size=GRID_SIZE, slice_index=slice_index, axis="x", no_label=True, vmin=-1, vmax=1)
#     if i == 0:
#         ax.set_ylabel("Phi")

# # Row 3: plot_3d_isosurface
# for i, step in enumerate(steps_to_visualize):

#     # ax = fig.add_subplot(4, 4, 8 + i + 1, projection='3d')
#     # plot_3d_isosurface(ax, step, checkpoint_data[step], grid_size=GRID_SIZE)
#     # if i == 0:
#     #     ax.set_zlabel("3D Isosurface", rotation=180)

#     ax = fig.add_subplot(4, 4, 8 + i + 1)
#     visualize_physics_loss(ax, epsilon=0.05, checkpoint=checkpoint_data[step], component="tension",
#                                # grid_size=GRID_SIZE, slice_index=GRID_SIZE//2, axis="z", no_label=True, vmin=0, vmax=400)
#                                grid_size=GRID_SIZE, slice_index=slice_index, axis="x", no_label=True, vmin=0, vmax=400)
#     if i == 0:
#         ax.set_ylabel("Tension energy")

# # Row 4: loss history
# assembled_loss = assemble_loss_history(checkpoint_data)
# ax = fig.add_subplot(4, 4, 12+1)
# plot_loss_history_ax(ax, assembled_loss)
# for i in [0,1,2]:
#     ax = fig.add_subplot(4, 4, 12+i+2)
#     if i == 2:
#         j = 3
#     else:
#         j=i
#     plot_normalized_loss_history_ax(ax, j, assembled_loss)
    

# plt.tight_layout()
# plt.savefig(f"../outputs/logs/{shape}/lambda_{lambda_1}_{lambda_2}_{lambda_3}/analysis.png", dpi=300, bbox_inches="tight")




# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from flax.training import checkpoints
# from pinn.cryoet_io import load_mrc_data
# from pinn.plot import visualize_cryoET_with_contours, visualize_physics_loss
# from data_generation.mesh_to_cryoet import plot_single_slice
# import mrcfile
# import jax
# import jax.numpy as jnp


# slice_list = np.arange(0, 64, 10)
# slice_max = 64
# step = 10000
# epsilon = 0.05
# axis = "x"
# # axis = "z"

# # Load input data
# # input_file = f"../data/experimental/downsampled/{shape}.mrc"
# # input_file = f"../data/experimental/masked/{shape}.mrc"
# input_file = f"../data/experimental/sam3/mrc/{shape}.mrc"
# if os.path.exists(input_file):
#     volume = load_mrc_data(input_file, grid_size=GRID_SIZE)
#     cryo_data = volume

# # Load pinn result
# # checkpoint_dir = os.path.abspath(f"../outputs/logs/{shape}/lambda_{lambda_1}_{lambda_2}_{lambda_3}")
# checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
# pinn_data = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)
        
# # Plot 
# n = len(slice_list)    # slice columns
# m = 3                  # rows

# fig = plt.figure(figsize=(2*n, 2*m))

# for j, slice_id in enumerate(slice_list): # columns
#     ax = fig.add_subplot(m, n, 0 * n + j + 1)
#     visualize_cryoET_with_contours(ax, step, pinn_data, cryo_data, grid_size=GRID_SIZE, 
#                                    slice_index=slice_id, axis=axis, no_label=True, thresholding=False)

#     # --- column label ---
#     ax.set_title(f"{slice_id}/{slice_max}", fontsize=16)

# for j, slice_id in enumerate(slice_list): # columns
#     ax = fig.add_subplot(m, n, 1 * n + j + 1)
#     visualize_physics_loss(ax, epsilon=epsilon, checkpoint=pinn_data, component="phi",
#                            grid_size=GRID_SIZE, slice_index=slice_id, axis=axis, no_label=True, vmin=-1, vmax=1)

# for j, slice_id in enumerate(slice_list): # columns
#     ax = fig.add_subplot(m, n, 2 * n + j + 1)
#     visualize_physics_loss(ax, epsilon=epsilon, checkpoint=pinn_data, component="tension",
#                                grid_size=GRID_SIZE, slice_index=slice_id, axis=axis, no_label=True, vmin=0, vmax=400)

    
# plt.tight_layout()
# plt.savefig(f"../outputs/logs/{shape}/lambda_{lambda_1}_{lambda_2}_{lambda_3}/slice.png", dpi=600, bbox_inches="tight")

