# %% [markdown]
# ## Main Calculation
# Checkpoint results are stored in `outputs/logs/` at regular intervals, `save_interval`.
# 

# %%
import sys

# %%
import jax
import jax.numpy as jnp
import os
import numpy as np
import matplotlib.pyplot as plt
from pinn.train import create_train_state, train_step
from pinn.cryoet_io import load_mrc_data
import pickle
from flax.training import checkpoints
from pinn.utils import initial_loss
from tqdm.notebook import trange

GRID_SIZE = 64
NUM_CHECKPOINTS_TO_KEEP = 1000  # Checkpoint retention count (older ones get removed)

# Set loss function weights
lambda_1 = 1000000      # Weight for boundary loss
lambda_2 = 0            # Weight for physics loss
shape = "bud_04"
sdf_pretrain="sphere"   # sphere or plane

# Initialize JAX random key
key = jax.random.PRNGKey(0)
# 
# Set directory for initial condition
# ckpt_dir = os.path.abspath(f"../outputs/logs/test_phase/snwitch2-2/30000/default")
# ckpt_dir = os.path.abspath(f"../outputs/logs/biconcave/round_1/lambda_5000000_1e-05/checkpoint_30000/")
# init_ckpt = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)

# Create train state
# state, model = create_train_state(key, lambda_1=lambda_1, lambda_2=lambda_2, learning_rate=1e-3, init_ckpt=init_ckpt)
state, model = create_train_state(key, lambda_1=lambda_1, lambda_2=lambda_2, sdf_pretrain=sdf_pretrain, learning_rate=1e-3)
print(f"Using lambda_1: {state.lambda_1}, lambda_2: {state.lambda_2}")

#### LOAD MRC DATA #####
mrc_file_path = f"../data/synthetic/{shape}.mrc"
# missing_ratio=0.998
# mrc_file_path = f"../data/synthetic/biconcave_fragment_{str(missing_ratio).replace('.', '')}.mrc"
# flip_ratio=0.2
# mrc_file_path = f"../data/synthetic/biconcave_noisy_{str(flip_ratio).replace('.', '')}.mrc"
# sigma_blur = 5
# mrc_file_path = f"../data/synthetic/biconcave_blur_{str(sigma_blur).replace('.', '')}.mrc"
# num_patch=10
# rad_patch=25
# mrc_file_path = f"../data/synthetic/biconcave_patch_{num_patch}_{rad_patch}.mrc"

flip_ratio=0.2
# wedge_axis='y'
# mrc_file_path = f"../data/synthetic/flip_noise/{shape}_f{str(flip_ratio).replace('.', '')}.mrc"
# mrc_file_path = f"../data/synthetic/flip_noise/{shape}_f{str(flip_ratio).replace('.', '')}_w{wedge_axis}.mrc"

cryoET_data = load_mrc_data(mrc_file_path, grid_size=GRID_SIZE)
print("MRC data loaded successfully! Shape:", cryoET_data.shape)
print("Cryo-ET Data Shape:", cryoET_data.shape)
print("Cryo-ET Data Min:", cryoET_data.min())
print("Cryo-ET Data Max:", cryoET_data.max())
print("Cryo-ET Unique Values:", jnp.unique(cryoET_data))

# Training Data
num_collocation = 10000
x_train = (jax.random.uniform(key, (num_collocation, 3), minval=-1, maxval=1)) # Sampled from [-1, 1]^3


# Training loop
num_steps = 10000
# num_steps = 1000
save_interval = 100


# # temporary 
# binary_mask = jnp.where(cryoET_data > 0.8, 1, 0)
# membrane_indices = jnp.where(binary_mask.ravel() == 1)[0]

checkpoint_dir = os.path.abspath(f"../outputs/logs/{shape}/lambda_{lambda_1}_{lambda_2}")

checkpoints.save_checkpoint(
    ckpt_dir=checkpoint_dir,
    target={"state": state, "loss": initial_loss(state, x_train, cryoET_data)},
    # target={"state": state, "loss": initial_loss(state, x_train, cryoET_data, membrane_indices)},
    step=0,
    overwrite=False,
    keep=NUM_CHECKPOINTS_TO_KEEP,
)
loss_history = []


# for step in range(num_steps):
for step in trange(num_steps):
    state, loss_val, loss_data_val, loss_physics_val = train_step(state, x_train, cryoET_data)
    # state, loss_val, loss_data_val, loss_physics_val = train_step(state, x_train, cryoET_data, membrane_indices)

    loss_history.append({
        "step": step + 1,
        "total_loss": loss_val,
        "data_loss": loss_data_val,
        "physics_loss": loss_physics_val
    })

    if (step+1) % save_interval == 0:

        loss_history = {key: np.array([entry[key] for entry in loss_history]) for key in loss_history[0].keys()}
        
        checkpoints.save_checkpoint(
            ckpt_dir = checkpoint_dir,
            target={"state": state, "loss": loss_history},
            step=step+1, 
            overwrite=False, 
            keep=NUM_CHECKPOINTS_TO_KEEP,
        )
        print(f"Checkpoint saved at step {step+1}")

        loss_history = []


# %% [markdown]
# ### Analysis

# %%
## make one combined image file

import os
import numpy as np
import matplotlib.pyplot as plt
from flax.training import checkpoints
from pinn.cryoet_io import load_mrc_data
from pinn.model import PINN
from pinn.plot import (
    visualize_checkpoint_result,
    visualize_cryoET_with_contours,
    plot_3d_isosurface,
    plot_loss_history_ax,
    plot_normalized_loss_history_ax,
)
from pinn.utils import assemble_loss_history

GRID_SIZE = 64
# lambda_1 = 1000000
# lambda_2 = 100
num_collocation = 10000
# num_collocation = 1000
num_layers = 2
# shape = "bud"


# steps_to_visualize = [0, 200, 600, 1000]
# steps_series = list(range(0, 1001, 100))
steps_to_visualize = [0, 100, 1000, 10000]
steps_series = list(range(0, 10001, 100))

# Load checkpoint data
checkpoint_dir = os.path.abspath(f"../outputs/logs/{shape}/lambda_{lambda_1}_{lambda_2}")
print(checkpoint_dir)
checkpoint_data = {}
for step in steps_series:
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    checkpoint_data[step] = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)

# Load CryoET data
mrc_file_path = f"../data/synthetic/{shape}.mrc"
cryoET_data = load_mrc_data(mrc_file_path, grid_size=GRID_SIZE)

# Create a big figure with 4 rows: cryoET + contours, prediction result, 3D isosurface, and leave room for loss plots
fig = plt.figure(figsize=(3*4, 12))

# Row 1: visualize_cryoET_with_contours
for i, step in enumerate(steps_to_visualize):
    ax = fig.add_subplot(4, 4, i + 1)
    visualize_cryoET_with_contours(ax, step, checkpoint_data[step], cryoET_data, grid_size=GRID_SIZE, slice_index=32, axis="z")
    if i == 0:
        ax.set_ylabel("CryoET + Contours")

# Row 2: visualize_checkpoint_result
for i, step in enumerate(steps_to_visualize):
    ax = fig.add_subplot(4, 4, 4 + i + 1)
    visualize_checkpoint_result(ax, step, checkpoint_data[step], grid_size=GRID_SIZE, slice_index=32, axis="z")
    if i == 0:
        ax.set_ylabel("Level-set function")

# Row 3: plot_3d_isosurface
for i, step in enumerate(steps_to_visualize):
    ax = fig.add_subplot(4, 4, 8 + i + 1, projection='3d')
    plot_3d_isosurface(ax, step, checkpoint_data[step], grid_size=GRID_SIZE)
    if i == 0:
        ax.set_zlabel("3D Isosurface", rotation=180)

# Row 4: loss history
assembled_loss = assemble_loss_history(checkpoint_data)
ax = fig.add_subplot(4, 4, 12+1)
plot_loss_history_ax(ax, assembled_loss)
for i in [0,1,2]:
    ax = fig.add_subplot(4, 4, 12+i+2)
    plot_normalized_loss_history_ax(ax, i, assembled_loss)
    

info_text = (
    f"GRID_SIZE = {GRID_SIZE}, mask_thre = 0.8\n"
    f"num_layers = {num_layers}, hidden_dim = 16, num_freq = 10, learning_rate = 1e-3\n"
    f"lambda_data = {lambda_1:.1e}, lambda_physics = {lambda_2:.1e}\n"
    f"epsilon = 0.05, num_collocation = {num_collocation}"
)

fig.text(
    0.02, 1.08,  # (x, y) position in figure coordinates
    info_text,
    fontsize=12,
    ha='left',
    va='top'
)

plt.tight_layout()
# plt.savefig("../outputs/figs/analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
## make one combined physics analysis image

import os
import numpy as np
import matplotlib.pyplot as plt
from flax.training import checkpoints
from pinn.cryoet_io import load_mrc_data
from pinn.model import PINN
from pinn.plot import (
    visualize_checkpoint_result,
    visualize_physics_loss,
)
from pinn.utils import assemble_loss_history

GRID_SIZE = 64
# lambda_1 = 5000000
# lambda_2 = 1
num_collocation = 10000
num_layers = 2
epsilon = 0.05
# shape = "multi"

# steps_to_visualize = [0, 200, 600, 1000]
# steps_to_visualize = [0, 2000, 6000, 10000]
# steps_to_visualize = [0, 1000, 6000, 10000]
steps_to_visualize = [0, 100, 1000, 10000]
# steps_to_visualize = [0, 10000, 20000, 30000]
# steps_to_visualize = [0, 30000, 60000, 90000]

# Load checkpoint data
checkpoint_dir = os.path.abspath(f"../outputs/logs/{shape}/lambda_{lambda_1}_{lambda_2}")
checkpoint_data = {}
for step in steps_to_visualize:
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    checkpoint_data[step] = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)

# Load CryoET data
mrc_file_path = f"../data/synthetic/{shape}.mrc"
# mrc_file_path = f"../data/synthetic/flip_noise/{shape}_f{str(flip_ratio).replace('.', '')}.mrc"
# mrc_file_path = f"../data/synthetic/flip_noise/{shape}_f{str(flip_ratio).replace('.', '')}_w{wedge_axis}.mrc"
cryoET_data = load_mrc_data(mrc_file_path, grid_size=GRID_SIZE)


# Create a big figure with 4 rows: cryoET + contours, prediction result, 3D isosurface, and leave room for loss plots
fig = plt.figure(figsize=(4*4, 4*4))


# Row 1: Phi
for i, step in enumerate(steps_to_visualize):
    ax = fig.add_subplot(4, 4, i + 1)
    visualize_physics_loss(ax, epsilon=epsilon, step=step,
                           checkpoint=checkpoint_data[step], component="phi",
                           grid_size=GRID_SIZE, slice_index=32, axis="z", vmin=-1, vmax=1)
    # visualize_checkpoint_result(ax, step, checkpoint_data[step], grid_size=GRID_SIZE, slice_index=32, axis="z", colorbar = True)
    if i == 0:
        ax.set_ylabel(r"$\phi$")

# Row 2: data loss
for i, step in enumerate(steps_to_visualize):
    ax = fig.add_subplot(4, 4, 4 + i + 1)
    visualize_physics_loss(ax, epsilon=epsilon, step=step,
                           checkpoint=checkpoint_data[step], component="data",
                           grid_size=GRID_SIZE, slice_index=int(GRID_SIZE/2), axis="z", vmin=0, vmax=5e-7, cryoET_data=cryoET_data)
    if i == 0:
        ax.set_ylabel(r"Data loss")

# Row 3: physics loss
for i, step in enumerate(steps_to_visualize):
    ax = fig.add_subplot(4, 4, 8 + i + 1)
    visualize_physics_loss(ax, epsilon=epsilon, step=step,
                           checkpoint=checkpoint_data[step], component="residual",
                           grid_size=GRID_SIZE, slice_index=32, axis="z", vmin=0, vmax=150)

    if i == 0:
        ax.set_ylabel(r"$|\Delta \phi - (1/\epsilon^2)(\phi^2 - 1)\phi|$")

# Row 4: tension energy
for i, step in enumerate(steps_to_visualize):
    ax = fig.add_subplot(4, 4, 12 + i + 1)
    visualize_physics_loss(ax, epsilon=epsilon, step=step,
                           checkpoint=checkpoint_data[step], component="tension",
                           grid_size=GRID_SIZE, slice_index=int(GRID_SIZE/2), axis="z", vmin=0, vmax=400)
    if i == 0:
        ax.set_ylabel(r"$|\nabla \phi|^2 + (1/2\epsilon^2) (\phi^2 -1)^2$")
        # ax.set_ylabel(r"$|\nabla\phi|^2$")


info_text = (
    f"GRID_SIZE = {GRID_SIZE}, mask_thre = 0.8\n"
    f"num_layers = {num_layers}, hidden_dim = 16, num_freq = 10, learning_rate = 1e-3\n"
    f"lambda_data = {lambda_1:.1e}, lambda_physics = {lambda_2:.1e}\n"
    f"epsilon = {epsilon}, num_collocation = {num_collocation}"
)

fig.text(
    0.02, 1.08,  # (x, y) position in figure coordinates
    info_text,
    fontsize=12,
    ha='left',
    va='top'
)

plt.tight_layout()
# plt.savefig("../outputs/figs/physics.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
## make one combined derivative analysis image

import os
import numpy as np
import matplotlib.pyplot as plt
from flax.training import checkpoints
from pinn.cryoet_io import load_mrc_data
from pinn.model import PINN
from pinn.plot import (
    visualize_physics_loss,
)
from pinn.utils import assemble_loss_history

GRID_SIZE = 64
lambda_1 = 5000000
lambda_2 = 1
num_collocation = 1000
num_layers = 2
epsilon = 0.05

steps_to_visualize = [0, 200, 600, 1000]
# steps_to_visualize = [0, 30000, 60000, 90000]

# Load checkpoint data
checkpoint_dir = os.path.abspath(f"../outputs/logs/biconcave/lambda_{lambda_1}_{lambda_2}")
checkpoint_data = {}
for step in steps_to_visualize:
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    checkpoint_data[step] = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)


# Create a big figure with 4 rows: cryoET + contours, prediction result, 3D isosurface, and leave room for loss plots
fig = plt.figure(figsize=(2.5*4, 2*6))


# Row 2: grad_x
for i, step in enumerate(steps_to_visualize):
    ax = fig.add_subplot(6, 4, i + 1)
    visualize_physics_loss(ax, epsilon = 0.05, step=step, checkpoint=checkpoint_data[step], component = "grad_x", grid_size = GRID_SIZE, slice_index=32, axis="z", vmin=0, vmax=1e1)
    if i == 0:
        ax.set_ylabel(r"$|\partial \phi / \partial x|$")

# Row 3: grad_y
for i, step in enumerate(steps_to_visualize):
    ax = fig.add_subplot(6, 4, 4 + i + 1)
    visualize_physics_loss(ax, epsilon = 0.05, step=step, checkpoint=checkpoint_data[step], component = "grad_y", grid_size = GRID_SIZE, slice_index=32, axis="z", vmin=0, vmax=1e1)
    if i == 0:
        ax.set_ylabel(r"$|\partial \phi / \partial y|$")

# Row 4: grad_z
for i, step in enumerate(steps_to_visualize):
    ax = fig.add_subplot(6, 4, 8 + i + 1)
    visualize_physics_loss(ax, epsilon = 0.05, step=step, checkpoint=checkpoint_data[step], component = "grad_z", grid_size = GRID_SIZE, slice_index=32, axis="z", vmin=0, vmax=1e1)
    if i == 0:
        ax.set_ylabel(r"$|\partial \phi / \partial z|$")

# Row 5: hess_xx
for i, step in enumerate(steps_to_visualize):
    ax = fig.add_subplot(6, 4, 12 + i + 1)
    visualize_physics_loss(ax, epsilon = 0.05, step=step, checkpoint=checkpoint_data[step], component = "hess_xx", grid_size = GRID_SIZE, slice_index=32, axis="z", vmin=0, vmax=1e1)
    if i == 0:
        ax.set_ylabel(r"$|\partial^2 \phi / \partial x^2|$")

# Row 6: hess_yy
for i, step in enumerate(steps_to_visualize):
    ax = fig.add_subplot(6, 4, 16 + i + 1)
    visualize_physics_loss(ax, epsilon = 0.05, step=step, checkpoint=checkpoint_data[step], component = "hess_yy", grid_size = GRID_SIZE, slice_index=32, axis="z", vmin=0, vmax=1e1)
    if i == 0:
        ax.set_ylabel(r"$|\partial^2 \phi / \partial y^2|$")

# Row 7: hess_zz
for i, step in enumerate(steps_to_visualize):
    ax = fig.add_subplot(6, 4, 20 + i + 1)
    visualize_physics_loss(ax, epsilon = 0.05, step=step, checkpoint=checkpoint_data[step], component = "hess_zz", grid_size = GRID_SIZE, slice_index=32, axis="z", vmin=0, vmax=1e1)
    if i == 0:
        ax.set_ylabel(r"$|\partial^2 \phi / \partial z^2|$")

info_text = (
    f"GRID_SIZE = {GRID_SIZE}, mask_thre = 0.8\n"
    f"num_layers = {num_layers}, hidden_dim = 16, num_freq = 10, learning_rate = 1e-3\n"
    f"lambda_data = {lambda_1:.1e}, lambda_physics = {lambda_2:.1e}\n"
    f"epsilon = {epsilon}, num_collocation = {num_collocation}"
)

fig.text(
    0.02, 1.08,  # (x, y) position in figure coordinates
    info_text,
    fontsize=12,
    ha='left',
    va='top'
)

plt.tight_layout()
# plt.savefig("../outputs/figs/grad.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# Make a movie

import matplotlib.pyplot as plt
import os
from pinn.plot import plot_3d_isosurface
from flax.training import checkpoints
import imageio
import re

steps_to_visualize = list(range(0, 30001, 1000))
log_dir = os.path.abspath(f"../outputs/logs/biconcave/lambda_{lambda_1}_{lambda_2}/")
frame_dir = os.path.join(log_dir, f"frames")

def save_isosurface_frame(step, checkpoint, save_dir="frames", grid_size=64):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    plot_3d_isosurface(ax, step, checkpoint, grid_size)
    
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"frame_{step:06d}.png")
    plt.savefig(filename, dpi=150)
    plt.close(fig)

def make_movie_from_frames(frame_dir="frames", output="movie.mp4", fps=12, start=0, end=30000):
    # Match files like frame_XXXX.png and extract the numeric part
    pattern = re.compile(r"frame_(\d+)\.png")

    frames = []
    for fname in os.listdir(frame_dir):
        match = pattern.match(fname)
        if match:
            index = int(match.group(1))
            if start <= index <= end:
                frames.append((index, os.path.join(frame_dir, fname)))

    frames.sort(key=lambda x: x[0])

    with imageio.get_writer(output, fps=fps) as writer:
        for _, frame_path in frames:
            image = imageio.imread(frame_path)
            writer.append_data(image)

for step in steps_to_visualize:
    ckpt_path = os.path.join(log_dir, f"checkpoint_{step}")
    ckpt_data = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None)
    print(f"saving step {step}")
    save_isosurface_frame(step, ckpt_data, save_dir=frame_dir)

output=os.path.abspath(f"{frame_dir}/movie.mp4")
make_movie_from_frames(frame_dir=frame_dir, output=output)

# %% [markdown]
# ## Individual plot

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from flax.training import checkpoints
from pinn.cryoet_io import load_mrc_data
from pinn.model import PINN
from pinn.plot import visualize_checkpoint_result, visualize_cryoET_with_contours

GRID_SIZE = 64
# lambda_1 = 5000000.0 # Weight for data loss
# lambda_2 = 0.0001 # Weight for physics loss

# Define the checkpoint steps to visualize
# steps_to_visualize = [0, 20, 40, 60]
steps_to_visualize = [0, 20, 60, 100]
# steps_to_visualize = [0, 200, 600, 1000]
# steps_to_visualize = [0, 60, 180, 300]
# steps_to_visualize = [0, 100, 300, 500]

# Load checkpoint data
checkpoint_dir = os.path.abspath(f"../outputs/logs/biconcave/lambda_{lambda_1}_{lambda_2}")
checkpoint_data = {}
for step in steps_to_visualize:
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    checkpoint_data[step] = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)

# Create a figure with horizontally arranged subplots
fig, axes = plt.subplots(1, len(steps_to_visualize), figsize=(15, 5))

# Load CryoET data (modify based on your dataset)
mrc_file_path = "../data/synthetic/biconcave.mrc"
cryoET_data = load_mrc_data(mrc_file_path, grid_size=64)

# Load and visualize results for different steps
for i, step in enumerate(steps_to_visualize):
    visualize_cryoET_with_contours(axes[i], step, checkpoint_data[step], cryoET_data = cryoET_data, grid_size = GRID_SIZE, slice_index=32, axis="z")

plt.tight_layout()
plt.show()


# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from flax.training import checkpoints
from pinn.cryoet_io import load_mrc_data
from pinn.model import PINN
from pinn.plot import visualize_physics_loss

GRID_SIZE = 64
lambda_1 = 5000000.0 # Weight for data loss
lambda_2 = 1e-5 # Weight for physics loss

# Define the checkpoint steps to visualize
# steps_to_visualize = [0, 20, 60, 100]
steps_to_visualize = [0, 200, 600, 1000]

# Load checkpoint data
checkpoint_dir = os.path.abspath(f"../outputs/logs/biconcave/lambda_{lambda_1}_{lambda_2}")
checkpoint_data = {}
for step in steps_to_visualize:
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    checkpoint_data[step] = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)

# Create a figure with horizontally arranged subplots
fig, axes = plt.subplots(1, len(steps_to_visualize), figsize=(15, 5))

# Load CryoET data (modify based on your dataset)
mrc_file_path = "../data/synthetic/biconcave.mrc"
cryoET_data = load_mrc_data(mrc_file_path, grid_size=64)

# Load and visualize results for different steps
for i, step in enumerate(steps_to_visualize):
    # visualize_physics_loss(axes[i], step, checkpoint_data[step], epsilon = 0.05, component = "residual", grid_size = GRID_SIZE, slice_index=32, axis="z", vmin=0, vmax=1e10)
    # visualize_physics_loss(axes[i], step, checkpoint_data[step], epsilon = 0.05, component = "laplacian", grid_size = GRID_SIZE, slice_index=32, axis="z", vmin=0, vmax=1e10)
    # visualize_physics_loss(axes[i], step, checkpoint_data[step], epsilon = 0.05, component = "nonlinear", grid_size = GRID_SIZE, slice_index=32, axis="z", vmin=0, vmax=0.15)
    # visualize_physics_loss(axes[i], step, checkpoint_data[step], epsilon = 0.05, component = "grad_x", grid_size = GRID_SIZE, slice_index=32, axis="z", vmin=0, vmax=1e4)
    # visualize_physics_loss(axes[i], step, checkpoint_data[step], epsilon = 0.05, component = "grad_y", grid_size = GRID_SIZE, slice_index=32, axis="z", vmin=0, vmax=1e4)
    # visualize_physics_loss(axes[i], step, checkpoint_data[step], epsilon = 0.05, component = "grad_z", grid_size = GRID_SIZE, slice_index=32, axis="z", vmin=0, vmax=1e4)
    # visualize_physics_loss(axes[i], step, checkpoint_data[step], epsilon = 0.05, component = "hess_xx", grid_size = GRID_SIZE, slice_index=32, axis="z", vmin=0, vmax = 1e9)
    # visualize_physics_loss(axes[i], step, checkpoint_data[step], epsilon = 0.05, component = "hess_yy", grid_size = GRID_SIZE, slice_index=32, axis="z", vmin=0, vmax = 1e9)
    visualize_physics_loss(axes[i], step, checkpoint_data[step], epsilon = 0.05, component = "hess_zz", grid_size = GRID_SIZE, slice_index=32, axis="z", vmin=0, vmax = 1e9)


plt.tight_layout()
plt.show()


# %%
import sys
sys.path
os.getcwd()

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from flax.training import checkpoints
from pinn.plot import visualize_checkpoint_result

GRID_SIZE = 64
lambda_1 = 5000000.0  # Weight for data loss
lambda_2 = 1e-5    # Weight for physics loss

# steps_to_visualize = [0, 20, 40, 60]
steps_to_visualize = [0, 20, 60, 100]
# steps_to_visualize = [0, 200, 600, 1000]
# steps_to_visualize = [0, 60, 180, 300]
# steps_to_visualize = [0, 100, 300, 500]

# Load checkpoint data
checkpoint_dir = os.path.abspath(f"../outputs/logs/biconcave/lambda_{lambda_1}_{lambda_2}")
checkpoint_data = {}
for step in steps_to_visualize:
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    checkpoint_data[step] = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)

# Create a figure with horizontally arranged subplots
fig, axes = plt.subplots(1, len(steps_to_visualize), figsize=(15, 5))

for i, step in enumerate(steps_to_visualize):
    visualize_checkpoint_result(axes[i], step, checkpoint_data[step], grid_size = GRID_SIZE, slice_index=32, axis="z")


# Adjust layout and add colorbar
# fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.02)
plt.tight_layout()
plt.show()

# %%
import numpy as np
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pinn.model import PINN
from flax.training import checkpoints
from pinn.plot import plot_3d_isosurface

GRID_SIZE = 64
# lambda_1 = 50000.0  # Weight for data loss
# lambda_2 = 0.000    # Weight for physics loss

# steps_to_visualize = [0, 20, 40, 60]
steps_to_visualize = [0, 20, 60, 100]
# steps_to_visualize = [0, 200, 600, 1000]
# steps_to_visualize = [0, 60, 180, 300]
# steps_to_visualize = [0, 100, 300, 500]

# Load checkpoint data
checkpoint_dir = os.path.abspath(f"../outputs/logs/biconcave/lambda_{lambda_1}_{lambda_2}")
checkpoint_data = {}
for step in steps_to_visualize:
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    checkpoint_data[step] = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)

# Create a figure with horizontally arranged subplots
fig, axes = plt.subplots(1, len(steps_to_visualize), figsize=(15, 5), subplot_kw={'projection': '3d'})

for i, step in enumerate(steps_to_visualize):
    plot_3d_isosurface(axes[i], step, checkpoint_data[step], grid_size = GRID_SIZE)

plt.tight_layout()
plt.show()


# %%
import os
from flax.training import checkpoints
from pinn.utils import assemble_loss_history
from pinn.plot import plot_loss_history, plot_normalized_loss_history

# lambda_1 = 50000.0  # Weight for data loss
# lambda_2 = 0.000    # Weight for physics loss

steps_to_visualize = [0, 20, 40, 60, 80, 100]
# steps_to_visualize = list(range(0, 301, 20))
# steps_to_visualize = list(range(0, 501, 20))
# steps_to_visualize = list(range(0, 1001, 20))

# Load checkpoint data
checkpoint_dir = os.path.abspath(f"../outputs/logs/biconcave/lambda_{lambda_1}_{lambda_2}")
checkpoint_data = {}
for step in steps_to_visualize:
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    checkpoint_data[step] = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)

assembled_loss = assemble_loss_history(checkpoint_data)
plot_loss_history(assembled_loss)
plot_normalized_loss_history(assembled_loss)


# %%
# calculate volume and surface area
from pinn.plot import plot_phase_metrics_ax
import os
import numpy as np
from flax.training import checkpoints
import matplotlib.pyplot as plt

steps_to_visualize = list(range(0, 101, 20))
V_0 = 4*np.pi/3*0.5**3
A_0 = 4*np.pi*0.5**2

# Load checkpoint data
# checkpoint_dir = os.path.abspath(f"../outputs/logs/biconcave/lambda_5000000_0/")
checkpoint_dir = os.path.abspath(f"../outputs/logs/biconcave/lambda_5000000_1e-05/")
# checkpoint_dir = os.path.abspath(f"../outputs/logs/biconcave/lambda_{lambda_1}_{lambda_2}")
checkpoint_data = {}
for step in steps_to_visualize:
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    checkpoint_data[step] = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)


fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Plot volume
plot_phase_metrics_ax(axs[0], checkpoint_data, metrics="volume", epsilon=0.05, V_0=V_0, grid_size=64)
axs[0].set_title("Volume over Training")

# Plot surface area
plot_phase_metrics_ax(axs[1], checkpoint_data, metrics="area", epsilon=0.05, A_0=A_0, grid_size=64)
axs[1].set_title("Surface Area over Training")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Junk code

# %%
import numpy as np
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pinn.model import PINN
from flax.training import checkpoints
from pinn.plot import plot_3d_isosurface
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_3d_isosurface(ax, step, checkpoint, grid_size=64):
    """
    Load a checkpoint, compute φ values, extract the isosurface, and plot it.

    Args:
        ax: Matplotlib subplot axis to plot on.
        checkpoint_path (str): Path to the checkpoint directory.
        checkpoint_label (str): Label for the plot title.
    """
    # Define normalized coordinate grid (-1 to 1)
    x = jnp.linspace(-1, 1, grid_size)
    y = jnp.linspace(-1, 1, grid_size)
    z = jnp.linspace(-1, 1, grid_size)
    
    # Generate a 3D grid using meshgrid
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    grid_points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    
    # Load model and parameters
    model = PINN()
    state = checkpoint["state"]
    params = state["params"]
    phi_fn = lambda x: model.apply(params, x.reshape(-1, 3))

    # Compute φ values over a 3D grid
    phi_values = phi_fn(grid_points).reshape(grid_size, grid_size, grid_size)
    phi_values = phi_values.T

    # Convert φ values from JAX to NumPy for visualization
    phi_values_np = np.array(phi_values)

    # Apply marching cubes to extract the isosurface
    verts, faces, _, _ = marching_cubes(phi_values_np, level=0, spacing=(2/grid_size, 2/grid_size, 2/grid_size))
    # verts, faces, _, _ = marching_cubes(phi_values_np, level=0, spacing=(1/grid_size, 1/grid_size, 1/grid_size))
    # verts, faces, _, _ = marching_cubes(phi_values_np, level=0, spacing=(1, 1, 1))

    verts -= 1.0    


    print(f"Number of vertices: {len(verts)}")
    print(f"Number of faces: {len(faces)}")

    print(f"X range: {verts[:, 0].min()} to {verts[:, 0].max()}")
    print(f"Y range: {verts[:, 1].min()} to {verts[:, 1].max()}")
    print(f"Z range: {verts[:, 2].min()} to {verts[:, 2].max()}")
    
    print(f"X grid range: {X.min()} to {X.max()}")
    print(f"Y grid range: {Y.min()} to {Y.max()}")
    print(f"Z grid range: {Z.min()} to {Z.max()}")



    # Add the surface mesh with improved appearance
    mesh = Poly3DCollection(verts[faces], alpha=0.1, edgecolor="k", linewidth=0.2, facecolor="cyan")
    ax.add_collection3d(mesh)

    # Improve visualization by adding a wireframe effect
    # ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color="gray", alpha=0.15, edgecolor="black", linewidth=0.05)


    # Set axis labels
    ax.set_xlabel("X-axis", fontsize=10, labelpad=8)
    ax.set_ylabel("Y-axis", fontsize=10, labelpad=8)
    ax.set_zlabel("Z-axis", fontsize=10, labelpad=8)
    ax.set_title(f"Step {step}", fontsize=12)

    # Adjust camera angle & axis limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1, 1, 1])

    # ax.view_init(elev=0, azim=0)  # Top-down view
    # ax.view_init(elev=90, azim=0)  # Side view
    # ax.view_init(elev=0, azim=90)  # Rotate 90° around Z-axis
    # ax.view_init(elev=45, azim=30)  # Custom view
    ax.view_init(elev=30, azim=45)  # Adjust view angle
    # ax.view_init(elev=180, azim=0)  # Upside down    

    # Improve aesthetics
    ax.grid(False)  # Hide the default grid
    ax.set_facecolor("white")  # Change background color



GRID_SIZE = 64
steps_to_visualize = [0, 100, 200]
# steps_to_visualize = [0, 100, 200, 300, 400, 500]
# steps_to_visualize = [0, 100, 200, 300]

# Load checkpoint data
checkpoint_dir = os.path.abspath("../outputs/checkpoints")
checkpoint_data = {}
for step in steps_to_visualize:
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    checkpoint_data[step] = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)

# Create a figure with horizontally arranged subplots
fig, axes = plt.subplots(1, len(steps_to_visualize), figsize=(15, 5), subplot_kw={'projection': '3d'})

for i, step in enumerate(steps_to_visualize):
    plot_3d_isosurface(axes[i], step, checkpoint_data[step], grid_size = GRID_SIZE)

plt.tight_layout()
plt.show()


# %%
import os
from flax.training import checkpoints
from pinn.model import PINN
from pinn.train import generate_sdf
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Define checkpoint directory
checkpoint_dir = os.path.abspath("../outputs/checkpoints/checkpoint_0")

# Restore checkpoint (automatically loads the latest checkpoint file in the directory)
checkpoint = checkpoints.restore_checkpoint(checkpoint_dir, target=None)

# Extract the saved state
state = checkpoint["state"]

# Print checkpoint details
print(f"Checkpoint loaded successfully at step {state['step']}")
print(f"Lambda_1: {state['lambda_1']}, Lambda_2: {state['lambda_2']}")

# Load the trained PINN model
model = PINN()

# Generate the voxel grid used in SDF pretraining
grid_size = 64  # Match the training grid size
grid_points, true_sdf = generate_sdf(grid_size=grid_size, radius=0.5)

# **Normalize true SDF values using sphere radius**
# true_sdf = true_sdf / 15.0  # Expected range should be [-1,1]

# Compute the neural network's predicted SDF using normalized grid points
predicted_sdf = model.apply(state["params"], grid_points).reshape(grid_size, grid_size, grid_size)

# Select a middle slice for visualization
slice_index = grid_size // 2  # Middle slice at Z=32

# Create side-by-side plots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# **1. True SDF**
img1 = axs[0].imshow(true_sdf[:, :, slice_index].T, cmap="bwr", origin="lower", vmin=-1, vmax=1)
axs[0].set_title("True SDF (Slice at Z=32)")
axs[0].set_xlabel("X-axis (normalized)")
axs[0].set_ylabel("Y-axis (normalized)")
cbar1 = fig.colorbar(img1, ax=axs[0])
cbar1.set_label("Signed Distance (Normalized)")

# **2. NN-Predicted SDF**
img2 = axs[1].imshow(predicted_sdf[:, :, slice_index].T, cmap="bwr", origin="lower", vmin=-1, vmax=1)
axs[1].set_title("NN-Predicted SDF (Slice at Z=32)")
axs[1].set_xlabel("X-axis (normalized)")
axs[1].set_ylabel("Y-axis (normalized)")

# Add colorbar for NN prediction
cbar2 = fig.colorbar(img2, ax=axs[1])
cbar2.set_label("Signed Distance (Normalized)")

plt.tight_layout()
plt.show()

# Print some sample values for numerical comparison
print(f"True SDF Min: {true_sdf.min()}, Max: {true_sdf.max()}")
print(f"NN-Predicted SDF Min: {predicted_sdf.min()}, Max: {predicted_sdf.max()}")


# %%
sample_idx = np.random.choice(grid_points.shape[0], 10, replace=False)
pred_values = model.apply(state["params"], grid_points[sample_idx])

print("Sample Grid Points:", grid_points[sample_idx])
print("True SDF Values:", true_sdf.ravel()[sample_idx])
print("NN-Predicted SDF Values:", pred_values)


# %%
from data_generation.mesh_to_cryoet import plot_single_slice

mrc_file_path = "../data/synthetic/biconcave.mrc"
cryoET_data = load_mrc_data(mrc_file_path, grid_size=64)

plot_single_slice(cryoET_data, axis='z')

# %%
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

def visualize_checkpoint_result(ax, step, checkpoint, model, grid_size=64, cryoET_data=None):
    """
    Compute and visualize the level-set function from a given checkpoint.
    """
    state = checkpoint["state"]  # Extract saved model state
    params = state["params"]  # Extract trained parameters

    # Use the model's apply function
    phi_fn = lambda x: model.apply(params, x.reshape(-1, 3))

    # Define the real coordinate values
    x = jnp.linspace(-1.5, 1.5, grid_size)
    y = jnp.linspace(-1.5, 1.5, grid_size)
    z = jnp.linspace(-1.5, 1.5, grid_size)

    # Generate a 3D grid
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    grid_points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    
    # Compute φ values
    phi_values = phi_fn(grid_points).reshape(grid_size, grid_size, grid_size)

    # Extract the central XY slice at Z=0
    mid_slice = grid_size // 2
    img = ax.imshow(phi_values[:, :, mid_slice].T, cmap='bwr', origin='lower',
                    extent=[x.min(), x.max(), y.min(), y.max()],
                    vmin=-0.5, vmax=0.5, alpha=1.0)

    # Plot the CryoET data as a heatmap if available
    if cryoET_data is not None:
        cryoET_numpy = np.array(cryoET_data[:, :, mid_slice])  # Convert to NumPy for visualization
        alpha_mask = np.where(cryoET_numpy > 0.7, 0.5, 0.0)  # Only show intensities > 0.7
        ax.imshow(np.ones_like(cryoET_numpy), cmap='gray', origin='lower',
                  extent=[x.min(), x.max(), y.min(), y.max()], alpha=alpha_mask)

    # Plot contour lines for φ=0
    contour = ax.contour(X[:, :, mid_slice], Y[:, :, mid_slice],
                         phi_values[:, :, mid_slice], levels=[0], colors='black')

    ax.clabel(contour, fmt="φ=0", colors='black')  # Label contour line

    # Set axis labels
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # Set appropriate tick positions and labels
    tick_positions = jnp.linspace(x.min(), x.max(), num=5)
    tick_labels = [f"{val:.1f}" for val in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    ax.set_title(f"Level-Set Function at z=0 (Step {step})")

    return img


# %%
step_to_load = 100  # Choose a step
checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step_to_load}")

print(checkpoint_path)

# Restore the specific checkpoint
checkpoint_data = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)
print("Keys stored in checkpoint:", checkpoint_data.keys())

print(checkpoint_data['state'].keys())
print(checkpoint_data['loss'].keys())
print(checkpoint_data['loss']['data_loss'].shape)


# %%
loss_dict = checkpoint_data['loss']

# Convert to a list of dictionaries (sorted by step order)
loss_history = [loss_dict[key] for key in sorted(loss_dict.keys(), key=int)]

# Convert to Pandas DataFrame for easy analysis
import pandas as pd

df = pd.DataFrame(loss_history)  # Each row corresponds to one saved loss entry
print(df.head())  # Display the first few loss entries


# %%
for step in [100, 200, 300, 400]:
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    data = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)
    print(f"Checkpoint {step}: step stored inside = {data['step']}")


# %%
# Restore checkpoint
state_dict = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=None)

# Extract trained model parameters
params = state_dict['params']

# Use restored parameters in the model for inference
# output = model.apply(params, some_input_data)

state = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=state)
print(f"Resumed training from step {state.step}")

import os
print("Available checkpoints:", os.listdir(checkpoint_dir))


# %%


# %%
import os

# List all files in the checkpoint directory
checkpoint_files = os.listdir(checkpoint_dir)
print("Available checkpoints:", checkpoint_files)


# %%
from flax.training import checkpoints

# Restore checkpoint
state = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=state)
print("Checkpoint restored, ready for visualization.")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a 3D grid
grid_size = 50
x_vals = np.linspace(-1.5, 1.5, grid_size)
y_vals = np.linspace(-1.5, 1.5, grid_size)
z_vals = np.linspace(-1.5, 1.5, grid_size)

X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
xyz_grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

# Compute predictions using the restored model
phi_values = state.apply_fn(state.params, xyz_grid)
phi_values = phi_values.reshape(grid_size, grid_size, grid_size)

# Plot an isosurface where φ(x,y,z) = 0
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection="3d")
ax.contourf(X, Y, Z, phi_values, levels=[0], cmap="coolwarm")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("3D Isosurface of φ(x,y,z) = 0")
plt.show()


# %%
import pickle
import jax
import numpy as np
from pinn.model import PINN
from pinn.train import create_train_state
from pinn.plot import visualize_results
from pinn.cryoet_io import load_mrc_data

# Load trained model parameters
with open("pinn_params.pkl", "rb") as f:
    restored_params = pickle.load(f)

# Load optimizer state (if resuming training)
with open("pinn_opt_state.pkl", "rb") as f:
    restored_opt_state = pickle.load(f)

# Load training step
with open("pinn_step.pkl", "rb") as f:
    restored_step = pickle.load(f)

# Recreate model and training state
key = jax.random.PRNGKey(0)
model = PINN()  # Recreate model architecture
state, _ = create_train_state(key)  # Initialize new state

# Restore parameters and optimizer
state = state.replace(params=restored_params, opt_state=restored_opt_state, step=restored_step)

print("Model and training state restored successfully!")


# Load MRC data 
mrc_file_path = "../data/synthetic/biconcave.mrc"
cryoET_data = load_mrc_data(mrc_file_path, grid_size=64)
print("MRC data loaded successfully! Shape:", cryoET_data.shape)


visualize_results(lambda x: state.apply_fn(state.params, x), cryoET_data=cryoET_data)

pixsel

# %%
