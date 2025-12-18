import sys
sys.path.append('../src')

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import jax
import jax.numpy as jnp
import os
import numpy as np
from pinn.train import create_train_state, train_step
from pinn.cryoet_io import load_mrc_data
from flax.training import checkpoints
from pinn.utils import initial_loss
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


# Utility function to convert GPU/JAX arrays to CPU/NumPy arrays
def to_numpy(tree):
    return jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x)) if isinstance(x, (jnp.ndarray, np.ndarray)) else x,
        tree,
    )
def as_f32_scalar(x):
    # Make sure to extract as a scalar (float32) even if it's an array/DeviceArray
    return np.asarray(x, dtype=np.float32).reshape(()).item()



#### PARAMETERS ####
GRID_SIZE = 64
NUM_CHECKPOINTS_TO_KEEP = 1000  # Checkpoint retention count

lambda_1 = 1000000
lambda_2_list = [0, 50]
# shape_list = ["biconcave", "bud_04", "multi"]
# shape_list = ["biconcave", "bud_04"]
shape_list = ["biconcave"]

# flip_list=[0.06, 0.07, 0.08, 0.09, 0.11, 0.12, 0.13, 0.14]
# flip_list=[0.17, 0.19, 0.21, 0.23, 0.25]
flip_list=[0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25]
# missing_list = [0.6, 0.7, 0.8, 0.9]
missing_list = [0.5]

threshold=0.8

num_collocation = 10000
num_steps = 10000   # 10000
save_interval = 5000  # 100
key = jax.random.PRNGKey(0)
learning_rate = 1e-3  # 1e-3
data_version = "data_1115"


for flip_ratio in flip_list:
	for missing_ratio in missing_list:
		for shape in shape_list:

			if shape == "biconcave":
				sdf_pretrain="sphere"
				radius = 0.4
			elif shape == "bud_04":
				sdf_pretrain="plane"
				radius = None
			elif shape == "multi":
				sdf_pretrain = "multi"
				radius = None

			# sdf_pretrain = "sphere"
			# radius = 0.4

			flip_str = str(flip_ratio).replace('.', '')
			missing_str = str(missing_ratio).replace('.', '')


			#### LOAD MRC DATA #####
			# mrc_file_path = f"../data/synthetic/{shape}.mrc"
			mrc_file_path = f"../data/synthetic/combine/{shape}_a{flip_str}_m{missing_str}.mrc"
			print(f"Input file: {mrc_file_path}")

			cryoET_data = load_mrc_data(mrc_file_path, grid_size=GRID_SIZE)
			print("MRC data loaded successfully! Shape:", cryoET_data.shape)
			print("Cryo-ET Data Shape:", cryoET_data.shape)
			print("Cryo-ET Data Min:", cryoET_data.min())
			print("Cryo-ET Data Max:", cryoET_data.max())
			print("Cryo-ET Unique Values:", jnp.unique(cryoET_data))

			x_train = (jax.random.uniform(key, (num_collocation, 3), minval=-1, maxval=1)) # Sampled from [-1, 1]^3


			# # Load initial checkpoint data
			# init_ckpt_dir= os.path.abspath(f"../outputs/logs/{shape}/{data_version}/a{flip_str}/init/lambda_{lambda_1}_0")
			# init_ckpt_path = os.path.join(init_ckpt_dir, f"checkpoint_{init_steps}")
			# init_ckpt = checkpoints.restore_checkpoint(ckpt_dir=init_ckpt_path, target=None)


			for lambda_2 in lambda_2_list:

				state, model = create_train_state(key, lambda_1=lambda_1, lambda_2=lambda_2, threshold=threshold, sdf_pretrain=sdf_pretrain, learning_rate=learning_rate, radius=radius)
				# checkpoint_dir = os.path.abspath(f"../outputs/logs/{shape}/lambda_{lambda_1}_{lambda_2}")
				# checkpoint_dir = os.path.abspath(f"../outputs/logs/{shape}/{data_version}/clean/{sdf_pretrain}/lambda_{lambda_1}_{lambda_2}")
				checkpoint_dir = os.path.abspath(f"../outputs/logs/{shape}/{data_version}/combine/a{flip_str}_m{missing_str}/lambda_{lambda_1}_{lambda_2}")
				# checkpoint_dir = os.path.abspath(f"../outputs/logs/{shape}/{data_version}/combine-sphere/a{flip_str}_m{missing_str}/lambda_{lambda_1}_{lambda_2}")
				print(f"Using lambda_1: {state.lambda_1}, lambda_2: {state.lambda_2}")


				init_d = initial_loss(state, x_train, cryoET_data)     # Get the dictionary of initial losses
				init_total = init_d.get("total_loss", init_d.get("total", init_d.get("loss", 0.0)))
				init_data  = init_d.get("data_loss",  init_d.get("data",  0.0))
				init_phys  = init_d.get("physics_loss", init_d.get("phys", init_d.get("physics", 0.0)))

				payload0 = {
				    "state": to_numpy(state),
				    "loss": {
				        "step":         np.asarray([0], dtype=np.int64),
				        "total_loss":   np.asarray([as_f32_scalar(init_total)], dtype=np.float32),
				        "data_loss":    np.asarray([as_f32_scalar(init_data)],  dtype=np.float32),
				        "physics_loss": np.asarray([as_f32_scalar(init_phys)],  dtype=np.float32),
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

				for step in trange(num_steps):
				# for step in range(num_steps):
				    state, loss_val, loss_data_val, loss_physics_val = train_step(state, x_train, cryoET_data)

				    loss_history.append({
				        "step": step + 1,
				        "total_loss": loss_val,
				        "data_loss": loss_data_val,
				        "physics_loss": loss_physics_val
				    })

				    if (step+1) % save_interval == 0:

				        batched_loss = {
				          	"step":         np.asarray([e["step"] for e in loss_history], dtype=np.int64),
				        	"total_loss":   np.asarray([as_f32_scalar(e["total_loss"])   for e in loss_history], dtype=np.float32),
				        	"data_loss":    np.asarray([as_f32_scalar(e["data_loss"])    for e in loss_history], dtype=np.float32),
				        	"physics_loss": np.asarray([as_f32_scalar(e["physics_loss"]) for e in loss_history], dtype=np.float32),
				        }

				        payload = {
				            "state": to_numpy(state),  # 항상 CPU NumPy로
				            "loss":  batched_loss
				        }
				        
				        checkpoints.save_checkpoint(
				            ckpt_dir = checkpoint_dir,
				            target=payload,
				            step=step+1, 
				            overwrite=False, 
				            keep=NUM_CHECKPOINTS_TO_KEEP,
				        )
				        # print(f"Checkpoint saved at step {step+1}")

				        loss_history = []
