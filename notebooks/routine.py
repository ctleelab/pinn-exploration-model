import sys
sys.path.append('../src/pinn')
sys.path.append('../src/data_generation')


import jax
import jax.numpy as jnp
import os
import numpy as np
from pinn.train import create_train_state, train_step
from pinn.cryoet_io import load_mrc_data
from flax.training import checkpoints
from pinn.utils import initial_loss
from tqdm.notebook import trange


GRID_SIZE = 64
NUM_CHECKPOINTS_TO_KEEP = 1000  # Checkpoint retention count (older ones get removed)



# Parameters
lambda_1 = 1000000
# lambda_2_list = [0, 10, 100, 1000]
# lambda_2_list = [40, 50, 60, 70]
# lambda_2_list = [0, 10, 100, 1000]
lambda_2_list = [0, 10]
# lambda_2_list = [100, 1000]
shape = "multi"
sdf_pretrain="multi"   # Initial condition ("sphere" or "plane")
flip_ratio=0.15
# wedge_axis='y'

num_collocation = 10000
num_steps = 10000
save_interval = 100


#### LOAD MRC DATA #####
# mrc_file_path = f"../data/synthetic/{shape}.mrc"
mrc_file_path = f"../data/synthetic/flip_noise/{shape}_f{str(flip_ratio).replace('.', '')}.mrc"
# mrc_file_path = f"../data/synthetic/flip_noise/{shape}_f{str(flip_ratio).replace('.', '')}_w{wedge_axis}.mrc"
print(f"Input file: {mrc_file_path}")

cryoET_data = load_mrc_data(mrc_file_path, grid_size=GRID_SIZE)
print("MRC data loaded successfully! Shape:", cryoET_data.shape)
print("Cryo-ET Data Shape:", cryoET_data.shape)
print("Cryo-ET Data Min:", cryoET_data.min())
print("Cryo-ET Data Max:", cryoET_data.max())
print("Cryo-ET Unique Values:", jnp.unique(cryoET_data))



for lambda_2 in lambda_2_list:

	key = jax.random.PRNGKey(0)
	state, model = create_train_state(key, lambda_1=lambda_1, lambda_2=lambda_2, sdf_pretrain=sdf_pretrain, learning_rate=1e-3)
	print(f"Using lambda_1: {state.lambda_1}, lambda_2: {state.lambda_2}")

	x_train = (jax.random.uniform(key, (num_collocation, 3), minval=-1, maxval=1)) # Sampled from [-1, 1]^3
	checkpoint_dir = os.path.abspath(f"../outputs/logs/{shape}/lambda_{lambda_1}_{lambda_2}")

	checkpoints.save_checkpoint(
	    ckpt_dir=checkpoint_dir,
	    target={"state": state, "loss": initial_loss(state, x_train, cryoET_data)},
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

	        loss_history = {key: np.array([entry[key] for entry in loss_history]) for key in loss_history[0].keys()}
	        
	        checkpoints.save_checkpoint(
	            ckpt_dir = checkpoint_dir,
	            target={"state": state, "loss": loss_history},
	            step=step+1, 
	            overwrite=False, 
	            keep=NUM_CHECKPOINTS_TO_KEEP,
	        )
	        # print(f"Checkpoint saved at step {step+1}")

	        loss_history = []
