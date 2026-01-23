import jax
import jax.numpy as jnp
import optax
from jax import jit
from flax.training import train_state
from pinn.model import PINN, loss_data, loss_physics, loss_sign, total_loss, GRID_SIZE
import numpy as np
from typing import NamedTuple

# Define TrainState for managing training parameters and optimizer
class TrainState(train_state.TrainState):
    lambda_1: float  # Weight for loss_data
    lambda_2: float  # Weight for loss_physics
    lambda_3: float  # Weight for loss_sign
    # threshold: float # Threshold for data loss

def create_train_state(
    key, 
    lambda_1=1000000,
    lambda_2=1,
    lambda_3=1,
    learning_rate=1e-3,
    threshold=0.8,
    sdf_pretrain=None, # "sphere" or "plane"
    cryoET_data=None,
    init_ckpt=None,
    radius=None):

    """Initializes the model, parameters, optimizer, and loss weights inside TrainState."""
    model = PINN()  # Create model instance
    params = model.init(key, jnp.ones((1, 3)))  # Initialize model parameters

    optimizer = optax.adam(learning_rate)  # Adam optimizer

    # Perform SDF pretraining before training
    if sdf_pretrain is not None:
        print("Pretraining the network using SDF...")
        grid_points, sdf_initial = generate_sdf(kind=sdf_pretrain, radius=radius)

        # Select random training points
        train_idx = np.random.choice(grid_points.shape[0], 10000, replace=False)
        # train_idx = np.random.choice(grid_points.shape[0], grid_points.shape[0], replace=False)
        x_train = grid_points[train_idx]
        y_train = sdf_initial.ravel()[train_idx]

        params = initialize_network_with_sdf(model, params, y_train, x_train)

    # Set initial condition based on the binary input image
    if cryoET_data is not None:
        print("Pretraining the network using MRC")

        x_grid, y_grid, z_grid = jnp.meshgrid(
            jnp.linspace(-1, 1, GRID_SIZE),
            jnp.linspace(-1, 1, GRID_SIZE),
            jnp.linspace(-1, 1, GRID_SIZE),
            indexing="ij"
        )
        grid_points = jnp.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=-1)
        binary_mask = jnp.where(cryoET_data > threshold, 0, 1).ravel()

        params = initialize_network_with_sdf(model, params, binary_mask, grid_points, steps=10000, learning_rate=1e-3)

    # Use pretrained network structure as initial condition
    if init_ckpt is not None:
        print("Use pretrained data as initial condition...")
        # params = init_ckpt["params"]
        params = init_ckpt['state']['params']


    return TrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        opt_state=optimizer.init(params),
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
        threshold=threshold
    ), model


def generate_sdf(
    grid_size=64,
    kind="plane",                   # "sphere" or "plane" or "multi" or "uniform"
    radius=None,                     # used for sphere
    epsilon=0.05,                   # smoothing width for tanh
):
    if radius is None:
        radius = 0.5

    # Coordinate grid in [-1, 1]^3
    x, y, z = jnp.meshgrid(
        jnp.linspace(-1, 1, grid_size),
        jnp.linspace(-1, 1, grid_size),
        jnp.linspace(-1, 1, grid_size),
        indexing="ij"
    )

    if kind == "sphere":
        sdf_values = jnp.sqrt(x**2 + y**2 + z**2) - radius

    elif kind == "plane":
        # sdf_values = x
        sdf_values = x + 0.5

    elif kind == "multi":
        d1 = jnp.sqrt((x-0.5)**2 + y**2 + z**2) - 0.4
        d2 = jnp.sqrt((x+0.5)**2 + y**2 + z**2) - 0.4
        sdf_values = jnp.where(jnp.abs(d1) < jnp.abs(d2), d1, d2)
    elif kind == "uniform":
        sdf_values = jnp.ones_like(x)

    else:
        raise ValueError("kind must be 'sphere', 'plane', 'multi', or 'uniform'")

    sdf_values = -jnp.tanh(sdf_values/epsilon)

    # Flatten grid points for training
    grid_points = jnp.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    sdf_values = sdf_values.ravel()  # Flatten the SDF values

    return grid_points, sdf_values.reshape(grid_size, grid_size, grid_size)



def initialize_network_with_sdf(model, params, sdf_values, grid_points, learning_rate=1e-3, steps=5000):
    """Optimize initial weights so that the PINN approximates the given SDF before training."""
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jit
    def loss_fn(p):
        """Compute L2 loss between the model prediction and the known SDF."""
        predictions = model.apply(p, grid_points)
        return jnp.mean((predictions - sdf_values)**2)  # Mean Squared Error (MSE)

    @jit
    def train_step(p, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, opt_state = optimizer.update(grads, opt_state)
        p = optax.apply_updates(p, updates)

        return p, opt_state, loss

    # Optimization loop
    for step in range(steps):
        params, opt_state, loss_val = train_step(params, opt_state)

        if step % 20 == 0:
            # print(f"Pre-training Step {step}, Loss: {loss_val:.6f}")
            predictions = model.apply(params, grid_points)
            # print(f"Pre-training Step {step}, Loss: {loss_val:.6f}, "
              # f"Pred min: {predictions.min():.6f}, Pred max: {predictions.max():.6f}")

    return params



# Training step function
@jit
# def train_step(state, x_train, cryoET_data, data_sign):
def train_step(state, x_train, data_edge, data_sign):
    """ Performs one training step, computing gradients and updating parameters. """

    def compute_losses(params):
        phi_fn = lambda x: state.apply_fn(params, x.reshape(-1, 3))
        # loss_data_val = loss_data(phi_fn, cryoET_data, state.threshold)
        loss_data_val = loss_data(phi_fn, data_edge)
        loss_physics_val = loss_physics(phi_fn, x_train)
        loss_sign_val = loss_sign(phi_fn, data_sign)
        total_loss_val = state.lambda_1 * loss_data_val + state.lambda_2 * loss_physics_val + state.lambda_3 * loss_sign_val
        return total_loss_val, (loss_data_val, loss_physics_val, loss_sign_val)

    (loss, aux_losses), grads = jax.value_and_grad(compute_losses, has_aux=True)(state.params)

    new_state = state.apply_gradients(grads=grads)  # Update parameters using gradients
    loss_data_val, loss_physics_val, loss_sign_val = aux_losses

    return new_state, loss, loss_data_val, loss_physics_val, loss_sign_val




