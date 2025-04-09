import jax
import jax.numpy as jnp
import optax
from jax import jit
from flax.training import train_state
from pinn.model import PINN, loss_data, loss_physics, total_loss
import numpy as np

# Define TrainState for managing training parameters and optimizer
class TrainState(train_state.TrainState):
    lambda_1: float  # Weight for loss_data
    lambda_2: float  # Weight for loss_physics

# # Initialize model and training state
# def create_train_state(key, learning_rate=1e-3, lambda_1=100000.0, lambda_2=0.001):
#     """Initializes the model, parameters, optimizer, and loss weights inside TrainState."""
#     model = PINN()  # Create model instance
#     params = model.init(key, jnp.ones((1, 3)))  # Initialize model parameters
#     optimizer = optax.adam(learning_rate)  # Adam optimizer

#     return TrainState(
#         step=0,
#         apply_fn=model.apply,
#         params=params,
#         tx=optimizer,
#         opt_state=optimizer.init(params),
#         lambda_1=lambda_1,  # Store loss weight for data term
#         lambda_2=lambda_2   # Store loss weight for physics term
#     ), model


def create_train_state(key, learning_rate=1e-3, lambda_1=100000.0, lambda_2=0.001, sdf_pretrain=False):
    """Initializes the model, parameters, optimizer, and loss weights inside TrainState."""
    model = PINN()  # Create model instance
    params = model.init(key, jnp.ones((1, 3)))  # Initialize model parameters
    optimizer = optax.adam(learning_rate)  # Adam optimizer

    # Perform SDF pretraining before training
    if sdf_pretrain:
        print("Pretraining the network using SDF...")
        grid_points, sdf_initial = generate_sdf()

        # Select random training points
        train_idx = np.random.choice(grid_points.shape[0], 10000, replace=False)
        x_train = grid_points[train_idx]
        y_train = sdf_initial.ravel()[train_idx]

        params = initialize_network_with_sdf(model, params, y_train, x_train)


    return TrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        opt_state=optimizer.init(params),
        lambda_1=lambda_1,
        lambda_2=lambda_2
    ), model


# def generate_sdf(grid_size=64, radius=15):
#     """Generate a signed distance function (SDF) for a sphere of given radius."""
#     x, y, z = jnp.meshgrid(
#         jnp.linspace(-1.5, 1.5, grid_size),
#         jnp.linspace(-1.5, 1.5, grid_size),
#         jnp.linspace(-1.5, 1.5, grid_size),
#         indexing="ij"
#     )
#     grid_points = jnp.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
#     sdf_values = jnp.maximum(jnp.maximum(jnp.abs(x) - radius, jnp.abs(y) - radius), jnp.abs(z) - radius)
#     return grid_points, sdf_values.reshape(grid_size, grid_size, grid_size)


def generate_sdf(grid_size=64, radius=0.5):
    """Generate a signed distance function (SDF) for a sphere of given radius."""

    # Define voxel-based coordinate grid
    x, y, z = jnp.meshgrid(
        jnp.linspace(-1, 1, grid_size),
        jnp.linspace(-1, 1, grid_size),
        jnp.linspace(-1, 1, grid_size),
        indexing="ij"
    )

    # Compute SDF for a sphere (normalized)
    sdf_values = jnp.sqrt(x**2 + y**2 + z**2) - radius
    # sdf_values = jnp.abs(sdf_values)
    # sdf_values = -sdf_values

    # Flatten grid points for training
    grid_points = jnp.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    # sdf_values = sdf_values.ravel()  # Flatten the SDF values
    sdf_values = -sdf_values.ravel()  # Flatten the SDF values

    return grid_points, sdf_values.reshape(grid_size, grid_size, grid_size)



def initialize_network_with_sdf(model, params, sdf_values, grid_points, learning_rate=1e-3, steps=500):
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
            print(f"Pre-training Step {step}, Loss: {loss_val:.6f}, "
              f"Pred min: {predictions.min():.6f}, Pred max: {predictions.max():.6f}")

    return params



# Training step function
@jit
def train_step(state, x_train, cryoET_data):
    """ Performs one training step, computing gradients and updating parameters. """

    def compute_losses(params):
        phi_fn = lambda x: state.apply_fn(params, x.reshape(-1, 3))
        loss_data_val = loss_data(phi_fn, cryoET_data)
        loss_physics_val = loss_physics(phi_fn, x_train)
        total_loss_val = total_loss(phi_fn, x_train, cryoET_data, state.lambda_1, state.lambda_2)
        return total_loss_val, (loss_data_val, loss_physics_val)

    (loss, aux_losses), grads = jax.value_and_grad(compute_losses, has_aux=True)(state.params)

    new_state = state.apply_gradients(grads=grads)  # Update parameters using gradients
    loss_data_val, loss_physics_val = aux_losses

    return new_state, loss, loss_data_val, loss_physics_val




