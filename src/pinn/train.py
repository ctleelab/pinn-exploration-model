import jax
import jax.numpy as jnp
import optax
from jax import jit
from flax.training import train_state
from pinn.model import PINN, loss_data, loss_physics, total_loss

# Define TrainState for managing training parameters and optimizer
class TrainState(train_state.TrainState):
    lambda_1: float  # Weight for loss_data
    lambda_2: float  # Weight for loss_physics

# Initialize model and training state
def create_train_state(key, learning_rate=1e-3, lambda_1=100000.0, lambda_2=0.001):
    """Initializes the model, parameters, optimizer, and loss weights inside TrainState."""
    model = PINN()  # Create model instance
    params = model.init(key, jnp.ones((1, 3)))  # Initialize model parameters
    optimizer = optax.adam(learning_rate)  # Adam optimizer

    return TrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        opt_state=optimizer.init(params),
        lambda_1=lambda_1,  # Store loss weight for data term
        lambda_2=lambda_2   # Store loss weight for physics term
    ), model


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




