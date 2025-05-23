import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit
from jax import vmap
from flax.training import train_state
from pinn.model import PINN, grad_phi, laplacian_phi
from pinn.train import generate_sdf, initialize_network_with_sdf
import matplotlib.pyplot as plt


# Define TrainState for managing training parameters and optimizer
class TrainState(train_state.TrainState):
    lambda_phys: float
    lambda_volm: float
    lambda_surf: float
    lambda_cent: float
    epsilon: 	 float
    V_0: 		 float
    A_0: 		 float


def create_train_state(
    key, 
    learning_rate=1e-3, 
    lambda_phys=1, 
    lambda_volm=1, 
    lambda_surf=1, 
    lambda_cent=1, 
    epsilon=0.05, 
    V_0=0.5, 
    A_0=3.1, 
    sdf_pretrain=True,
    init_ckpt=None):
    """Initializes the model, parameters, optimizer, and loss weights inside TrainState."""
    model = PINN()  # Create model instance
    params = model.init(key, jnp.ones((1, 3)))  # Initialize model parameters
    optimizer = optax.adam(learning_rate)  # Adam optimizer

    # Perform SDF pretraining before training
    if sdf_pretrain:
        print("Pretraining the network using SDF...")
        grid_points, sdf_initial = generate_sdf()
        sdf_initial = -sdf_initial

        # Select random training points
        train_idx = np.random.choice(grid_points.shape[0], 10000, replace=False)
        x_train = grid_points[train_idx]
        y_train = sdf_initial.ravel()[train_idx]

        params = initialize_network_with_sdf(model, params, y_train, x_train)

    # Use pretrained network structure as initial condition
    if init_ckpt is not None:
        print("Use pretrained data as initial condition...")
        params = init_ckpt["params"]
        # state = init_ckpt["state"]
        # params = state["params"]


    return TrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        opt_state=optimizer.init(params),
        lambda_phys=lambda_phys,
        lambda_volm=lambda_volm,
        lambda_surf=lambda_surf,
        lambda_cent=lambda_cent,
        epsilon=epsilon,
        V_0=V_0,
        A_0=A_0,
    ), model



@jit
def train_step(state, x_train):
    """ Performs one training step, computing gradients and updating parameters. """
    def compute_losses(params):
        phi_fn = lambda x: state.apply_fn(params, x.reshape(-1, 3))
        loss_phys_val = loss_physics(phi_fn, x_train, state.epsilon)
        loss_volm_val = loss_volume (phi_fn, x_train, state.V_0)
        loss_surf_val = loss_surface(phi_fn, x_train, state.epsilon, state.A_0)
        loss_cent_val = loss_center (phi_fn, x_train)
        total_loss_val =  state.lambda_phys * loss_phys_val 
        total_loss_val += state.lambda_volm * loss_volm_val 
        total_loss_val += state.lambda_surf * loss_surf_val
        total_loss_val += state.lambda_cent * loss_cent_val

        return total_loss_val, (loss_phys_val, loss_volm_val, loss_surf_val, loss_cent_val)

    (loss, aux_losses), grads = jax.value_and_grad(compute_losses, has_aux=True)(state.params)

    new_state = state.apply_gradients(grads=grads)  # Update parameters using gradients
    loss_phys_val, loss_volm_val, loss_surf_val, loss_cent_val = aux_losses

    return new_state, loss, loss_phys_val, loss_volm_val, loss_surf_val, loss_cent_val



def initial_loss(state, x_train):
    """
    Compute the actual initial loss values before training starts.
    Returns a structured dictionary compatible with checkpoint storage.
    """

    # Define the function using current model parameters
    params = state.params
    phi_fn = lambda x: state.apply_fn(params, x.reshape(-1, 3))

    # Compute the losses
    loss_phys_val = loss_physics(phi_fn, x_train, state.epsilon)
    loss_volm_val = loss_volume (phi_fn, x_train, state.V_0)
    loss_surf_val = loss_surface(phi_fn, x_train, state.epsilon, state.A_0)
    loss_cent_val = loss_center (phi_fn, x_train)

    total_loss_val =  state.lambda_phys * loss_phys_val
    total_loss_val += state.lambda_volm * loss_volm_val
    total_loss_val += state.lambda_surf * loss_surf_val 
    total_loss_val += state.lambda_cent * loss_cent_val

    # Convert to structured format (single-step array)
    loss_log = {
        "step": np.array([0]),  # Ensure consistency with later steps
        "total_loss": np.array([total_loss_val]),
        "phys_loss": np.array([loss_phys_val]),
        "volm_loss": np.array([loss_volm_val]),
        "surf_loss": np.array([loss_surf_val]),
        "cent_loss": np.array([loss_cent_val]),
    }

    return loss_log


def loss_physics(phi_fn, x, epsilon, V_box=8):
    phi_vals = vmap(lambda x_i: phi_fn(jnp.atleast_2d(x_i)).squeeze())(x)
    lap_phi = laplacian_phi(phi_fn, x)

    value = epsilon * lap_phi - (1 / epsilon) * (phi_vals**2 - 1) * phi_vals
    return jnp.sum(value**2)

def loss_volume(phi_fn, x, V_0, V_box=8):
    phi_vals = vmap(lambda x_i: phi_fn(jnp.atleast_2d(x_i)).squeeze())(x)
    value = 1 + phi_vals
    volume = jnp.sum(value*0.5)
    volume *= V_box / x.shape[0]

    # jax.debug.print("Volume = {}", volume)
    # jax.debug.print("Residual = {}", volume-V_0)
    # jax.debug.print("Loss = {}", (volume-V_0)**2)

    return (volume - V_0)**2

def loss_surface(phi_fn, x, epsilon, A_0, V_box=8):
    phi_vals = vmap(lambda x_i: phi_fn(jnp.atleast_2d(x_i)).squeeze())(x)
    grad_val = grad_phi(phi_fn, x)
    sq_grad = jnp.sum(grad_val ** 2, axis=1)

    value = epsilon * sq_grad + (1 / 2.0 / epsilon) * (1 - phi_vals**2)**2
    value *= 3/4/jnp.sqrt(2)
    area = jnp.sum(value)
    area *= V_box / x.shape[0]
    return (area - A_0)**2

def loss_center(phi_fn, x, V_box=8):
    phi_vals = vmap(lambda r_i: phi_fn(jnp.atleast_2d(r_i)).squeeze())(x)
    weights = 0.5 * (1 + phi_vals)  # shape (N,)
    weighted_pos = x * weights[:, None]  # broadcast weights to shape (N, 3)

    R_val = jnp.sum(weighted_pos, axis=0)  # shape (3,)
    R_val *= V_box / x.shape[0]

    return jnp.dot(R_val, R_val)


def plot_loss_history_ax(ax, assembled_loss):
    """
    Plot the loss function over the entire training process.

    Args:
        assembled_loss (dict): Aggregated loss history containing 'step', 'total_loss', 'data_loss', 'physics_loss'.
    """
    
    # Plot the losses
    ax.plot(assembled_loss["step"], assembled_loss["phys_loss"], label='Physics Loss', marker='s', linestyle='-')
    ax.plot(assembled_loss["step"], assembled_loss["volm_loss"], label='Volume Loss', marker='s', linestyle='-')
    ax.plot(assembled_loss["step"], assembled_loss["surf_loss"], label='Area Loss', marker='s', linestyle='-')
    ax.plot(assembled_loss["step"], assembled_loss["cent_loss"], label='Center Loss', marker='s', linestyle='-')
    ax.plot(assembled_loss["step"], assembled_loss["total_loss"], label='Total Loss', marker='^', linestyle='-')

    # Use log scale for better visualization (especially if physics loss is large)
    ax.set_yscale('log')

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss Function Over Training')

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()



def plot_normalized_loss_history_ax(ax, id, assembled_loss):
    """
    Plot three normalized loss functions (Data Loss, Physics Loss, and Total Loss) side by side.
    Each function is scaled by its initial value at step = 0.

    Args:
        assembled_loss (dict): Aggregated loss history containing 'step', 'total_loss', 'data_loss', 'physics_loss'.
    """

    # Normalize loss values by their initial step=0 value
    phys_loss_norm = assembled_loss["phys_loss"] / assembled_loss["phys_loss"][0]
    volm_loss_norm = assembled_loss["volm_loss"] / assembled_loss["volm_loss"][0]
    surf_loss_norm = assembled_loss["surf_loss"] / assembled_loss["surf_loss"][0]
    cent_loss_norm = assembled_loss["cent_loss"] / assembled_loss["cent_loss"][0]
    total_loss_norm = assembled_loss["total_loss"] / assembled_loss["total_loss"][0]


    if id == 0:     # Plot Physics Loss
        ax.plot(assembled_loss["step"], phys_loss_norm, marker='o', linestyle='-')
        ax.set_yscale('log')
        ax.set_title("Normalized Physics Loss")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss Value (scaled)")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    elif id == 1:    # Plot Volume Loss
        ax.plot(assembled_loss["step"], volm_loss_norm, marker='s', linestyle='-')
        ax.set_yscale('log')
        ax.set_title("Normalized Volume Loss")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss Value (scaled)")        
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    elif id == 2:    # Plot Area Loss
        ax.plot(assembled_loss["step"], surf_loss_norm, marker='^', linestyle='-')
        ax.set_yscale('log')
        ax.set_title("Normalized Area Loss")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss Value (scaled)")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    elif id == 3:    # Plot Total Loss
        ax.plot(assembled_loss["step"], cent_loss_norm, marker='^', linestyle='-')
        ax.set_yscale('log')
        ax.set_title("Normalized Center Loss")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss Value (scaled)")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    elif id == 4:    # Plot Total Loss
        ax.plot(assembled_loss["step"], total_loss_norm, marker='^', linestyle='-')
        ax.set_yscale('log')
        ax.set_title("Normalized Total Loss")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss Value (scaled)")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

def assemble_loss_history(checkpoint_data):
    """
    Assemble loss history from all available checkpoints.

    Args:
        checkpoint_data (dict): Dictionary containing loss values from multiple checkpoints.

    Returns:
        dict: Aggregated loss history with 'step', 'total_loss', 'data_loss', and 'physics_loss'.
    """
    assembled_loss = {"step": [], "total_loss": [], "phys_loss": [], "volm_loss": [], "surf_loss": [], "cent_loss": [], }

    for step, checkpoint in checkpoint_data.items():
        if 'loss' in checkpoint:
            loss_data = checkpoint['loss']
            assembled_loss["step"].extend(loss_data["step"].tolist())
            assembled_loss["total_loss"].extend(loss_data["total_loss"].tolist())
            assembled_loss["phys_loss"].extend(loss_data["phys_loss"].tolist())
            assembled_loss["volm_loss"].extend(loss_data["volm_loss"].tolist())
            assembled_loss["surf_loss"].extend(loss_data["surf_loss"].tolist())
            assembled_loss["cent_loss"].extend(loss_data["cent_loss"].tolist())

    # Convert lists to NumPy arrays for easier handling
    for key in assembled_loss:
        assembled_loss[key] = np.array(assembled_loss[key])

    return assembled_loss




