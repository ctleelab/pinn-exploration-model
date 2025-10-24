import numpy as np
import jax
import jax.numpy as jnp
from pinn.model import loss_data_batched, loss_physics_batched, total_loss_batched

def make_phi_apply(state):
    # pts: [B,3] -> phi: [B]
    def phi_apply(pts):
        y = state.apply_fn(state.params, jnp.atleast_2d(pts))  # [B,1] or [1]
        return jnp.ravel(y)
    return jax.jit(phi_apply)


def initial_loss(state, x_train, cryoET_data):
# def initial_loss(state, x_train, cryoET_data, membrane_indices):
    """
    Compute the actual initial loss values before training starts.
    Returns a structured dictionary compatible with checkpoint storage.
    """

    # Define the function using current model parameters
    params = state.params
    phi_apply = make_phi_apply(state)

    # Compute the losses
    loss_data_val = loss_data_batched(
                        phi_apply, 
                        cryoET_data, 
                        cryoET_data.shape, 
                        batch_size=1_000_000,
                        threshold=0.8,
                        weight_in=0.8,
                        eps=1e-8)
    
    loss_physics_val = loss_physics_batched(
        phi_apply, x_train, epsilon=0.05, batch_size=65_536
    )

    total_loss_val = total_loss_batched(
        phi_apply, x_train, cryoET_data, state.lambda_1, state.lambda_2,
        grid_shape=cryoET_data.shape,
        batch_size_data=1_000_000,
        batch_size_phys=65_536,
        epsilon=0.05,
    )

    # Convert to structured format (single-step array)
    loss_log = {
        "step": np.array([0]),  # Ensure consistency with later steps
        "total_loss": np.array([total_loss_val]),
        "data_loss": np.array([loss_data_val]),
        "physics_loss": np.array([loss_physics_val])
    }

    return loss_log


def assemble_loss_history(checkpoint_data):
    """
    Assemble loss history from all available checkpoints.

    Args:
        checkpoint_data (dict): Dictionary containing loss values from multiple checkpoints.

    Returns:
        dict: Aggregated loss history with 'step', 'total_loss', 'data_loss', and 'physics_loss'.
    """
    assembled_loss = {"step": [], "total_loss": [], "data_loss": [], "physics_loss": []}

    for step, checkpoint in checkpoint_data.items():
        if 'loss' in checkpoint:
            loss_data = checkpoint['loss']
            assembled_loss["step"].extend(loss_data["step"].tolist())
            assembled_loss["total_loss"].extend(loss_data["total_loss"].tolist())
            assembled_loss["data_loss"].extend(loss_data["data_loss"].tolist())
            assembled_loss["physics_loss"].extend(loss_data["physics_loss"].tolist())

    # Convert lists to NumPy arrays for easier handling
    for key in assembled_loss:
        assembled_loss[key] = np.array(assembled_loss[key])

    return assembled_loss


