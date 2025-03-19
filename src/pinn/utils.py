import numpy as np
from pinn.model import loss_data, loss_physics, total_loss

def initial_loss(state, x_train, cryoET_data):
    """
    Compute the actual initial loss values before training starts.
    Returns a structured dictionary compatible with checkpoint storage.
    """

    # Define the function using current model parameters
    params = state.params
    phi_fn = lambda x: state.apply_fn(params, x.reshape(-1, 3))

    # Compute the losses
    loss_data_val = loss_data(phi_fn, cryoET_data)
    loss_physics_val = loss_physics(phi_fn, x_train)
    total_loss_val = total_loss(phi_fn, x_train, cryoET_data, state.lambda_1, state.lambda_2)

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


