import numpy as np
from pinn.model import PINN
import optax
from flax.training.train_state import TrainState
import matplotlib.pyplot as plt

import os
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer


class TrainState(TrainState):
    pass

def create_checkpoint_manager(ckpt_dir, max_to_keep=10):
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    options = CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    checkpointer = PyTreeCheckpointer()
    return CheckpointManager(ckpt_dir, checkpointer, options)

def generate_phase_field(grid_bounds=(-1, 1), grid_size=128, R=0.5, epsilon=0.05):
    """
    Generates a 3D scalar field φ(x, y, z) with a smooth interface across a sphere.

    Args:
        grid_bounds (tuple): The min and max values for x, y, z axes (default = (-1, 1)).
        grid_size (int): Number of grid points along each axis (default = 128).
        R (float): Radius of the inner sphere where φ ≈ 1.
        epsilon (float): Interface thickness (transition width).

    Returns:
        phi (np.ndarray): 3D array of shape (grid_size, grid_size, grid_size).
        coords (tuple): Tuple of 1D arrays (x, y, z) used to construct the grid.
    """
    x = np.linspace(grid_bounds[0], grid_bounds[1], grid_size)
    y = np.linspace(grid_bounds[0], grid_bounds[1], grid_size)
    z = np.linspace(grid_bounds[0], grid_bounds[1], grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Distance from the center
    r = np.sqrt(X**2 + Y**2 + Z**2)

    # Phase field with smooth transition
    phi = np.tanh((R - r) / (np.sqrt(2) * epsilon))

    return phi, (x, y, z)


import matplotlib.pyplot as plt

def visualize_phase_slice(phi, x, y, z, axis='z', slice_index=None, cmap='coolwarm'):
    """
    Visualize a 2D slice of the 3D scalar field φ(x, y, z).

    Args:
        phi (np.ndarray): 3D array of phase field values.
        x, y, z (np.ndarray): 1D coordinate arrays used to generate φ.
        axis (str): Axis along which to slice ('x', 'y', or 'z'). Default is 'z'.
        slice_index (int or None): Index of the slice. If None, use the middle slice.
        cmap (str): Colormap for visualization. Default is 'coolwarm'.

    Returns:
        None
    """
    assert axis in ['x', 'y', 'z'], "Axis must be 'x', 'y', or 'z'"
    
    # Determine slice index if not provided
    if slice_index is None:
        slice_index = phi.shape['xyz'.index(axis)] // 2

    if axis == 'z':
        img = phi[:, :, slice_index]
        extent = [x[0], x[-1], y[0], y[-1]]
        xlabel, ylabel = 'x', 'y'
    elif axis == 'y':
        img = phi[:, slice_index, :]
        extent = [x[0], x[-1], z[0], z[-1]]
        xlabel, ylabel = 'x', 'z'
    elif axis == 'x':
        img = phi[slice_index, :, :]
        extent = [y[0], y[-1], z[0], z[-1]]
        xlabel, ylabel = 'y', 'z'

    plt.figure(figsize=(6, 5))
    # plt.imshow(img.T, origin='lower', extent=extent, cmap=cmap, aspect='equal')
    plt.imshow(img.T, origin='lower', extent=extent, cmap=cmap, aspect='equal', vmin=-1, vmax=1)
    plt.colorbar(label='φ')
    plt.title(f'Mid-slice of φ(x, y, z) along {axis.upper()}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()



def compute_volume_fdm(phi, x, y, z):
    """
    Computes the approximate volume of the φ ≈ 1 region using a smooth indicator.

    Args:
        phi (np.ndarray): 3D array of phase field values.
        x, y, z (np.ndarray): 1D coordinate arrays used to construct the grid.

    Returns:
        volume (float): Approximate volume of the φ ≈ 1 region.
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    dV = dx * dy * dz

    volume_integrand = 0.5 * (1 + phi)
    volume = np.sum(volume_integrand) * dV

    return volume



def compute_surface_area_fdm(phi, x, y, z, epsilon):
    """
    Computes the surface area using the Modica-Mortola phase-field approximation.

    Args:
        phi (np.ndarray): 3D array of phase field values.
        x, y, z (np.ndarray): 1D coordinate arrays.
        epsilon (float): Interface thickness.

    Returns:
        area (float): Approximate surface area.
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    dV = dx * dy * dz

    # Gradient components
    dphi_dx, dphi_dy, dphi_dz = np.gradient(phi, dx, dy, dz, edge_order=2)
    grad_phi_sq = dphi_dx**2 + dphi_dy**2 + dphi_dz**2

    # Potential term: (1 - phi^2)^2
    potential_term = (1 - phi**2)**2

    # Full integrand
    integrand = epsilon * grad_phi_sq + (0.5 / epsilon) * potential_term

    # Surface area via Modica-Mortola
    area = (3 / (4 * np.sqrt(2))) * np.sum(integrand) * dV

    return area


import jax
import jax.numpy as jnp

def analytic_phi_fn(x, R=0.5, epsilon=0.05):
    """
    Analytic definition of the phase field φ(x, y, z) for a sphere.

    Args:
        x (jnp.ndarray): Array of shape (..., 3) with spatial coordinates.
        R (float): Radius of the sphere.
        epsilon (float): Interface thickness.

    Returns:
        jnp.ndarray: Scalar field φ at each point.
    """
    r = jnp.linalg.norm(x, axis=-1)
    return jnp.tanh((R - r) / (jnp.sqrt(2) * epsilon))

def compute_volume_autodiff(phi_fn, x_batch, V_box=8.0):
    """
    Computes volume using autodiff-compatible φ(x) function.

    Args:
        phi_fn (callable): Function mapping x (shape [N, 3]) → φ (shape [N] or [N, 1]).
        x_batch (jnp.ndarray): Array of shape (N, 3) with evaluation points.
        V_box (float): Total volume of the domain.

    Returns:
        float: Approximated volume.
    """
    phi_vals = jax.vmap(phi_fn)(x_batch).squeeze()
    integrand = 0.5 * (1 + phi_vals)

    dV = V_box / x_batch.shape[0]
    volume = jnp.sum(integrand) * dV
    return volume


def compute_surface_area_autodiff(phi_fn, x_batch, epsilon, V_box=8.0):
    """
    Computes surface area using autodiff and a user-supplied φ(x) function.

    Args:
        phi_fn (callable): Function that maps x (shape [N, 3]) → φ (shape [N] or [N, 1]).
                           Must support JAX autodiff.
        x_batch (jnp.ndarray): Array of shape (N, 3) with evaluation points.
        epsilon (float): Interface thickness.
        V_box (float): Total volume of the domain.

    Returns:
        float: Approximated surface area using the Modica-Mortola functional.
    """
    # Compute gradient of φ using autodiff
    # grad_phi_single = lambda x: jax.grad(lambda x_: phi_fn(x_.reshape(1, -1))[0])(x)
    grad_phi_single = lambda x: jax.grad(lambda x_: phi_fn(x_.reshape(1, -1)).squeeze())(x)
    grad_phi = jax.vmap(grad_phi_single)(x_batch)
    phi_vals = jax.vmap(phi_fn)(x_batch).squeeze()

    # Compute integrand of Modica-Mortola surface area
    grad_phi_sq = jnp.sum(grad_phi**2, axis=1)
    potential_term = (1 - phi_vals**2)**2
    integrand = epsilon * grad_phi_sq + (0.5 / epsilon) * potential_term

    # Monte Carlo integration
    dV = V_box / x_batch.shape[0]
    area = (3 / (4 * jnp.sqrt(2))) * jnp.sum(integrand) * dV
    return area



def generate_phi_dataset_from_grid(grid_bounds=(-1, 1), grid_size=128, R=0.5, epsilon=0.05):
    """
    Generates (x, φ(x)) pairs from the phase field defined on a regular 3D grid.

    Args:
        grid_bounds (tuple): Bounds for each axis (default [-1, 1])
        grid_size (int): Number of grid points per axis
        R (float): Radius of the φ = 1 region
        epsilon (float): Transition thickness

    Returns:
        x_train (jnp.ndarray): Flattened coordinates of shape (N, 3)
        y_train (jnp.ndarray): Corresponding φ values of shape (N,)
    """

    # Use the existing generator
    phi, (x, y, z) = generate_phase_field(grid_bounds, grid_size, R, epsilon)

    # Flatten 3D grid and φ values
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    phi_flat = phi.ravel()

    # Convert to JAX arrays
    x_train = jnp.array(coords)
    y_train = jnp.array(phi_flat)

    return x_train, y_train


def generate_phi_dataset_random_sample(num_points=100000, grid_bounds=(-1, 1), R=0.5, epsilon=0.05, seed=0):
    """
    Randomly samples points in the 3D domain and evaluates φ(x) analytically.

    Args:
        num_points (int): Number of random points to sample.
        grid_bounds (tuple): Bounds of the domain for x, y, z.
        R (float): Radius of the φ = 1 region.
        epsilon (float): Transition thickness.
        seed (int): PRNG seed.

    Returns:
        x_train (jnp.ndarray): Sampled coordinates of shape (N, 3)
        y_train (jnp.ndarray): Corresponding φ values of shape (N,)
    """
    key = jax.random.PRNGKey(seed)
    x_train = jax.random.uniform(key, shape=(num_points, 3), minval=grid_bounds[0], maxval=grid_bounds[1])
    y_train = analytic_phi_fn(x_train, R=R, epsilon=epsilon)
    return x_train, y_train



from tqdm.notebook import trange

# def train_neural_network(learning_rate=1e-3, num_steps=1000, log_interval=100):
#     """
#     Initializes the model, optimizer, and trains it to mimic the analytic phase field φ(x).
    
#     Args:
#         learning_rate: optimizer learning rate
#         num_steps: number of training steps
        
#     Returns:
#         A TrainState with trained parameters
#         The model instance
#     """
#     model = PINN()
#     key = jax.random.PRNGKey(0)
#     params = model.init(key, jnp.ones((1, 3)))
#     optimizer = optax.adam(learning_rate)
#     opt_state = optimizer.init(params)

#     # x_train, y_train = generate_phi_dataset_from_grid()
#     x_train, y_train = generate_phi_dataset_random_sample(num_points=50000)


#     def loss_fn(p):
#         pred = model.apply(p, x_train).squeeze()
#         return jnp.mean((pred - y_train) ** 2)

#     losses = []
#     volume = []
#     areas  = []

#     for step in trange(1, num_steps + 1):
#         grads = jax.grad(loss_fn)(params)
#         updates, opt_state = optimizer.update(grads, opt_state)
#         params = optax.apply_updates(params, updates)

#         if step % log_interval == 0 or step == 1:
#             phi_fn = lambda x: model.apply(params, x)
#             loss_val = loss_fn(params).item()
#             volm_val = compute_volume_autodiff(phi_fn, x_train)
#             area_val = compute_surface_area_autodiff(phi_fn, x_train, epsilon=0.05)
#             losses.append(loss_val)
#             volume.append(volm_val)
#             areas. append(area_val)

#             print(f"Step {step}/{num_steps}, Loss: {loss_val:.6f}")

#     state = TrainState(
#         step=num_steps,
#         apply_fn=model.apply,
#         params=params,
#         tx=optimizer,
#         opt_state=opt_state,
#     )
#     return state, model, losses, volume, areas


def train_neural_network(
    learning_rate=1e-3, 
    num_steps=7000, 
    log_interval=100, 
    ckpt_dir="../outputs/logs/test_phase", 
    alpha=0.01,
    beta=0.001,
    switch_first=2000,
    switch_second=4000,
    ):
    """
    Trains the PINN model and logs loss, volume, and surface area.
    Periodically saves training state.
    """
    model = PINN()
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, 3)))
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    x_train, y_train = generate_phi_dataset_random_sample(num_points=50000)
    true_grad = jax.vmap(jax.grad(analytic_phi_fn))(x_train)  # shape [N, 3]
    true_lap  = jax.vmap(lambda x: jnp.trace(jax.hessian(analytic_phi_fn)(x)))(x_train)

    # For volume/area evaluation
    N = 100000
    key = jax.random.PRNGKey(42)
    x_batch = jax.random.uniform(key, (N, 3), minval=-1.0, maxval=1.0)
    V_box = 8.0
    epsilon = 0.05

    ## original version
    def loss_fn(p):
        pred = model.apply(p, x_train).squeeze()
        return jnp.mean((pred - y_train) ** 2)

    ## updated loss term
    def loss_grad(p):
        def model_fn(x):
            return model.apply(p, x).squeeze()

        pred_grad = jax.vmap(jax.grad(model_fn))(x_train)  # shape [N, 3]
        return jnp.mean(jnp.sum((pred_grad - true_grad) ** 2, axis=1))

    def loss_lap(p):
        def model_fn(x):
            return model.apply(p, x).squeeze()

        # Compute Laplacian: sum of second-order partial derivatives
        def lap_fn(x):
            return jnp.trace(jax.hessian(model_fn)(x))  # trace = sum of diagonal = Δf

        pred_lap = jax.vmap(lap_fn)(x_train)  # shape [N]
        return jnp.mean((pred_lap - true_lap) ** 2)


    def combined_loss_2(p):
        return loss_fn(p) + alpha * loss_grad(p)

    def combined_loss_3(p):
        return loss_fn(p) + alpha * loss_grad(p) + beta * loss_lap(p)

    training_logs = []

    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_manager = create_checkpoint_manager(ckpt_dir, max_to_keep=50)

    for step in trange(1, num_steps + 1):

        if step < switch_first:
            loss = loss_fn
        elif step < switch_second:
            loss = combined_loss_2
        else:
            loss = combined_loss_3

        grads = jax.grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if step % log_interval == 0 or step == 1:
            # Compute metrics
            loss_0 = loss_fn(params).item()
            loss_1 = loss_grad(params).item()
            loss_2 = loss_lap(params).item()
            loss_tot = loss(params).item()

            phi_fn = lambda x: model.apply(params, x)
            vol_val = compute_volume_autodiff(phi_fn, x_batch, V_box).item()
            area_val = compute_surface_area_autodiff(phi_fn, x_batch, epsilon, V_box).item()

            training_logs.append({
                'step'      : step,
                'loss_tot'  : loss_tot,
                'loss_fn'   : loss_0,
                'loss_grad' : loss_1,
                'loss_lap'  : loss_2,
                'volume'    : vol_val,
                'area'      : area_val,
            })

            # Save current state
            state = TrainState(
                step=step,
                apply_fn=model.apply,
                params=params,
                tx=optimizer,
                opt_state=opt_state,
            )
            ckpt_manager.save(step, state)

            print(f"Step {step}/{num_steps}, Loss: {loss_tot:.6f}, Volume: {vol_val:.4f}, Area: {area_val:.4f}")

    # Final state
    final_state = TrainState(
        step=num_steps,
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        opt_state=opt_state,
    )
    return final_state, model, training_logs



def visualize_phi_slice_from_nn(phi_fn, grid_bounds=(-1, 1), grid_size=128, z_value=0.0, cmap='coolwarm'):
    """
    Visualizes a 2D slice (at fixed z) of φ(x, y, z) from a trained NN φ_fn.

    Args:
        phi_fn (callable): Function mapping (N, 3) → (N,) scalar values.
        grid_bounds (tuple): Domain limits (default: [-1, 1]).
        grid_size (int): Resolution per axis.
        z_value (float): z-plane to slice at.
        cmap (str): Colormap.

    Returns:
        None (displays plot)
    """
    x = jnp.linspace(grid_bounds[0], grid_bounds[1], grid_size)
    y = jnp.linspace(grid_bounds[0], grid_bounds[1], grid_size)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    # Stack to shape (N^2, 3)
    xy_coords = jnp.stack([X.ravel(), Y.ravel(), jnp.full_like(X.ravel(), z_value)], axis=-1)
    
    # Evaluate φ on this slice
    phi_vals = jax.vmap(phi_fn)(xy_coords).reshape((grid_size, grid_size))

    # Plot
    plt.figure(figsize=(6, 5))
    # plt.imshow(phi_vals.T, extent=[-1, 1, -1, 1], origin='lower', cmap=cmap)
    plt.imshow(phi_vals.T, extent=[-1, 1, -1, 1], origin='lower', cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(label='φ')
    plt.title(f'φ(x, y, z={z_value}) from Neural Network')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()



import os
from flax.training import checkpoints

def save_checkpoint(state, ckpt_dir="../outputs/logs/test_phase", step=None):
    """
    Save the TrainState to a checkpoint directory using an absolute path.
    """
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(ckpt_dir, state, step or int(state.step), keep=3, overwrite=True)


def load_checkpoint(state, ckpt_dir="../outputs/logs/test_phase"):
    """
    Load TrainState from the latest checkpoint.

    Args:
        state: uninitialized TrainState object (used to match structure)
        ckpt_dir: path to the checkpoint directory

    Returns:
        Restored TrainState
    """
    return checkpoints.restore_checkpoint(ckpt_dir, state)



# def plot_training_logs(training_logs, log_interval=100):
#     """
#     Plots the training loss, volume, and surface area over training steps.
#     """

#     steps = [1] + list(range(log_interval, log_interval * len(training_logs), log_interval))

#     plt.figure(figsize=(15, 4))

#     # Plot Loss
#     plt.subplot(1, 3, 1)
#     plt.plot(steps, losses, marker='o')
#     plt.xlabel("Step")
#     plt.ylabel("Loss")
#     plt.yscale("log")
#     plt.title("Training Loss")

#     # Plot Volume
#     plt.subplot(1, 3, 2)
#     plt.plot(steps, volumes, marker='o')
#     plt.xlabel("Step")
#     plt.ylabel("Volume")
#     plt.yscale("log")
#     plt.title("Predicted Volume")

#     # Plot Surface Area
#     plt.subplot(1, 3, 3)
#     plt.plot(steps, areas, marker='o')
#     plt.xlabel("Step")
#     plt.ylabel("Surface Area")
#     plt.yscale("log")
#     plt.title("Predicted Surface Area")

#     plt.tight_layout()
#     plt.show()


import matplotlib.pyplot as plt

def plot_training_logs(training_logs, area_0=None, vol_0=None):
    steps      = [log['step']      for log in training_logs]
    loss_tot   = [log['loss_tot']  for log in training_logs]
    loss_fn    = [log['loss_fn']   for log in training_logs]
    loss_grad  = [log['loss_grad'] for log in training_logs]
    loss_lap   = [log['loss_lap']  for log in training_logs]
    volumes    = [log['volume']    for log in training_logs]
    areas      = [log['area']      for log in training_logs]

    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    # Top row
    axs[0, 0].plot(steps, loss_tot, color='black')
    axs[0, 0].set_title('Total Loss')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True)

    axs[0, 1].plot(steps, volumes, color='black')
    axs[0, 1].set_title('Volume')
    axs[0, 1].set_ylabel('Volume')
    axs[0, 1].grid(True)

    if vol_0:
        axs[0, 1].axhline(y=vol_0, color='gray', linestyle='--')

    axs[0, 2].plot(steps, areas, color='black')
    axs[0, 2].set_title('Area')
    axs[0, 2].set_ylabel('Area')
    axs[0, 2].grid(True)

    if area_0:
        axs[0, 2].axhline(y=area_0, color='gray', linestyle='--')


    # Bottom row
    axs[1, 0].plot(steps, loss_fn, color='black')
    axs[1, 0].set_title('Loss (0th)')
    axs[1, 0].set_xlabel('Step')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].grid(True)

    axs[1, 1].plot(steps, loss_grad, color='black')
    axs[1, 1].set_title('Loss (1st)')
    axs[1, 1].set_xlabel('Step')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].grid(True)

    axs[1, 2].plot(steps, loss_lap, color='black')
    axs[1, 2].set_title('Loss (2nd)')
    axs[1, 2].set_xlabel('Step')
    axs[1, 2].set_ylabel('Loss')
    axs[1, 2].grid(True)

    plt.tight_layout()
    plt.show()


