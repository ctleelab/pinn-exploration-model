import jax
import jax.numpy as jnp
import optax
from jax import jit
from flax.training import train_state
from pinn.model import PINN, loss_data, loss_physics, total_loss
import numpy as np
from typing import NamedTuple

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


import jax, jax.numpy as jnp
import optax
from flax.training import train_state
from typing import Callable
from dataclasses import dataclass

def cosine_with_restarts(lr_max: float, lr_min: float, T0: int, T_mult: float=2.0) -> Callable[[int], float]:
    """SGDR schedule: cosine decay that restarts; step is a Python int or jnp scalar."""
    def schedule(step):
        step = jnp.asarray(step, dtype=jnp.float32)
        # compute which restart we're in and local step within it
        # (closed-form version of SGDR)
        # Find k s.t. step < T0 * (1 - T_mult**k) / (1 - T_mult)
        # Use while_loop to avoid Python control flow.
        def body_fun(carry):
            k, acc, Tk = carry
            return (k+1, acc + Tk, Tk*T_mult)
        def cond_fun(carry):
            k, acc, Tk = carry
            return (acc + Tk) <= step
        k0, acc0, Tk0 = jnp.int32(0), jnp.float32(0.0), jnp.float32(T0)
        k, acc, Tk = jax.lax.while_loop(cond_fun, body_fun, (k0, acc0, Tk0))
        # local position in current cycle
        t_local = step - acc
        cos = 0.5 * (1.0 + jnp.cos(jnp.pi * t_local / jnp.maximum(Tk, 1.0)))
        return lr_min + (lr_max - lr_min) * cos
    return schedule



# class AdditiveGaussianNoise(optax.GradientTransformation):
#     def __init__(self, noise_std: float, key):
#         self.noise_std = noise_std
#         self.key = key

#     def init(self, params):
#         return ()

#     def update(self, updates, state, params=None):
#         # Split PRNG for reproducibility
#         self.key, subkey = jax.random.split(self.key)
#         noisy_updates = jax.tree_util.tree_map(
#             lambda g: g + self.noise_std * jax.random.normal(subkey, g.shape),
#             updates
#         )
#         return noisy_updates, state


# def additive_gaussian_noise(noise_std: float, key):
#     return AdditiveGaussianNoise(noise_std, key)


from typing import Callable, NamedTuple

# ----- schedules -----
def sqrt_decay(sigma0: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    # σ(t) = σ0 / sqrt(t+1)
    return lambda step: sigma0 / jnp.sqrt(step + 1)

def exp_decay(sigma0: float, tau: float = 5000.0) -> Callable[[jnp.ndarray], jnp.ndarray]:
    # σ(t) = σ0 * exp(-t/τ)
    return lambda step: sigma0 * jnp.exp(-step / tau)

# ----- state -----
class _NoiseState(NamedTuple):
    rng: jax.Array            # PRNGKey array
    step: jnp.ndarray         # int32

# ----- transform factory -----
def additive_gaussian_noise_annealed(*, sigma0: float, key, mode: str = "sqrt", **kwargs) -> optax.GradientTransformation:
    """Add N(0, σ(t)^2) noise to updates; σ(t) anneals with step."""
    if mode == "sqrt":
        schedule = sqrt_decay(sigma0)
    elif mode == "exp":
        tau = float(kwargs.get("tau", 10000.0))
        schedule = exp_decay(sigma0, tau=tau)
    else:
        raise ValueError("mode must be 'sqrt' or 'exp'")

    def init_fn(params):
        return _NoiseState(rng=key, step=jnp.array(0, dtype=jnp.int32))

    def update_fn(updates, state: _NoiseState, params=None):
        sigma_t = schedule(state.step)

        leaves, treedef = jax.tree_util.tree_flatten(updates)
        # split RNG: one per leaf + carry
        splits = jax.random.split(state.rng, len(leaves) + 1)
        new_rng, ks = splits[0], splits[1:]

        noisy_leaves = [
            u + sigma_t * jax.random.normal(k, u.shape) for u, k in zip(leaves, ks)
        ]
        noisy_updates = jax.tree_util.tree_unflatten(treedef, noisy_leaves)
        new_state = _NoiseState(rng=new_rng, step=state.step + 1)
        return noisy_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)




def create_train_state(
    key, 
    learning_rate=1e-3, 
    lambda_1=100000.0, 
    lambda_2=0.001, 
    sdf_pretrain=None, # "sphere" or "plane"
    init_ckpt=None):
    """Initializes the model, parameters, optimizer, and loss weights inside TrainState."""
    model = PINN()  # Create model instance
    params = model.init(key, jnp.ones((1, 3)))  # Initialize model parameters

    optimizer = optax.adam(learning_rate)  # Adam optimizer

    # Cosine Annealing Schedule
    # lr_sched = cosine_with_restarts(lr_max=5e-3, lr_min=1e-5, T0=1000, T_mult=1.0)
    # lr_sched = cosine_with_restarts(lr_max=1e-2, lr_min=1e-4, T0=1000, T_mult=1.0)
    # lr_sched = cosine_with_restarts(lr_max=5e-3, lr_min=1e-5, T0=2000, T_mult=1.0)
    # lr_sched = cosine_with_restarts(lr_max=1e-3, lr_min=1e-5, T0=1000, T_mult=1.0)
    # optimizer = optax.adam(learning_rate=lr_sched)  # Adam with scheduled LR

    # key = jax.random.PRNGKey(0)
    # optimizer = optax.chain(
    #     optax.adam(learning_rate),
    #     # additive_gaussian_noise(noise_std=1e-4, key=key)
    #     additive_gaussian_noise_annealed(sigma0=1e-3, key=key, mode="exp") 
    # )



    # Perform SDF pretraining before training
    if sdf_pretrain is not None:
        print("Pretraining the network using SDF...")
        grid_points, sdf_initial = generate_sdf(kind=sdf_pretrain)

        # Select random training points
        train_idx = np.random.choice(grid_points.shape[0], 10000, replace=False)
        x_train = grid_points[train_idx]
        y_train = sdf_initial.ravel()[train_idx]

        params = initialize_network_with_sdf(model, params, y_train, x_train)

    # Use pretrained network structure as initial condition
    if init_ckpt is not None:
        print("Use pretrained data as initial condition...")
        params = init_ckpt["params"]
        # params = init_ckpt['state']['params']


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


def generate_sdf_ori(grid_size=64, radius=0.5, epsilon=0.05):
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
    # sdf_values = -jnp.tanh(sdf_values)
    sdf_values = -jnp.tanh(sdf_values/epsilon)
    # sdf_values = jnp.abs(sdf_values)
    # sdf_values = -sdf_values

    # Flatten grid points for training
    grid_points = jnp.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    sdf_values = sdf_values.ravel()  # Flatten the SDF values

    return grid_points, sdf_values.reshape(grid_size, grid_size, grid_size)



def generate_sdf(
    grid_size=64,
    kind="plane",                  # "sphere" or "plane"
    radius=0.5,                     # used for sphere
    epsilon=0.05,                   # smoothing width for tanh
):

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
    else:
        raise ValueError("kind must be 'sphere' or 'plane'")

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
def train_step(state, x_train, cryoET_data):
# def train_step(state, x_train, cryoET_data, membrane_indices):
    """ Performs one training step, computing gradients and updating parameters. """

    def compute_losses(params):
        phi_fn = lambda x: state.apply_fn(params, x.reshape(-1, 3))
        loss_data_val = loss_data(phi_fn, cryoET_data)
        # loss_data_val = loss_data(phi_fn, cryoET_data, membrane_indices)
        loss_physics_val = loss_physics(phi_fn, x_train)
        # total_loss_val = total_loss(phi_fn, x_train, cryoET_data, state.lambda_1, state.lambda_2)
        # total_loss_val = total_loss(phi_fn, x_train, cryoET_data, state.lambda_1, state.lambda_2, membrane_indices)
        total_loss_val = state.lambda_1 * loss_data_val + state.lambda_2 * loss_physics_val
        return total_loss_val, (loss_data_val, loss_physics_val)

    (loss, aux_losses), grads = jax.value_and_grad(compute_losses, has_aux=True)(state.params)

    new_state = state.apply_gradients(grads=grads)  # Update parameters using gradients
    loss_data_val, loss_physics_val = aux_losses

    return new_state, loss, loss_data_val, loss_physics_val




