import os
import jax
import jax.numpy as jnp
import optax
from jax import jit, lax
from flax.training import train_state
from pinn.model import PINN, loss_data, loss_physics, LEARNING_RATE, LAMBDA_1, LAMBDA_2, EPSILON
from pinn.checkpoint_io import _ensure_dir, load_state_bytes, save_params_bytes
import numpy as np
from typing import NamedTuple
from typing import Callable
from dataclasses import dataclass
from typing import Callable, NamedTuple

# Define TrainState for managing training parameters and optimizer
class TrainState(train_state.TrainState):
    lambda_1: float  # Weight for loss_data
    lambda_2: float  # Weight for loss_physics


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
    learning_rate=LEARNING_RATE, 
    lambda_1=LAMBDA_1, 
    lambda_2=LAMBDA_2, 
    sdf_pretrain=None, # "sphere" or "plane"
    init_ckpt=None,
    sdf_cache_dir=None
    ):
    """Initializes the model, parameters, optimizer, and loss weights inside TrainState."""
    model = PINN()  # Create model instance
    params = model.init(key, jnp.ones((1, 3)))  # Initialize model parameters

    optimizer = optax.adam(learning_rate)  # Adam optimizer

    # Perform SDF pretraining before training
    if sdf_pretrain is not None:
        _ensure_dir(sdf_cache_dir)
        cache_path = os.path.join(sdf_cache_dir, f"{sdf_pretrain}.msgpack")

        if os.path.isfile(cache_path):
            params = load_state_bytes(cache_path, params)
        else:
            print("Pretraining the network using SDF...")
            grid_points, sdf_initial = generate_sdf(kind=sdf_pretrain)

            # Select random training points
            train_idx = np.random.choice(grid_points.shape[0], 10000, replace=False)
            x_train = grid_points[train_idx]
            y_train = sdf_initial.ravel()[train_idx]

            params = initialize_network_with_sdf(model, params, y_train, x_train)
            print(f"[SDF cache] Saving pretrained params to: {cache_path}")
            save_params_bytes(cache_path, params)

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



def generate_sdf_ori(grid_size=64, radius=0.5, epsilon=0.05):
    """Generate a signed distance function (SDF) for a sphere of given radius.

    grid_size may be an int (cubic) or a 3-tuple (nx, ny, nz). The returned
    grid_points are stacked as (x, y, z) and the volume is reshaped to
    (nx, ny, nz) which matches meshgrid(indexing='ij') ordering.
    """
    # Normalize grid_size to (nx, ny, nz)
    if isinstance(grid_size, int):
        nx = ny = nz = int(grid_size)
    elif isinstance(grid_size, (tuple, list)) and len(grid_size) == 3:
        nx, ny, nz = map(int, grid_size)
    else:
        raise ValueError("grid_size must be an int or a 3-tuple")

    # Define voxel-based coordinate grid
    x, y, z = jnp.meshgrid(
        jnp.linspace(-1, 1, nx),
        jnp.linspace(-1, 1, ny),
        jnp.linspace(-1, 1, nz),
        indexing="ij"
    )

    # Compute SDF for a sphere (normalized)
    sdf_values = jnp.sqrt(x**2 + y**2 + z**2) - radius
    sdf_values = -jnp.tanh(sdf_values / epsilon)

    # Flatten grid points for training
    grid_points = jnp.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    sdf_values = sdf_values.ravel()

    return grid_points, sdf_values.reshape(nx, ny, nz)



def generate_sdf(
    grid_size=64,
    kind="plane",                  # "sphere" or "plane"
    radius=0.5,                     # used for sphere
    epsilon=EPSILON,                # smoothing width for tanh
):
    """Generate an SDF on a grid. Accepts an int or a (nx,ny,nz) tuple.

    Returns: (grid_points, sdf_volume) where grid_points.shape == (N,3)
    and sdf_volume.shape == (nx, ny, nz) (meshgrid ordering 'ij').
    """
    # Normalize grid_size to (nx, ny, nz)
    if isinstance(grid_size, int):
        nx = ny = nz = int(grid_size)
    elif isinstance(grid_size, (tuple, list)) and len(grid_size) == 3:
        nx, ny, nz = map(int, grid_size)
    else:
        raise ValueError("grid_size must be an int or a 3-tuple")

    # Coordinate grid in [-1, 1]^3 with meshgrid(indexing='ij') -> shape (nx,ny,nz)
    x, y, z = jnp.meshgrid(
        jnp.linspace(-1, 1, nx),
        jnp.linspace(-1, 1, ny),
        jnp.linspace(-1, 1, nz),
        indexing="ij"
    )

    kind = str(kind).lower().strip().replace("-", "_")

    print(kind)

    if kind == "sphere":
        sdf_values = jnp.sqrt(x**2 + y**2 + z**2) - 0.8

    elif kind == "sine":
        r = jnp.sqrt(x**2 + y**2 + z**2)
        base_radius = radius
        layer_thickness = radius * 0.5  # 층 두께 조절

        # sin() 기반으로 주기적인 sign flip → 동심 링 패턴
        sdf_values = jnp.sin(jnp.pi * (r - base_radius) / layer_thickness) * (r - base_radius)

    elif kind == "double":
        r = jnp.sqrt(x**2 + y**2 + z**2)
        sdf_values = jnp.abs(r - radius*1.3) - (radius - radius*0.8)

    elif kind == "plane":
        # sdf_values = x
        sdf_values = x + 0.5

    elif kind == "multi":
        d1 = jnp.sqrt((x-0.5)**2 + y**2 + z**2) - 0.4
        d2 = jnp.sqrt((x+0.5)**2 + y**2 + z**2) - 0.4
        sdf_values = jnp.where(jnp.abs(d1) < jnp.abs(d2), d1, d2)

    else:
        raise ValueError("kind must be 'sphere', 'plane', 'double', 'sine', or 'multi'")

    sdf_values = -jnp.tanh(sdf_values / epsilon)

    # Flatten grid points for training
    grid_points = jnp.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    sdf_values = sdf_values.ravel()

    return grid_points, sdf_values.reshape(nx, ny, nz)






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
def train_step(state, x_train, cryoET_data, thre, phy_app):
# def train_step(state, x_train, cryoET_data, membrane_indices):
    """ Performs one training step, computing gradients and updating parameters. """

    def compute_losses(params):
        phi_fn = lambda x: state.apply_fn(params, x.reshape(-1, 3))
        loss_data_val = loss_data(phi_fn, cryoET_data, thre)

        pred = jnp.asarray(phy_app).astype(bool)  # ensure JAX scalar bool
        loss_physics_val = lax.cond(
            pred,
            lambda _: loss_physics(phi_fn, x_train, cryoET_data.shape),
            lambda _: jnp.array(1.0, dtype=jnp.float32),
            operand=None
        )
        total_loss_val = state.lambda_1 * loss_data_val + state.lambda_2 * loss_physics_val
        return total_loss_val, (loss_data_val, loss_physics_val)

    (loss, aux_losses), grads = jax.value_and_grad(compute_losses, has_aux=True)(state.params)

    new_state = state.apply_gradients(grads=grads)  # Update parameters using gradients
    loss_data_val, loss_physics_val = aux_losses

    return new_state, loss, loss_data_val, loss_physics_val




