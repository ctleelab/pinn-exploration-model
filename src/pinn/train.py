from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import optax
from flax.training import train_state

from pinn.model import PINN, loss_curv, loss_data, loss_phys, loss_sign



class TrainState(train_state.TrainState):
    """Training state containing model parameters and loss weights."""

    lambda_data: float
    lambda_phys: float
    lambda_sign: float
    lambda_curv: float


def create_train_state(
    key: jax.Array,
    lambda_data: float = 100_000.0,
    lambda_phys: float = 10.0,
    lambda_sign: float = 100_000.0,
    lambda_curv: float = 0.0,
    learning_rate: float = 1e-3,
    sdf_pretrain: str | None = None,
    radius: float | None = None,
    init_checkpoint: Mapping[str, Any] | None = None,
) -> tuple[TrainState, PINN]:
    """Create the PINN, initialize its parameters, and configure the optimizer.

    Parameters
    ----------
    key
        JAX random key used to initialize the model parameters.
    lambda_data
        Weight applied to the data loss.
    lambda_phys
        Weight applied to the physics loss.
    lambda_sign
        Weight applied to the sign loss.
    lambda_curv
        Weight applied to the curvature loss.
    learning_rate
        Adam optimizer learning rate.
    init_checkpoint
        Optional checkpoint used to initialize the model parameters. The
        checkpoint is expected to contain ``checkpoint["state"]["params"]``.

    Returns
    -------
    state
        Initialized training state.
    model
        PINN model instance.
    """
    model = PINN()
    params = model.init(key, jnp.ones((1, 3)))

    if sdf_pretrain is not None:
        print(f"Pretraining the network using '{sdf_pretrain}' initialization...")
        grid_points, sdf_initial = generate_sdf(kind=sdf_pretrain, radius=radius)
        train_idx = np.random.choice(grid_points.shape[0], 10000, replace=False)
        x_train = grid_points[train_idx]
        y_train = sdf_initial.ravel()[train_idx]

        params = initialize_network_with_sdf(model, params, y_train, x_train)

    if init_checkpoint is not None:
        print("Initializing model parameters from checkpoint.")
        params = init_checkpoint["state"]["params"]

    optimizer = optax.adam(learning_rate)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        lambda_data=lambda_data,
        lambda_phys=lambda_phys,
        lambda_sign=lambda_sign,
        lambda_curv=lambda_curv,
    )
    return state, model


def generate_sdf(
    grid_size=64,
    kind="plane",  # "sphere" or "plane" or "multi" or "uniform"
    radius=None,   # used for sphere
    epsilon=0.05, 
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
            predictions = model.apply(params, grid_points)

    return params



def compute_initial_losses(
    state: TrainState,
    data_edge: Mapping[str, jax.Array],
    data_sign: Mapping[str, jax.Array],
    data_phys: Mapping[str, jax.Array],
    data_curv: Mapping[str, jax.Array],
) -> dict[str, np.ndarray]:
    """Compute and format the losses before training begins."""

    def phi_fn(points: jax.Array) -> jax.Array:
        return state.apply_fn(
            state.params,
            points.reshape(-1, 3),
        )

    data_loss = loss_data(phi_fn, data_edge)
    phys_loss = loss_phys(phi_fn, data_phys)
    sign_loss = loss_sign(phi_fn, data_sign)
    curv_loss = loss_curv(phi_fn, data_curv)

    total_loss = (
        state.lambda_data * data_loss
        + state.lambda_phys * phys_loss
        + state.lambda_sign * sign_loss
        + state.lambda_curv * curv_loss
    )

    return {
        "step": np.asarray([0], dtype=np.int64),
        "total_loss": np.asarray([total_loss], dtype=np.float32),
        "data_loss": np.asarray([data_loss], dtype=np.float32),
        "phys_loss": np.asarray([phys_loss], dtype=np.float32),
        "sign_loss": np.asarray([sign_loss], dtype=np.float32),
        "curv_loss": np.asarray([curv_loss], dtype=np.float32),
    }


def make_train_step(use_curvature_loss: bool = False):
    """Create a JIT-compiled PINN training step.

    Parameters
    ----------
    use_curvature_loss
        Whether to evaluate and include the curvature loss. Disabling this
        avoids computing the expensive higher-order derivatives required by
        the curvature term.
    """

    def train_step(
        state: TrainState,
        data_edge: Mapping[str, jax.Array],
        data_sign: Mapping[str, jax.Array],
        data_phys: Mapping[str, jax.Array],
        data_curv: Mapping[str, jax.Array],
    ):
        def compute_losses(params):
            def phi_fn(points):
                return state.apply_fn(
                    params,
                    points.reshape(-1, 3),
                )

            data_loss = loss_data(phi_fn, data_edge)
            phys_loss = loss_phys(phi_fn, data_phys)
            sign_loss = loss_sign(phi_fn, data_sign)

            if use_curvature_loss:
                curv_loss = loss_curv(phi_fn, data_curv)
            else:
                curv_loss = jnp.zeros((), dtype=data_loss.dtype)

            total_loss = (
                state.lambda_data * data_loss
                + state.lambda_phys * phys_loss
                + state.lambda_sign * sign_loss
                + state.lambda_curv * curv_loss
            )

            losses = {
                "total_loss": total_loss,
                "data_loss": data_loss,
                "phys_loss": phys_loss,
                "sign_loss": sign_loss,
                "curv_loss": curv_loss,
            }

            return total_loss, losses

        (_, losses), gradients = jax.value_and_grad(
            compute_losses,
            has_aux=True,
        )(state.params)

        new_state = state.apply_gradients(grads=gradients)

        return new_state, losses

    return jax.jit(train_step, donate_argnums=(0,))

