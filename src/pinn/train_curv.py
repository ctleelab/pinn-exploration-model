import jax
import jax.numpy as jnp
import optax
from jax import jit
from flax.training import train_state
from pinn.model import PINN, loss_data, loss_phys, loss_sign, loss_curv, loss_lapH, loss_forc
from pinn.model import debug_grad_s_H_point, find_bad_points, find_bad_points_fast, find_bad_points_from_calc_grad_s_H, calc_grad_s_H, debug_bad_point
import numpy as np
from typing import NamedTuple
from functools import partial

# Define TrainState for managing training parameters and optimizer
class TrainState(train_state.TrainState):
    lambda_1: float  # Weight for loss_data
    lambda_2: float  # Weight for loss_physics
    lambda_3: float  # Weight for loss_sign
    lambda_4: float  # Weight for loss_curv
    lambda_5: float  # Weight for loss_lapH
    lambda_6: float  # Weight for loss_forc
    learning_rate: float
    warmup_steps: float  # Warmup steps for lambda_2

def create_train_state(
    key, 
    lambda_1=100000,
    lambda_2=10,
    lambda_3=100000,
    lambda_4=0,
    lambda_5=0,
    lambda_6=0,
    learning_rate=1e-3,
    warmup_steps=10000,
    sdf_pretrain=None,
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
        x_train = grid_points[train_idx]
        y_train = sdf_initial.ravel()[train_idx]

        params = initialize_network_with_sdf(model, params, y_train, x_train)

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
        lambda_4=lambda_4,
        lambda_5=lambda_5,
        lambda_6=lambda_6,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
    ), model


def initial_loss(state, data_edge, data_sign, data_phys, data_curv):
    """
    Compute the actual initial loss values before training starts.
    Returns a structured dictionary compatible with checkpoint storage.
    """

    # Define the function using current model parameters
    params = state.params
    phi_fn = lambda x: state.apply_fn(params, x.reshape(-1, 3))

    # Compute the losses
    loss_data_val = loss_data(phi_fn, data_edge)
    loss_phys_val = loss_phys(phi_fn, data_phys)
    loss_sign_val = loss_sign(phi_fn, data_sign)
    loss_curv_val = loss_curv(phi_fn, data_curv)
    loss_lapH_val = loss_lapH(phi_fn, data_curv)
    loss_forc_val = loss_forc(phi_fn, data_curv)

    total_loss_val = (
        state.lambda_1 *   loss_data_val
        + state.lambda_2 * loss_phys_val
        + state.lambda_3 * loss_sign_val
        + state.lambda_4 * loss_curv_val
        + state.lambda_5 * loss_lapH_val
        + state.lambda_6 * loss_forc_val
    )

    # Convert to structured format (single-step array)
    loss_log = {
        "step": np.array([0]),  # Ensure consistency with later steps
        "total_loss": np.array([total_loss_val]),
        "data_loss": np.array([loss_data_val]),
        "phys_loss": np.array([loss_phys_val]),
        "sign_loss": np.array([loss_sign_val]),
        "curv_loss": np.array([loss_curv_val]),
        "lapH_loss": np.array([loss_lapH_val]),
        "forc_loss": np.array([loss_forc_val])
    }

    # print("debugging...")
    # bad_list = find_bad_points(phi_fn, data_curv, n_check=50, normal_eps=1e-6, curvature_eps=1e-6)
    # bad_list = find_bad_points_fast(phi_fn, data_curv, n_check=None, normal_eps=1e-6, curvature_eps=1e-6)
    # bad_list = find_bad_points_from_calc_grad_s_H(phi_fn, data_curv)

    # x_bad = jnp.array([-0.9835849, -0.28222656, -0.9855232], dtype=jnp.float32)
    # debug_bad_point(phi_fn, x_bad, normal_eps=1e-6, curvature_eps=1e-6)

    # lc = loss_curv(phi_fn, data_curv)
    # lc = jax.device_get(lc)
    # print("lc =", loss_curv_val)

    return loss_log

def as_f32_scalar(x):
    # Make sure to extract as a scalar (float32) even if it's an array/DeviceArray
    return np.asarray(x, dtype=np.float32).reshape(()).item()

def loss_dict_to_batched(loss_list):
    """loss_list: list[dict(step=int, total_loss=..., ...)] -> dict of np arrays"""
    return {
        "step":       np.asarray([e["step"] for e in loss_list], dtype=np.int64),
        "total_loss": np.asarray([as_f32_scalar(e["total_loss"]) for e in loss_list], dtype=np.float32),
        "data_loss":  np.asarray([as_f32_scalar(e["data_loss"])  for e in loss_list], dtype=np.float32),
        "phys_loss":  np.asarray([as_f32_scalar(e["phys_loss"])  for e in loss_list], dtype=np.float32),
        "sign_loss":  np.asarray([as_f32_scalar(e["sign_loss"])  for e in loss_list], dtype=np.float32),
        "curv_loss":  np.asarray([as_f32_scalar(e["curv_loss"])  for e in loss_list], dtype=np.float32),
        "lapH_loss":  np.asarray([as_f32_scalar(e["lapH_loss"])  for e in loss_list], dtype=np.float32),
        "forc_loss":  np.asarray([as_f32_scalar(e["forc_loss"])  for e in loss_list], dtype=np.float32),
    }


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

# # Training step function
# @jit
# def train_step(state, data_edge, data_sign, data_phys):
#     """ Performs one training step, computing gradients and updating parameters. """

#     def compute_losses(params):
#         phi_fn = lambda x: state.apply_fn(params, x.reshape(-1, 3))
#         loss_data_val = loss_data(phi_fn, data_edge)
#         loss_phys_val = loss_phys(phi_fn, data_phys)
#         loss_sign_val = loss_sign(phi_fn, data_sign)
#         total_loss_val = state.lambda_1 * loss_data_val + state.lambda_2 * loss_phys_val + state.lambda_3 * loss_sign_val
#         return total_loss_val, (loss_data_val, loss_phys_val, loss_sign_val)

#     (loss, aux_losses), grads = jax.value_and_grad(compute_losses, has_aux=True)(state.params)

#     new_state = state.apply_gradients(grads=grads)  # Update parameters using gradients
#     loss_data_val, loss_phys_val, loss_sign_val = aux_losses

#     return new_state, loss, loss_data_val, loss_phys_val, loss_sign_val



def linear_schedule(step, max_value, warmup_steps):
    """
    Linearly increases from 0 → max_value over warmup_steps.
    """
    frac = jnp.clip(step / warmup_steps, 0.0, 1.0)
    return max_value * frac


# @partial(jax.jit, static_argnames=("schedule",))
# def train_step(state, step, data_edge, data_sign, data_phys, schedule=False):

#     if schedule:
#         lambda_2_step = linear_schedule(
#             step,
#             max_value=state.lambda_2,
#             warmup_steps=state.warmup_steps
#         )
#     else:
#         lambda_2_step = state.lambda_2


#     def compute_losses(params):
#         phi_fn = lambda x: state.apply_fn(params, x.reshape(-1, 3))

#         loss_data_val = loss_data(phi_fn, data_edge)
#         loss_phys_val = loss_phys(phi_fn, data_phys)
#         loss_sign_val = loss_sign(phi_fn, data_sign)

#         total_loss_val = (
#             state.lambda_1   * loss_data_val
#             + lambda_2_step  * loss_phys_val
#             + state.lambda_3 * loss_sign_val
#         )
#         return total_loss_val, (loss_data_val, loss_phys_val, loss_sign_val)

#     (loss, aux_losses), grads = jax.value_and_grad(compute_losses, has_aux=True)(state.params)

#     new_state = state.apply_gradients(grads=grads)
#     loss_data_val, loss_phys_val, loss_sign_val = aux_losses

#     return new_state, loss, loss_data_val, loss_phys_val, loss_sign_val

@jax.jit
def train_step_sched(state, step, data_edge, data_sign, data_phys, data_curv):
    lambda_2_step = linear_schedule(step, max_value=state.lambda_2, warmup_steps=state.warmup_steps)
    return _train_step_core(state, step, data_edge, data_sign, data_phys, data_curv, lambda_2_step)

@jax.jit
def train_step_nosched(state, step, data_edge, data_sign, data_phys, data_curv):
    lambda_2_step = state.lambda_2
    return _train_step_core(state, step, data_edge, data_sign, data_phys, data_curv, lambda_2_step)

# def _train_step_core(state, step, data_edge, data_sign, data_phys, data_curv, lambda_2_step):
#     def compute_losses(params):
#         phi_fn = lambda x: state.apply_fn(params, x.reshape(-1, 3))
#         ld = loss_data(phi_fn, data_edge)
#         lp = loss_phys(phi_fn, data_phys)
#         ls = loss_sign(phi_fn, data_sign)
#         lc = loss_curv(phi_fn, data_curv)
#         jax.debug.print("ld, lp, ls, lc = {}, {}, {}, {}", ld, lp, ls, lc)
#         total = state.lambda_1 * ld + lambda_2_step * lp + state.lambda_3 * ls + state.lambda_4 * lc
#         return total, (ld, lp, ls, lc)

#     (loss, (ld, lp, ls, lc)), grads = jax.value_and_grad(compute_losses, has_aux=True)(state.params)
#     new_state = state.apply_gradients(grads=grads)
#     return new_state, loss, ld, lp, ls, lc

def _tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum([jnp.sum(x * x) for x in leaves]))

def _train_step_core(state, step, data_edge, data_sign, data_phys, data_curv, lambda_2_step):
    def compute_losses(params):
        phi_fn = lambda x: state.apply_fn(params, x.reshape(-1, 3))
        ld = loss_data(phi_fn, data_edge)
        lp = loss_phys(phi_fn, data_phys)
        ls = loss_sign(phi_fn, data_sign)
        lc = loss_curv(phi_fn, data_curv)
        total = state.lambda_1 * ld + lambda_2_step * lp + state.lambda_3 * ls + state.lambda_4 * lc
        return total, (ld, lp, ls, lc)

    (loss, (ld, lp, ls, lc)), grads = jax.value_and_grad(compute_losses, has_aux=True)(state.params)
    grad_norm = _tree_l2_norm(grads)

    # jax.debug.print(
    #     "step={} loss={} ld={} lp={} ls={} lc={} grad_norm={}",
    #     step, loss, ld, lp, ls, lc, grad_norm
    # )

    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, ld, lp, ls, lc

def make_train_step(use_curv=False, use_lapH=False, use_forc=False):

    mode = []
    if use_curv: mode.append("curv")
    if use_lapH: mode.append("lapH")
    if use_forc: mode.append("forc")
    mode_str = "+".join(mode) if mode else "base"

    print(f"[make_train_step] mode = {mode_str}")
    print(f"  use_curv = {use_curv}")
    print(f"  use_lapH = {use_lapH}")
    print(f"  use_forc = {use_forc}")

    def train_step(state, step, data_edge, data_sign, data_phys, data_curv):
        def compute_losses(params):
            phi_fn = lambda x: state.apply_fn(params, x.reshape(-1, 3))

            l_data = loss_data(phi_fn, data_edge)
            l_phys = loss_phys(phi_fn, data_phys)
            l_sign = loss_sign(phi_fn, data_sign)

            l_curv = loss_curv(phi_fn, data_curv) if use_curv else jnp.array(0.0, dtype=l_data.dtype)
            l_lapH = loss_lapH(phi_fn, data_curv) if use_lapH else jnp.array(0.0, dtype=l_data.dtype)
            l_forc = loss_forc(phi_fn, data_curv) if use_forc else jnp.array(0.0, dtype=l_data.dtype)

            total = (
                state.lambda_1 * l_data
                + state.lambda_2 * l_phys
                + state.lambda_3 * l_sign
                + state.lambda_4 * l_curv
                + state.lambda_5 * l_lapH
                + state.lambda_6 * l_forc
            )

            return total, (l_data, l_phys, l_sign, l_curv, l_lapH, l_forc)

        (loss, aux), grads = jax.value_and_grad(compute_losses, has_aux=True)(state.params)
        (l_data, l_phys, l_sign, l_curv, l_lapH, l_forc) = aux

        # grad_norm = _tree_l2_norm(grads)
        # jax.debug.print(
        #     "step={} loss={} ld={} lp={} ls={} lc={} ll={} lf={} grad_norm={}",
        #     step, loss, l_data, l_phys, l_sign, l_curv, l_lapH, l_forc, grad_norm
        # )

        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, l_data, l_phys, l_sign, l_curv, l_lapH, l_forc

    return jax.jit(train_step, donate_argnums=(0,))


def assemble_loss_history(checkpoint_data):
    """
    Assemble loss history from all available checkpoints.

    Args:
        checkpoint_data (dict): Dictionary containing loss values from multiple checkpoints.

    Returns:
        dict: Aggregated loss history with 'step', 'total_loss', 'data_loss', and 'physics_loss'.
    """
    assembled_loss = {
        "step": [],
        "total_loss": [],
        "data_loss": [],
        "phys_loss": [],
        "sign_loss": [],
        "curv_loss": [],
        "lapH_loss": [],
        "forc_loss": [],
    }

    for step, checkpoint in checkpoint_data.items():
        if 'loss' in checkpoint:
            loss_data = checkpoint['loss']
            assembled_loss["step"].extend(loss_data["step"].tolist())
            assembled_loss["total_loss"].extend(loss_data["total_loss"].tolist())
            assembled_loss["data_loss"].extend(loss_data["data_loss"].tolist())
            assembled_loss["phys_loss"].extend(loss_data["phys_loss"].tolist())
            assembled_loss["sign_loss"].extend(loss_data["sign_loss"].tolist())
            assembled_loss["curv_loss"].extend(loss_data["curv_loss"].tolist())
            assembled_loss["lapH_loss"].extend(loss_data["lapH_loss"].tolist())
            assembled_loss["forc_loss"].extend(loss_data["forc_loss"].tolist())

    # Convert lists to NumPy arrays for easier handling
    for key in assembled_loss:
        assembled_loss[key] = np.array(assembled_loss[key])

    return assembled_loss


def plot_loss_history_ax(ax, assembled_loss):
    """
    Plot the loss function over the entire training process.

    Args:
        assembled_loss (dict): Aggregated loss history containing 'step', 'total_loss', 'data_loss', 'physics_loss'.
    """
    
    # Plot the losses
    ax.plot(assembled_loss["step"], assembled_loss["data_loss"], label='Data Loss', marker='o', linestyle='-')
    ax.plot(assembled_loss["step"], assembled_loss["phys_loss"], label='Physics Loss', marker='s', linestyle='-')
    ax.plot(assembled_loss["step"], assembled_loss["sign_loss"], label='Sign Loss', marker='^', linestyle='-')
    ax.plot(assembled_loss["step"], assembled_loss["curv_loss"], label='Curvature Loss', marker='^', linestyle='-')
    ax.plot(assembled_loss["step"], assembled_loss["lapH_loss"], label='Laplacian Loss', marker='^', linestyle='-')
    ax.plot(assembled_loss["step"], assembled_loss["forc_loss"], label='Force Loss', marker='^', linestyle='-')
    ax.plot(assembled_loss["step"], assembled_loss["total_loss"], label='Total Loss', marker='^', linestyle='-')

    # Use log scale for better visualization (especially if physics loss is large)
    ax.set_yscale('log')

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss Function Over Training')

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
