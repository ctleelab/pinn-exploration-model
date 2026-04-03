import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import grad, jacfwd, jit, vmap
import optax
from flax.training import train_state

# GRID_SIZE = 64
GRID_SIZE = 128
# GRID_SIZE = 200

class PINN(nn.Module):
    # hidden_dim: int = 64  # Hidden layer size
    # hidden_dim: int = 16  # Hidden layer size
    # hidden_dim: int = 256  # Hidden layer size
    # hidden_dim: int = 512  # Hidden layer size
    hidden_dim: int = 128  # Hidden layer size
    # num_frequencies: int = 10 # Number of frequencies for positional encoding

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)  # Ensure input is at least 2D
        if x.shape[-1] != 3:
            raise ValueError(f"Expected input shape (*,3), but got {x.shape}")

        # Apply positional encoding
        # x_encoded = positional_encoding(x, num_frequencies=self.num_frequencies)
        # x_encoded = x
        # x_skip = x_encoded

        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)

        # x = nn.Dense(self.hidden_dim)(x)
        # x = nn.tanh(x)

        # x = jnp.concatenate([x, x_skip], axis=-1)

        x = nn.Dense(1)(x)  # Output single scalar φ(x,y,z)
        x = nn.tanh(x)
        
        return x.squeeze()


def positional_encoding(x, num_frequencies=10, include_input=True):
    """
    Apply sinusoidal positional encoding to input x.

    Args:
        x: input of shape (..., 3)
        num_frequencies: number of frequency bands
        include_input: whether to include the raw input x in the output

    Returns:
        Encoded input of shape (..., 3 * (2 * num_frequencies + include_input))
    """
    x = jnp.atleast_2d(x)
    freq_bands = 2.0 ** jnp.arange(num_frequencies) * jnp.pi  # shape: (num_frequencies,)
    encodings = [x] if include_input else []

    for freq in freq_bands:
        encodings.append(jnp.sin(freq * x))
        encodings.append(jnp.cos(freq * x))

    return jnp.concatenate(encodings, axis=-1)



# Compute first derivatives (∇φ) with vectorized differentiation
def grad_phi(phi_fn, x):
    x = x.reshape(-1, 3)  # Ensure correct shape
    return vmap(lambda x_i: grad(lambda x: phi_fn(jnp.atleast_2d(x)).squeeze())(x_i))(x)

# Compute Δφ (Laplacian of φ)
def laplacian_phi(phi_fn, x):
    x = x.reshape(-1, 3)
    return vmap(lambda x_i: jnp.trace(jacfwd(grad(lambda x: phi_fn(jnp.atleast_2d(x)).squeeze()))(x_i)))(x)

# Compute second derivatives (Hessian ∇²φ) with vectorized differentiation
def hessian_phi(phi_fn, x):
    x = x.reshape(-1, 3)  # Ensure correct shape
    return vmap(lambda x_i: jacfwd(jacfwd(lambda x: phi_fn(jnp.atleast_2d(x)).squeeze()))(x_i))(x)

# # Compute mean curvature H = ∇ ⋅ (∇φ / |∇φ|)
# def mean_curvature(phi_fn, x):
#     gphi = grad_phi(phi_fn, x)  # Compute ∇φ (gradient of φ)
#     norm_gphi = jnp.linalg.norm(gphi, axis=-1, keepdims=True) + 1e-8  # Avoid division by zero
#     n = gphi / norm_gphi  # Compute unit normal vector

#     # def divergence(x_i):
#     #     gphi_x = grad_phi(phi_fn, x_i.reshape(1, 3))  # Compute gradient at x_i
#     #     norm_gphi_x = jnp.linalg.norm(gphi_x, axis=-1, keepdims=True) + 1e-8
#     #     n_x = gphi_x / norm_gphi_x  # Normal vector at x_i
#     #     return jnp.sum(jacfwd(lambda x: n_x)(x_i))  # Sum over spatial dimension

#     def divergence(x_i):
#         def normal_at_x(x):  
#             gphi_x = grad_phi(phi_fn, x)  # Gradient at x
#             norm_gphi_x = jnp.linalg.norm(gphi_x, axis=-1, keepdims=True) + 1e-8
#             return gphi_x / norm_gphi_x  # Normalized gradient

#         jac_n = jacfwd(normal_at_x)(x_i.reshape(1, 3))  # Compute Jacobian of n(x)
#         return jnp.trace(jac_n.squeeze())  # Compute divergence as trace of the Jacobian

#     div_n = vmap(divergence)(x.reshape(-1, 3))  # Apply to all points in batch

#     return div_n.squeeze()  # Ensure scalar output per point


def mean_curvature(phi_fn, x):
    gphi = grad_phi(phi_fn, x)            # (N,3)
    Hphi = hessian_phi(phi_fn, x)         # (N,3,3)

    gnorm = jnp.linalg.norm(gphi, axis=1, keepdims=True) + 1e-8

    # lap_phi = jnp.trace(Hphi, axis1=1, axis2=2, keepdims=True)
    lap_phi = jnp.trace(Hphi, axis1=1, axis2=2)[:, None]

    gHg = jnp.sum(
        gphi[:, None, :] * Hphi * gphi[:, :, None],
        axis=(1, 2),
        keepdims=True
    )

    H = 0.5 * (lap_phi / gnorm - gHg / (gnorm**3))
    return H.squeeze()


import jax
import jax.numpy as jnp
from jax import vmap


def calc_grad_s_H_point(
    phi_fn,
    x,
    normal_eps=1e-8,
    curvature_eps=1e-8,
):
    def phi_scalar(x_single):
        return phi_fn(x_single[None, :]).reshape(())

    grad_phi = jax.grad(phi_scalar)
    hess_phi_fn = jax.jacfwd(grad_phi)

    def mean_curvature(x_single):
        g = grad_phi(x_single)                  # (3,)
        Hphi = hess_phi_fn(x_single)            # (3,3)

        g2 = jnp.dot(g, g)
        Gsafe = jnp.sqrt(g2 + curvature_eps**2)

        lap_phi = jnp.trace(Hphi)
        gHg = jnp.dot(g, Hphi @ g)

        kappa = lap_phi / Gsafe - gHg / (Gsafe**3)
        return 0.5 * kappa

    g = grad_phi(x)
    g2 = jnp.dot(g, g)
    gnorm = jnp.sqrt(g2 + normal_eps**2)
    n = g / gnorm

    grad_H = jax.grad(mean_curvature)(x)

    P = jnp.eye(3, dtype=x.dtype) - jnp.outer(n, n)
    grad_s_H = P @ grad_H
    return grad_s_H


def calc_grad_s_H(
    phi_fn,
    points,
    normal_eps=1e-8,
    curvature_eps=1e-8,
):
    fn = lambda x: calc_grad_s_H_point(
        phi_fn,
        x,
        normal_eps=normal_eps,
        curvature_eps=curvature_eps,
    )
    return vmap(fn)(points)


import jax
import jax.numpy as jnp


def calc_delta_s_H(phi_fn, x, grad_threshold=1e-4):
    """
    Compute surface Laplacian of mean curvature, Δ_s H, at points x.

    Parameters
    ----------
    phi_fn : callable
        Takes (N,3) -> (N,) or (N,1)
    x : array, shape (N,3)
    grad_threshold : float
        Threshold to avoid division/projection issues when ||grad phi|| is too small.

    Returns
    -------
    delta_s_H : array, shape (N,)
    """
    def phi_scalar(x_single):
        return phi_fn(x_single[None, :]).reshape(())

    def grad_s_H_single(x_single):
        return calc_grad_s_H(phi_fn, x_single[None, :]).reshape(3,)

    # grad phi and unit normal
    grad_phi = jax.vmap(jax.grad(phi_scalar))(x)              # (N,3)
    gnorm = jnp.linalg.norm(grad_phi, axis=1, keepdims=True) # (N,1)

    n = grad_phi / jnp.clip(gnorm, a_min=grad_threshold)     # (N,3)

    # Jacobian of grad_s_H: J_ij = ∂_j (grad_s_H)_i
    jac_grad_s_H = jax.vmap(jax.jacfwd(grad_s_H_single))(x)  # (N,3,3)

    # Surface projection P = I - n ⊗ n
    I = jnp.eye(3, dtype=x.dtype)
    P = I[None, :, :] - n[:, :, None] * n[:, None, :]        # (N,3,3)

    # Δ_s H = tr(P @ J)
    delta_s_H = jnp.einsum("nij,nji->n", P, jac_grad_s_H)

    return delta_s_H


# Compute Laplacian of mean curvature ΔH = ∇²H
def laplacian_mean_curvature(phi_fn, x):
    return vmap(lambda x_i: jnp.trace(jacfwd(jacfwd(lambda x: mean_curvature(phi_fn, x)))(x_i)))(x.reshape(-1, 3)).squeeze()

# def gaussian_curvature(phi_fn, x):
#     """
#     Computes the Gaussian curvature K from the level-set function phi.
#     Uses the formula:
#         K = (∇φ ⋅ H(φ) ⋅ ∇φ - det(H(φ))) / ||∇φ||^4
#     """
#     gphi = grad_phi(phi_fn, x)  # Compute ∇φ
#     H_phi = hessian_phi(phi_fn, x)  # Compute Hessian H(φ)
    
#     norm_gphi_sq = jnp.sum(gphi**2, axis=-1, keepdims=True) + 1e-8  # Avoid division by zero

#     # Compute (∇φ ⋅ H(φ) ⋅ ∇φ)
#     numerator1 = jnp.sum(gphi[:, None, :] * H_phi * gphi[:, :, None], axis=(1, 2), keepdims=True)

#     # Compute determinant of Hessian det(H(φ))
#     determinant = jnp.linalg.det(H_phi)

#     # Compute Gaussian curvature K
#     K = (numerator1 - determinant[:, None]) / (norm_gphi_sq**2)

#     return K.squeeze()


def gaussian_curvature(phi_fn, x):
    gphi = grad_phi(phi_fn, x)            # (N,3)
    Hphi = hessian_phi(phi_fn, x)         # (N,3,3)

    gnorm2 = jnp.sum(gphi**2, axis=1, keepdims=True) + 1e-8

    gHg = jnp.sum(
        gphi[:, None, :] * Hphi * gphi[:, :, None],
        axis=(1, 2),
        keepdims=True
    )

    detH = jnp.linalg.det(Hphi).reshape(-1, 1)

    K = (gHg - detH) / (gnorm2**2)
    return K.squeeze()

# Data fidelity loss (ensures φ(x,y,z) aligns with cryo-ET data)
# def loss_data(phi_fn, x, cryoET_data):
    # return jnp.mean((phi_fn(x.reshape(-1, 3)) - cryoET_data) ** 2)


# def loss_data(phi_fn, cryoET_data, threshold=0.8):
#     """
#     Compute loss by enforcing φ=0 where I=1 (membrane locations).
#     """
#     x_grid, y_grid, z_grid = jnp.meshgrid(
#         jnp.linspace(-1, 1, GRID_SIZE),
#         jnp.linspace(-1, 1, GRID_SIZE),
#         jnp.linspace(-1, 1, GRID_SIZE),
#         indexing="ij"
#     )

#     grid_points = jnp.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=-1)  # Shape: (N, 3)
#     binary_mask = jnp.where(cryoET_data > threshold, 1, 0) # original
#     # binary_mask = cryoET_data

#     phi = phi_fn(grid_points).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)

#     weight_in = 0.5
#     inside_loss = jnp.sum(binary_mask * (phi - 0.0) ** 2) / jnp.sum(binary_mask)
#     outside_loss = jnp.sum((1 - binary_mask) * (phi ** 2 - 1.0) ** 2) / jnp.sum(1 - binary_mask)
#     membrane_loss = weight_in * inside_loss + (1 - weight_in) * outside_loss

#     return membrane_loss


def loss_data_original(phi_fn, cryoET_data, threshold=0.8, epsilon=0.05):
    """
    """
    x_grid, y_grid, z_grid = jnp.meshgrid(
        jnp.linspace(-1, 1, GRID_SIZE),
        jnp.linspace(-1, 1, GRID_SIZE),
        jnp.linspace(-1, 1, GRID_SIZE),
        indexing="ij"
    )
    grid_points = jnp.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=-1)
    binary_mask = (cryoET_data > threshold).astype(jnp.float32).ravel()
    phi_vals = phi_fn(grid_points).squeeze()
    gradient = grad_phi(phi_fn, grid_points)
    sq_grad = jnp.sum(gradient**2, axis=-1)
    # energy_density = epsilon**2 * sq_grad + 0.5 * (phi_vals**2 - 1.0)**2
    # membrane_loss = jnp.mean((energy_density - binary_mask)**2)

    # pos = jnp.mean(binary_mask) + 1e-8
    # neg = 1.0 - pos
    # w = binary_mask / pos + (1.0 - binary_mask) / neg
    # membrane_loss = jnp.mean(w * (energy_density - binary_mask)**2)

    # energy_density = sq_grad + (0.5 / epsilon**2) * (phi_vals**2 - 1.0)**2
    energy_density = epsilon**2 * sq_grad + 0.5 * (phi_vals**2 - 1.0)**2
    # target = (1/(epsilon*epsilon)) * binary_mask
    # target = 400 * binary_mask
    target = binary_mask
    membrane_loss = jnp.mean((energy_density - target)**2)

    # jax.debug.print("epsilon = {}", epsilon)
    # jax.debug.print("binary_mask = {}", binary_mask[1:10])
    # jax.debug.print("400 * binary_mask = {}", 400 * binary_mask[1:10])
    # jax.debug.print("(1/epsilon**2) * binary_mask = {}", (1/epsilon**2) * binary_mask[1:10])

    # pos = jnp.mean(binary_mask) + 1e-8
    # neg = 1.0 - pos
    # w = binary_mask / pos + (1.0 - binary_mask) / neg
    # membrane_loss = jnp.mean(w * (energy_density - target)**2)

    return membrane_loss


def loss_data(phi_fn, data_edge, epsilon=0.05):
    x = data_edge["points"]
    y = data_edge["label"]   # (1.0 on membrane and 0.0 on bulk)
    phi_vals = vmap(lambda x_i: phi_fn(jnp.atleast_2d(x_i)).squeeze())(x)
    gradient = grad_phi(phi_fn, x)
    sq_grad = jnp.sum(gradient**2, axis=-1)
    energy_density = epsilon**2 * sq_grad + 0.5 * (phi_vals**2 - 1.0)**2
    return jnp.mean((energy_density - y)**2)


def loss_phys(phi_fn, data_phys, epsilon = 0.05):
    x = data_phys["points"]
    phi_vals = vmap(lambda x_i: phi_fn(jnp.atleast_2d(x_i)).squeeze())(x)
    lap_phi = laplacian_phi(phi_fn, x)
    residual = lap_phi - (1 / epsilon**2) * (phi_vals**2 - 1) * phi_vals
    return jnp.mean(residual**2)


def loss_sign(phi_fn, data_sign):
    x = data_sign["points"]
    y = data_sign["label"]  # (+1.0 or -1.0)
    phi_vals = vmap(lambda x_i: phi_fn(jnp.atleast_2d(x_i)).squeeze())(x)
    residual = phi_vals - y
    return jnp.mean(residual**2)


def loss_curv(phi_fn, data_curv):
    x = data_curv["points"]              # (N,3)
    grad_s_H = calc_grad_s_H(phi_fn, x)  # (N,3)
    residual = jnp.sum(grad_s_H**2, axis=1)
    return jnp.mean(residual)

# def loss_curv(phi_fn, data_curv, grad_threshold=1e-4):
#     x = data_curv["points"]                      # (N,3)
#     grad_s_H = calc_grad_s_H(phi_fn, x)         # (N,3)

#     def phi_scalar(x_single):
#         return phi_fn(x_single[None, :]).reshape(())

#     grad_phi = jax.vmap(jax.grad(phi_scalar))(x)
#     gnorm = jnp.linalg.norm(grad_phi, axis=1)

#     finite_mask = jnp.all(jnp.isfinite(grad_s_H), axis=1)
#     grad_mask = gnorm > grad_threshold
#     valid_mask = finite_mask & grad_mask

#     grad_s_H_safe = jnp.where(jnp.isfinite(grad_s_H), grad_s_H, 0.0)
#     residual = jnp.sum(grad_s_H_safe**2, axis=1)

#     n_valid = jnp.sum(valid_mask)
#     loss = jnp.where(n_valid > 0, jnp.sum(residual * valid_mask) / n_valid, 0.0)

#     jax.debug.print("valid curvature points = {} / {}", n_valid, x.shape[0])
#     return loss


def loss_lapH(phi_fn, data_curv, grad_threshold=1e-4):
    x = data_curv["points"]                                # (N,3)
    delta_s_H = calc_delta_s_H(phi_fn, x, grad_threshold)  # (N,)
    residual = delta_s_H**2
    return jnp.mean(residual)


import jax
import jax.numpy as jnp

def inspect_shape_equation_jit(phi_fn, x, alpha=10.0, beta=1.0):
    H = mean_curvature(phi_fn, x)
    K = gaussian_curvature(phi_fn, x)
    lapH = calc_delta_s_H(phi_fn, x)

    term_lapH = lapH
    term_nl = 2.0 * H * (H**2 - K)
    term_tension = 2.0 * alpha * H
    term_pressure = beta
    residual = term_lapH + term_nl - term_tension - term_pressure

    jax.debug.print(
        "H     mean={:.3e}, std={:.3e}, min={:.3e}, max={:.3e}",
        jnp.mean(H), jnp.std(H), jnp.min(H), jnp.max(H)
    )
    jax.debug.print(
        "K     mean={:.3e}, std={:.3e}, min={:.3e}, max={:.3e}",
        jnp.mean(K), jnp.std(K), jnp.min(K), jnp.max(K)
    )
    jax.debug.print(
        "lapH  mean={:.3e}, std={:.3e}, min={:.3e}, max={:.3e}",
        jnp.mean(lapH), jnp.std(lapH), jnp.min(lapH), jnp.max(lapH)
    )
    jax.debug.print(
        "force mean={:.3e}, std={:.3e}, min={:.3e}, max={:.3e}",
        jnp.mean(term_lapH + term_nl), jnp.std(term_lapH + term_nl), jnp.min(term_lapH + term_nl), jnp.max(term_lapH + term_nl)
    )
    jax.debug.print(
        "residual mean={:.3e}, std={:.3e}, min={:.3e}, max={:.3e}",
        jnp.mean(residual), jnp.std(residual), jnp.min(residual), jnp.max(residual)
    )
    jax.debug.print(
        "tension     mean={:.3e}, std={:.3e}, min={:.3e}, max={:.3e}",
        jnp.mean(term_tension), jnp.std(term_tension), jnp.min(term_tension), jnp.max(term_tension)
    )
    jax.debug.print(
        "pressure    mean={:.3e}, std={:.3e}, min={:.3e}, max={:.3e}",
        jnp.mean(term_pressure), jnp.std(term_pressure), jnp.min(term_pressure), jnp.max(term_pressure)
    )    
    jax.debug.print(
        "finite H={}; finite K={}; finite lapH={}; finite res={}",
        jnp.all(jnp.isfinite(H)),
        jnp.all(jnp.isfinite(K)),
        jnp.all(jnp.isfinite(lapH)),
        jnp.all(jnp.isfinite(residual)),
    )

    return residual



def calc_bending_force(phi_fn, x):
    H = mean_curvature(phi_fn, x)     # (N,)
    K = gaussian_curvature(phi_fn, x) # (N,)
    lapH = calc_delta_s_H(phi_fn, x)       # (N,)

    return -(lapH + 2.0 * H * (H**2 - K))

def calc_shape_derivative(phi_fn, x):
    H = mean_curvature(phi_fn, x)      # (N,)
    K = gaussian_curvature(phi_fn, x)  # (N,)
    lapH = calc_delta_s_H(phi_fn, x)   # (N,)
    alpha = 0
    beta  = 0

    return lapH + 2.0 * H * (H**2 - K) - 2.0 * alpha * H - beta


def calc_grad_s_force(phi_fn, x):
    def phi_scalar(x_single):
        return phi_fn(x_single[None, :]).reshape(())

    def force_scalar(x_single):
        return calc_bending_force(phi_fn, x_single[None, :]).reshape(())

    grad_phi = jax.vmap(jax.grad(phi_scalar))(x)                    # (N,3)
    n = grad_phi / jnp.linalg.norm(grad_phi, axis=1, keepdims=True)

    grad_f = jax.vmap(jax.grad(force_scalar))(x)                    # (N,3)
    grad_s_f = grad_f - jnp.sum(grad_f * n, axis=1, keepdims=True) * n
    return grad_s_f


def loss_forc(phi_fn, data_curv):
    x = data_curv["points"]
    # grad_s_f = calc_grad_s_force(phi_fn, x)                         # (N,3)
    # residual = jnp.sum(grad_s_f**2, axis=1)
    lag_shape = calc_shape_derivative(phi_fn, x)                    # (N,)
    residual = lag_shape**2

    # inspect_shape_equation_jit(phi_fn, x)

    return jnp.mean(residual)


# Combined loss function
def total_loss(phi_fn, x, cryoET_data, lambda_1, lambda_2):
    return lambda_1 * loss_data(phi_fn, cryoET_data) + lambda_2 * loss_physics(phi_fn, x)


def phase_volume(phi_fn, x, V_box=8): # volume
    phi_vals = vmap(lambda x_i: phi_fn(jnp.atleast_2d(x_i)).squeeze())(x)
    value = 1 + phi_vals
    volume = jnp.sum(value*0.5)
    volume *= V_box / x.shape[0]

    return volume

def phase_surface(phi_fn, x, epsilon, V_box=8): # surface area
    phi_vals = vmap(lambda x_i: phi_fn(jnp.atleast_2d(x_i)).squeeze())(x)
    grad_val = grad_phi(phi_fn, x)
    sq_grad = jnp.sum(grad_val ** 2, axis=1)

    value = epsilon * sq_grad + (1 / 2.0 / epsilon) * (1 - phi_vals**2)**2
    value *= 3/4/jnp.sqrt(2)
    area = jnp.sum(value)
    area *= V_box / x.shape[0]
    
    return area

def phase_bend(phi_fn, x, epsilon, kappa, V_box=8): # bending energy
    phi_vals = vmap(lambda x_i: phi_fn(jnp.atleast_2d(x_i)).squeeze())(x)
    lap_phi = laplacian_phi(phi_fn, x)

    value = epsilon*lap_phi - (1 / epsilon) * (phi_vals**2 - 1) * phi_vals
    bend = jnp.sum(value**2)
    bend *= 3*kappa/4/jnp.sqrt(2)/epsilon
    bend *= V_box / x.shape[0]

    return bend









import jax
import jax.numpy as jnp
from jax import vmap


def debug_grad_s_H_point(phi_fn, x, normal_eps=1e-8, curvature_eps=1e-8):
    def phi_scalar(x_single):
        return phi_fn(jnp.atleast_2d(x_single)).squeeze()

    g = jax.grad(phi_scalar)(x)                    # (3,)
    g2 = jnp.dot(g, g)
    gnorm = jnp.sqrt(g2 + normal_eps**2)
    n = g / gnorm

    hess_phi = jax.hessian(phi_scalar)(x)          # (3,3)
    lap_phi = jnp.trace(hess_phi)
    gHg = jnp.dot(g, hess_phi @ g)

    Gsafe = jnp.sqrt(g2 + curvature_eps**2)
    kappa = lap_phi / Gsafe - gHg / (Gsafe**3)
    H = 0.5 * kappa

    def H_scalar(x_single):
        g_ = jax.grad(phi_scalar)(x_single)
        hess_ = jax.hessian(phi_scalar)(x_single)
        g2_ = jnp.dot(g_, g_)
        Gsafe_ = jnp.sqrt(g2_ + curvature_eps**2)
        lap_ = jnp.trace(hess_)
        gHg_ = jnp.dot(g_, hess_ @ g_)
        return 0.5 * (lap_ / Gsafe_ - gHg_ / (Gsafe_**3))

    grad_H = jax.grad(H_scalar)(x)
    P = jnp.eye(3) - jnp.outer(n, n)
    grad_s_H = P @ grad_H

    return {
        "phi": phi_scalar(x),
        "g": g,
        "g2": g2,
        "gnorm": gnorm,
        "n": n,
        "hess_phi": hess_phi,
        "lap_phi": lap_phi,
        "gHg": gHg,
        "H": H,
        "grad_H": grad_H,
        "grad_s_H": grad_s_H,
    }


import numpy as np
import jax
import jax.numpy as jnp


def find_bad_points_from_calc_grad_s_H(phi_fn, data_curv, n_check=None):
    points = jnp.asarray(data_curv["points"])
    if n_check is None:
        n_check = points.shape[0]
    points = points[:n_check]

    grad_s_H = calc_grad_s_H(phi_fn, points)
    grad_s_H_np = np.asarray(jax.device_get(grad_s_H))

    finite_mask = np.all(np.isfinite(grad_s_H_np), axis=1)
    n_bad = np.sum(~finite_mask)

    print(f"Number of bad points: {n_bad} / {points.shape[0]}")

    if n_bad == 0:
        print(f"No bad points found in first {points.shape[0]} points.")
        return []

    first_bad = int(np.where(~finite_mask)[0][0])
    print("First bad index:", first_bad)
    print("First bad point:", np.asarray(points[first_bad]))
    print("First bad grad_s_H:", grad_s_H_np[first_bad])

    return [{
        "index": first_bad,
        "point": np.asarray(points[first_bad]),
        "grad_s_H": grad_s_H_np[first_bad],
    }]


import numpy as np
import jax.numpy as jnp

def find_bad_points(phi_fn, data_curv, n_check=None, normal_eps=1e-6, curvature_eps=1e-6):
    points = data_curv["points"]
    if n_check is None:
        n_check = len(points)

    bad_list = []

    for i in range(min(n_check, len(points))):
        x = points[i]
        try:
            out = debug_grad_s_H_point(
                phi_fn,
                x,
                normal_eps=normal_eps,
                curvature_eps=curvature_eps,
            )

            finite_dict = {
                k: np.all(np.isfinite(np.asarray(v)))
                for k, v in out.items()
            }

            if not all(finite_dict.values()):
                bad_list.append({
                    "index": i,
                    "point": np.asarray(x),
                    "finite_dict": finite_dict,
                    "out": out,
                })
                print(f"Bad point found at index {i}")
                print("point =", x)
                print("finite_dict =", finite_dict)
                break

        except Exception as e:
            bad_list.append({
                "index": i,
                "point": np.asarray(x),
                "exception": repr(e),
            })
            print(f"Exception at index {i}")
            print("point =", x)
            print("exception =", repr(e))
            break

    if len(bad_list) == 0:
        print(f"No bad points found in first {min(n_check, len(points))} points.")

    return bad_list







import numpy as np
import jax
import jax.numpy as jnp


def make_phi_scalar(phi_fn):
    def phi_scalar(x):
        return phi_fn(x[None, :]).squeeze()
    return phi_scalar


def make_H_scalar(phi_fn, curvature_eps=1e-8):
    phi_scalar = make_phi_scalar(phi_fn)
    grad_phi = jax.grad(phi_scalar)
    hess_phi = jax.jacfwd(grad_phi)

    def H_scalar(x):
        g = grad_phi(x)                  # (3,)
        Hphi = hess_phi(x)               # (3,3)
        g2 = jnp.dot(g, g)
        Gsafe = jnp.sqrt(g2 + curvature_eps**2)
        lap_phi = jnp.trace(Hphi)
        gHg = jnp.dot(g, Hphi @ g)
        kappa = lap_phi / Gsafe - gHg / (Gsafe**3)
        return 0.5 * kappa

    return H_scalar


def make_screen_point_fn(phi_fn, normal_eps=1e-8, curvature_eps=1e-8):
    """
    Cheap-ish screening function:
    checks finiteness of the main geometric quantities, including grad_s_H,
    but returns only a boolean mask and a few compact flags.
    """
    phi_scalar = make_phi_scalar(phi_fn)
    grad_phi = jax.grad(phi_scalar)
    hess_phi = jax.jacfwd(grad_phi)
    H_scalar = make_H_scalar(phi_fn, curvature_eps=curvature_eps)
    grad_H_fn = jax.grad(H_scalar)

    def screen_point(x):
        phi = phi_scalar(x)

        g = grad_phi(x)
        g2 = jnp.dot(g, g)
        gnorm = jnp.sqrt(g2 + normal_eps**2)
        n = g / gnorm

        Hphi = hess_phi(x)
        lap_phi = jnp.trace(Hphi)
        gHg = jnp.dot(g, Hphi @ g)

        H = H_scalar(x)
        grad_H = grad_H_fn(x)
        P = jnp.eye(3, dtype=x.dtype) - jnp.outer(n, n)
        grad_s_H = P @ grad_H

        finite_phi = jnp.all(jnp.isfinite(phi))
        finite_g = jnp.all(jnp.isfinite(g))
        finite_n = jnp.all(jnp.isfinite(n))
        finite_hess = jnp.all(jnp.isfinite(Hphi))
        finite_lap = jnp.all(jnp.isfinite(lap_phi))
        finite_gHg = jnp.all(jnp.isfinite(gHg))
        finite_H = jnp.all(jnp.isfinite(H))
        finite_grad_H = jnp.all(jnp.isfinite(grad_H))
        finite_grad_s_H = jnp.all(jnp.isfinite(grad_s_H))

        all_finite = (
            finite_phi
            & finite_g
            & finite_n
            & finite_hess
            & finite_lap
            & finite_gHg
            & finite_H
            & finite_grad_H
            & finite_grad_s_H
        )

        flags = jnp.array([
            finite_phi,
            finite_g,
            finite_n,
            finite_hess,
            finite_lap,
            finite_gHg,
            finite_H,
            finite_grad_H,
            finite_grad_s_H,
        ], dtype=jnp.bool_)

        return all_finite, flags

    return jax.jit(jax.vmap(screen_point))


def make_debug_point_fn(phi_fn, normal_eps=1e-8, curvature_eps=1e-8):
    """
    Full debug output for a single point.
    Use only after you already know which point is bad.
    """
    phi_scalar = make_phi_scalar(phi_fn)
    grad_phi = jax.grad(phi_scalar)
    hess_phi = jax.jacfwd(grad_phi)
    H_scalar = make_H_scalar(phi_fn, curvature_eps=curvature_eps)
    grad_H_fn = jax.grad(H_scalar)

    @jax.jit
    def debug_point(x):
        phi = phi_scalar(x)

        g = grad_phi(x)
        g2 = jnp.dot(g, g)
        gnorm = jnp.sqrt(g2 + normal_eps**2)
        n = g / gnorm

        Hphi = hess_phi(x)
        lap_phi = jnp.trace(Hphi)
        gHg = jnp.dot(g, Hphi @ g)

        H = H_scalar(x)
        grad_H = grad_H_fn(x)
        P = jnp.eye(3, dtype=x.dtype) - jnp.outer(n, n)
        grad_s_H = P @ grad_H

        return {
            "phi": phi,
            "g": g,
            "g2": g2,
            "gnorm": gnorm,
            "n": n,
            "hess_phi": Hphi,
            "lap_phi": lap_phi,
            "gHg": gHg,
            "H": H,
            "grad_H": grad_H,
            "grad_s_H": grad_s_H,
        }

    return debug_point


def find_bad_points_fast(phi_fn, data_curv, n_check=None, normal_eps=1e-6, curvature_eps=1e-6):
    points = jnp.asarray(data_curv["points"])
    if n_check is None:
        n_check = points.shape[0]
    points = points[:n_check]

    screen_fn = make_screen_point_fn(
        phi_fn,
        normal_eps=normal_eps,
        curvature_eps=curvature_eps,
    )
    debug_fn = make_debug_point_fn(
        phi_fn,
        normal_eps=normal_eps,
        curvature_eps=curvature_eps,
    )

    try:
        all_finite_mask, flags = screen_fn(points)
        all_finite_mask = np.asarray(all_finite_mask)
        flags = np.asarray(flags)

        bad_idx = np.where(~all_finite_mask)[0]

        if len(bad_idx) == 0:
            print(f"No bad points found in first {points.shape[0]} points.")
            return []

        i = int(bad_idx[0])
        x = points[i]

        out = debug_fn(x)
        out_np = jax.tree_util.tree_map(lambda v: np.asarray(v), out)

        names = [
            "phi",
            "g",
            "n",
            "hess_phi",
            "lap_phi",
            "gHg",
            "H",
            "grad_H",
            "grad_s_H",
        ]
        finite_dict = {name: bool(flags[i, k]) for k, name in enumerate(names)}

        bad_list = [{
            "index": i,
            "point": np.asarray(x),
            "finite_dict": finite_dict,
            "out": out_np,
        }]

        print(f"Bad point found at index {i}")
        print("point =", np.asarray(x))
        print("finite_dict =", finite_dict)

        return bad_list

    except Exception as e:
        # fallback: if batched execution itself fails, locate the first exception point-by-point
        debug_fn = make_debug_point_fn(
            phi_fn,
            normal_eps=normal_eps,
            curvature_eps=curvature_eps,
        )

        for i in range(points.shape[0]):
            x = points[i]
            try:
                out = debug_fn(x)
                out_np = jax.tree_util.tree_map(lambda v: np.asarray(v), out)
                finite_dict = {
                    k: np.all(np.isfinite(v))
                    for k, v in out_np.items()
                }
                if not all(finite_dict.values()):
                    bad_list = [{
                        "index": i,
                        "point": np.asarray(x),
                        "finite_dict": finite_dict,
                        "out": out_np,
                    }]
                    print(f"Bad point found at index {i}")
                    print("point =", np.asarray(x))
                    print("finite_dict =", finite_dict)
                    return bad_list
            except Exception as e_single:
                bad_list = [{
                    "index": i,
                    "point": np.asarray(x),
                    "exception": repr(e_single),
                }]
                print(f"Exception at index {i}")
                print("point =", np.asarray(x))
                print("exception =", repr(e_single))
                return bad_list

        print(f"No bad points found in first {points.shape[0]} points.")
        return []


import numpy as np
import jax
import jax.numpy as jnp

def debug_bad_point(phi_fn, x, normal_eps=1e-6, curvature_eps=1e-6):
    x = jnp.asarray(x)

    def phi_scalar(x_single):
        return phi_fn(x_single[None, :]).squeeze()

    grad_phi = jax.grad(phi_scalar)
    hess_phi_fn = jax.jacfwd(grad_phi)

    phi = phi_scalar(x)
    g = grad_phi(x)
    g2 = jnp.dot(g, g)
    gnorm = jnp.sqrt(g2 + normal_eps**2)
    n = g / gnorm

    Hphi = hess_phi_fn(x)
    lap_phi = jnp.trace(Hphi)
    gHg = jnp.dot(g, Hphi @ g)

    Gsafe = jnp.sqrt(g2 + curvature_eps**2)
    kappa = lap_phi / Gsafe - gHg / (Gsafe**3)
    H = 0.5 * kappa

    def H_scalar(x_single):
        g_ = grad_phi(x_single)
        Hphi_ = hess_phi_fn(x_single)
        g2_ = jnp.dot(g_, g_)
        Gsafe_ = jnp.sqrt(g2_ + curvature_eps**2)
        lap_ = jnp.trace(Hphi_)
        gHg_ = jnp.dot(g_, Hphi_ @ g_)
        return 0.5 * (lap_ / Gsafe_ - gHg_ / (Gsafe_**3))

    grad_H = jax.grad(H_scalar)(x)
    P = jnp.eye(3, dtype=x.dtype) - jnp.outer(n, n)
    grad_s_H = P @ grad_H

    out = {
        "phi": phi,
        "g": g,
        "g2": g2,
        "gnorm": gnorm,
        "n": n,
        "Hphi": Hphi,
        "lap_phi": lap_phi,
        "gHg": gHg,
        "Gsafe": Gsafe,
        "H": H,
        "grad_H": grad_H,
        "P": P,
        "grad_s_H": grad_s_H,
    }

    out_np = {k: np.asarray(jax.device_get(v)) for k, v in out.items()}

    for k, v in out_np.items():
        finite = np.all(np.isfinite(v))
        vmax = np.max(np.abs(v)) if np.size(v) else 0.0
        print(f"{k:10s} finite={finite}  maxabs={vmax}")
        print(v)

    return out_np