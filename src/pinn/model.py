import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import grad, jacfwd, jit, vmap
import optax
from flax.training import train_state

GRID_SIZE = 64

class PINN(nn.Module):
    # hidden_dim: int = 64  # Hidden layer size
    hidden_dim: int = 16  # Hidden layer size
    num_frequencies: int = 10 # Number of frequencies for positional encoding

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)  # Ensure input is at least 2D
        if x.shape[-1] != 3:
            raise ValueError(f"Expected input shape (*,3), but got {x.shape}")

        # Apply positional encoding
        # x_encoded = positional_encoding(x, num_frequencies=self.num_frequencies)
        x_encoded = x

        x = nn.Dense(self.hidden_dim)(x_encoded)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)

        x = nn.tanh(x)
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

# Compute mean curvature H = ∇ ⋅ (∇φ / |∇φ|)
def mean_curvature(phi_fn, x):
    gphi = grad_phi(phi_fn, x)  # Compute ∇φ (gradient of φ)
    norm_gphi = jnp.linalg.norm(gphi, axis=-1, keepdims=True) + 1e-8  # Avoid division by zero
    n = gphi / norm_gphi  # Compute unit normal vector

    # def divergence(x_i):
    #     gphi_x = grad_phi(phi_fn, x_i.reshape(1, 3))  # Compute gradient at x_i
    #     norm_gphi_x = jnp.linalg.norm(gphi_x, axis=-1, keepdims=True) + 1e-8
    #     n_x = gphi_x / norm_gphi_x  # Normal vector at x_i
    #     return jnp.sum(jacfwd(lambda x: n_x)(x_i))  # Sum over spatial dimension

    def divergence(x_i):
        def normal_at_x(x):  
            gphi_x = grad_phi(phi_fn, x)  # Gradient at x
            norm_gphi_x = jnp.linalg.norm(gphi_x, axis=-1, keepdims=True) + 1e-8
            return gphi_x / norm_gphi_x  # Normalized gradient

        jac_n = jacfwd(normal_at_x)(x_i.reshape(1, 3))  # Compute Jacobian of n(x)
        return jnp.trace(jac_n.squeeze())  # Compute divergence as trace of the Jacobian

    div_n = vmap(divergence)(x.reshape(-1, 3))  # Apply to all points in batch

    # jax.debug.print("Grad Phi Min/Max: {}/{}", gphi.min(), gphi.max())
    # jax.debug.print("Normalized Grad Phi Min/Max: {}/{}", n.min(), n.max())
    # jax.debug.print("Div(N) Min/Max: {}/{}", div_n.min(), div_n.max())
    return div_n.squeeze()  # Ensure scalar output per point

# Compute Laplacian of mean curvature ΔH = ∇²H
def laplacian_mean_curvature(phi_fn, x):
    return vmap(lambda x_i: jnp.trace(jacfwd(jacfwd(lambda x: mean_curvature(phi_fn, x)))(x_i)))(x.reshape(-1, 3)).squeeze()

def gaussian_curvature(phi_fn, x):
    """
    Computes the Gaussian curvature K from the level-set function phi.
    Uses the formula:
        K = (∇φ ⋅ H(φ) ⋅ ∇φ - det(H(φ))) / ||∇φ||^4
    """
    gphi = grad_phi(phi_fn, x)  # Compute ∇φ
    H_phi = hessian_phi(phi_fn, x)  # Compute Hessian H(φ)
    
    norm_gphi_sq = jnp.sum(gphi**2, axis=-1, keepdims=True) + 1e-8  # Avoid division by zero

    # Compute (∇φ ⋅ H(φ) ⋅ ∇φ)
    numerator1 = jnp.sum(gphi[:, None, :] * H_phi * gphi[:, :, None], axis=(1, 2), keepdims=True)

    # Compute determinant of Hessian det(H(φ))
    determinant = jnp.linalg.det(H_phi)

    # Compute Gaussian curvature K
    K = (numerator1 - determinant[:, None]) / (norm_gphi_sq**2)

    return K.squeeze()


# Data fidelity loss (ensures φ(x,y,z) aligns with cryo-ET data)
# def loss_data(phi_fn, x, cryoET_data):
    # return jnp.mean((phi_fn(x.reshape(-1, 3)) - cryoET_data) ** 2)

# def loss_data(phi_fn, cryoET_data):
def loss_data(phi_fn, cryoET_data, membrane_indices):
    """
    Compute loss by enforcing φ=0 where I=1 (membrane locations).
    """
    x_grid, y_grid, z_grid = jnp.meshgrid(
        jnp.linspace(-1, 1, GRID_SIZE),
        jnp.linspace(-1, 1, GRID_SIZE),
        jnp.linspace(-1, 1, GRID_SIZE),
        indexing="ij"
    )

    grid_points = jnp.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=-1)  # Shape: (N, 3)

    # # Use jax.vmap() to compute gradient at multiple points
    # grad_phi_fn = jax.vmap(jax.grad(phi_fn, argnums=0))  # Apply grad to each point
    # grad_phi_x = grad_phi_fn(grid_points).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE, 3)

    # Compute the loss by enforcing φ=0 where cryoET_data is 1
    # binary_mask = jnp.where(cryoET_data > 0.5, 1, 0)

    binary_mask = jnp.where(cryoET_data > 0.8, 1, 0) # original
    # binary_mask = jnp.where(cryoET_data > 0.82, 1, 0)
    # binary_mask = cryoET_data

    # membrane_loss = jnp.mean((phi_fn(grid_points).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE) * binary_mask) ** 2)

    phi = phi_fn(grid_points).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)
    # membrane_loss = jnp.mean(
    #     (binary_mask * (phi - 0.0) ** 2) +                # penalize phi ≠ 0 where mask = 1
    #     ((1 - binary_mask) * (phi ** 2 - 1.0) ** 2)       # penalize phi ≠ ±1 where mask = 0
    # )

    # weight_in = 0.95
    weight_in = 0.8
    inside_loss = jnp.sum(binary_mask * (phi - 0.0) ** 2) / jnp.sum(binary_mask)
    outside_loss = jnp.sum((1 - binary_mask) * (phi ** 2 - 1.0) ** 2) / jnp.sum(1 - binary_mask)
    membrane_loss = weight_in * inside_loss + (1 - weight_in) * outside_loss

    # # case 2
    # outside_loss = jnp.sum((1 - binary_mask) * (phi ** 2 - 1.0) ** 2) / jnp.sum(1 - binary_mask)
    # membrane_loss = outside_loss

    # # case 3 (too slow)
    # # new data loss function based on gradient
    # gradient = grad_phi(phi_fn, grid_points)
    # grad_sqr = jnp.sum(gradient ** 2, axis=1).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)
    # membrane_loss = jnp.mean(
    #     ((1 - binary_mask) * grad_sqr)       # penalize phi ≠ 0 where mask = 0
    # # )

    # # case 3.5
    # total_points = grid_points.shape[0]
    # num_samples = 1000
    # rng_key = jax.random.PRNGKey(1234)
    # sample_indices = jax.random.choice(rng_key, total_points, shape=(num_samples,), replace=False)
    # sampled_points = grid_points[sample_indices]  # (num_samples, 3)
    # gradient = grad_phi(phi_fn, sampled_points)   # (num_samples, 3)
    # grad_sqr = jnp.sum(gradient ** 2, axis=1)     # (num_samples,)
    # sampled_mask = binary_mask.ravel()[sample_indices]
    # membrane_loss = jnp.mean(
    #     ((1 - sampled_mask) * grad_sqr)       # penalize phi ≠ 0 where mask = 0
    # )

    # # case 4
    # # membrane_indices = jnp.where(binary_mask.ravel() == 1)[0]
    # points_membrane = grid_points[membrane_indices]  # shape: (N_membrane, 3)
    # gradient = grad_phi(phi_fn, points_membrane)     # shape (N_membrane, 3)
    # grad_mag = jnp.linalg.norm(gradient, axis=1)     # shape (N_membrane,)
    # epsilon = 0.05
    # target = 1.0 / epsilon
    # membrane_loss = jnp.mean((grad_mag - target)**2)

    return membrane_loss


# def loss_physics(phi_fn, x, eps=1e-2):
#     phi = phi_fn(x)  # Evaluate φ(x)
    
#     # Compute physics terms
#     H = mean_curvature(phi_fn, x)
#     Delta_H = laplacian_mean_curvature(phi_fn, x)
#     K = gaussian_curvature(phi_fn, x)

#     # Residual from the Helfrich shape equation
#     residual = Delta_H + 2 * H * (H**2 - K)

#     # Mask: keep only points near the zero level set
#     mask = jnp.abs(phi) < eps

#     # Optional: avoid division by zero if no points in mask
#     masked_residual = jnp.where(mask, residual, 0.0)
#     normalization = jnp.maximum(mask.sum(), 1)

#     return jnp.sum(masked_residual**2) / normalization

# # Physics loss (enforces the Helfrich equation)
# def loss_physics(phi_fn, x):
#     H = mean_curvature(phi_fn, x)
#     Delta_H = laplacian_mean_curvature(phi_fn, x)
#     K = gaussian_curvature(phi_fn, x)

#     physics_residual = H

#     return jnp.mean(physics_residual**2)


# # Physics loss
# def loss_physics(phi_fn, x):
#     H = mean_curvature(phi_fn, x)
#     Delta_H = laplacian_mean_curvature(phi_fn, x)
#     K = gaussian_curvature(phi_fn, x)
#     physics_residual = Delta_H + 2 * H * (H**2 - K)

#     return jnp.mean(physics_residual**2)


# New physics loss based on Allen-Cahn-like equation
def loss_physics(phi_fn, x, epsilon = 0.05):
    phi_vals = vmap(lambda x_i: phi_fn(jnp.atleast_2d(x_i)).squeeze())(x)
    lap_phi = laplacian_phi(phi_fn, x)

    residual = lap_phi - (1 / epsilon**2) * (phi_vals**2 - 1) * phi_vals
    # residual = lap_phi - (1 / epsilon**2) * (phi_vals**2 - 1) * (phi_vals**2 - 1)
    # return (epsilon / 2) * jnp.mean(residual**2)
    return jnp.mean(residual**2)


# Combined loss function
# def total_loss(phi_fn, x, cryoET_data, lambda_1, lambda_2):
def total_loss(phi_fn, x, cryoET_data, lambda_1, lambda_2, membrane_indices):
    # return lambda_1 * loss_data(phi_fn, cryoET_data) + lambda_2 * loss_physics(phi_fn, x)
    return lambda_1 * loss_data(phi_fn, cryoET_data, membrane_indices) + lambda_2 * loss_physics(phi_fn, x)



def phase_volume(phi_fn, x, V_box=8):
    phi_vals = vmap(lambda x_i: phi_fn(jnp.atleast_2d(x_i)).squeeze())(x)
    value = 1 + phi_vals
    volume = jnp.sum(value*0.5)
    volume *= V_box / x.shape[0]

    return volume

def phase_surface(phi_fn, x, epsilon, V_box=8):
    phi_vals = vmap(lambda x_i: phi_fn(jnp.atleast_2d(x_i)).squeeze())(x)
    grad_val = grad_phi(phi_fn, x)
    sq_grad = jnp.sum(grad_val ** 2, axis=1)

    value = epsilon * sq_grad + (1 / 2.0 / epsilon) * (1 - phi_vals**2)**2
    value *= 3/4/jnp.sqrt(2)
    area = jnp.sum(value)
    area *= V_box / x.shape[0]
    
    return area


