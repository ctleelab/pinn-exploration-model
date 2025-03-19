import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import grad, jacfwd, jit, vmap
import optax
from flax.training import train_state

GRID_SIZE = 64

class PINN(nn.Module):
    hidden_dim: int = 64  # Hidden layer size

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)  # Ensure input is at least 2D
        if x.shape[-1] != 3:
            raise ValueError(f"Expected input shape (*,3), but got {x.shape}")
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)  # Output single scalar φ(x,y,z)
        return x.squeeze()


# Compute first derivatives (∇φ) with vectorized differentiation
def grad_phi(phi_fn, x):
    x = x.reshape(-1, 3)  # Ensure correct shape
    return vmap(lambda x_i: grad(lambda x: phi_fn(jnp.atleast_2d(x)).squeeze())(x_i))(x)

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
def loss_data(phi_fn, cryoET_data):
    """
    Compute loss by enforcing φ=0 where I=1 (membrane locations).
    """
    # x_grid, y_grid, z_grid = jnp.meshgrid(
    #     jnp.linspace(-1.5, 1.5, GRID_SIZE),
    #     jnp.linspace(-1.5, 1.5, GRID_SIZE),
    #     jnp.linspace(-1.5, 1.5, GRID_SIZE),
    #     indexing="ij"
    # )
    # x_grid, y_grid, z_grid = jnp.meshgrid(
    #     jnp.linspace(0, GRID_SIZE - 1, GRID_SIZE),
    #     jnp.linspace(0, GRID_SIZE - 1, GRID_SIZE),
    #     jnp.linspace(0, GRID_SIZE - 1, GRID_SIZE),
    #     indexing="ij"
    # )
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
    binary_mask = jnp.where(cryoET_data > 0.5, 1, 0)
    membrane_loss = jnp.mean((phi_fn(grid_points).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE) * binary_mask) ** 2)

    return membrane_loss



# Physics loss (enforces the Helfrich equation)
def loss_physics(phi_fn, x):
    H = mean_curvature(phi_fn, x)
    Delta_H = laplacian_mean_curvature(phi_fn, x)
    K = gaussian_curvature(phi_fn, x)

    physics_residual = Delta_H + 2 * H * (H**2 - K)
    # physics_residual = Delta_H + 2 * H * (H**2)

    # jax.debug.print("Mean Curvature Min/Max: {}/{}", H.min(), H.max())
    # jax.debug.print("Laplacian H Min/Max: {}/{}", Delta_H.min(), Delta_H.max())
    # jax.debug.print("Physics Residual Min/Max: {}/{}", physics_residual.min(), physics_residual.max())

    return jnp.mean(physics_residual**2)

# Combined loss function
def total_loss(phi_fn, x, cryoET_data, lambda_1, lambda_2):
    return lambda_1 * loss_data(phi_fn, cryoET_data) + lambda_2 * loss_physics(phi_fn, x)


