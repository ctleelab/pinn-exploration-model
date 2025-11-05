import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import grad, jacfwd, jit, vmap
from pinn.grid import phi_on_cryo_grid_xyz, axes_from_cryo_shape 
import optax
from flax.training import train_state

# Model Parameters
WEIGHT_IN = 0.5
EPSILON = 0.01
LEARNING_RATE = 1e-3
LAMBDA_1 = 100000.0
LAMBDA_2 = 0.001

class PINN(nn.Module):
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
def laplacian_phi(phi_fn, x, L):
    """
    Compute Δφ with coordinate scaling.

    Args:
        phi_fn: function φ(x)
        x: (N, 3) coordinates (normalized to [-1, 1])
        L: tuple of real domain lengths (Lx, Ly, Lz)
            → scale factor (2/Li)^2 is applied to each 2nd derivative.
    """
    x = x.reshape(-1, 3)
    L = jnp.array(L)
    scale = (2.0 / L) ** 2  # (2/Lx)^2, (2/Ly)^2, (2/Lz)^2

    def lap_single(x_i):

        # Hessian (3x3) = second derivatives matrix
        H = jacfwd(grad(lambda x: phi_fn(jnp.atleast_2d(x)).squeeze()))(x_i)

        # Apply anisotropic scaling on each axis (xx, yy, zz)
        return jnp.sum(scale * jnp.diag(H))
    
    return vmap(lap_single)(x)

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
def loss_data(phi_fn, cryoET_data, thre):
    """
    Compute loss by enforcing φ=0 where I=1 (membrane locations).
    """

    # Get (x, y, z) grid from cryoET_data.shape (Z, Y, X)
    phi_xyz, _, _ = phi_on_cryo_grid_xyz(phi_fn, cryoET_data.shape, lo=-1.0, hi=1.0)
    # Change phi_xyz into phi_zyx when we have to compare with cryoET_data
    phi_zyx = jnp.transpose(phi_xyz, (2, 1, 0))

    if thre != None:
        binary_mask = (cryoET_data > thre).astype(phi_zyx.dtype)
    else:
        binary_mask = cryoET_data

    inside_loss = jnp.sum(binary_mask * (phi_zyx - 0.0) ** 2) / jnp.sum(binary_mask)
    outside_loss = jnp.sum((1 - binary_mask) * (1.0 - phi_zyx ** 2) ** 2) / jnp.sum(1 - binary_mask)
    # inside_loss = jnp.sum(binary_mask * (phi_zyx - 0.0) ** 2) / (jnp.sum(binary_mask) + EPSILON)
    # outside_loss = jnp.sum((1 - binary_mask) * (1.0 - phi_zyx ** 2) ** 2) / (jnp.sum(1 - binary_mask) + EPSILON)
    membrane_loss = WEIGHT_IN * inside_loss + (1 - WEIGHT_IN) * outside_loss

    return membrane_loss


# New physics loss based on Allen-Cahn-like equation
# x is the randomly selected points for Monte Carlo approximation
def loss_physics(phi_fn, x, L):
    phi_vals = vmap(lambda x_i: phi_fn(jnp.atleast_2d(x_i)).squeeze())(x)   # we apply phi function to all the points x to get the phi values
    lap_phi = laplacian_phi(phi_fn, x, L)                                      # we calculate the laplacian for all the points x
    residual = lap_phi - (1 / EPSILON**2) * (phi_vals**2 - 1) * phi_vals    
    return jnp.mean(residual**2)


# Combined loss function
def total_loss(phi_fn, x, cryoET_data, lambda_1, lambda_2, thre):
    return lambda_1 * loss_data(phi_fn, cryoET_data, thre) + lambda_2 * loss_physics(phi_fn, x, cryoET_data.shape)


def phase_volume(phi_fn, x, V_box=8): # volume
    phi_vals = vmap(lambda x_i: phi_fn(jnp.atleast_2d(x_i)).squeeze())(x)
    value = 1 + phi_vals
    volume = jnp.sum(value*0.5)
    volume *= V_box / x.shape[0]

    return volume

def phase_surface(phi_fn, x, V_box=8): # surface area
    phi_vals = vmap(lambda x_i: phi_fn(jnp.atleast_2d(x_i)).squeeze())(x)
    grad_val = grad_phi(phi_fn, x)
    sq_grad = jnp.sum(grad_val ** 2, axis=1)

    value = EPSILON * sq_grad + (1 / 2.0 / EPSILON) * (1 - phi_vals**2)**2
    value *= 3/4/jnp.sqrt(2)
    area = jnp.sum(value)
    area *= V_box / x.shape[0]
    
    return area

def phase_bend(phi_fn, x, kappa, V_box=8): # bending energy
    phi_vals = vmap(lambda x_i: phi_fn(jnp.atleast_2d(x_i)).squeeze())(x)
    lap_phi = laplacian_phi(phi_fn, x)

    value = EPSILON*lap_phi - (1 / EPSILON) * (phi_vals**2 - 1) * phi_vals
    bend = jnp.sum(value**2)
    bend *= 3*kappa/4/jnp.sqrt(2)/EPSILON
    bend *= V_box / x.shape[0]

    return bend


