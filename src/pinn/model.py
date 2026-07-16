import jax
import jax.numpy as jnp
import flax.linen as nn


DEFAULT_EPSILON = 0.05
DEFAULT_NUMERICAL_EPS = 1e-8


class PINN(nn.Module):
    """MLP representing a scalar phase field φ: R³ → [-1, 1]."""

    hidden_dim: int = 128
    num_hidden_layers: int = 2

    @nn.compact
    def __call__(self, x):
        x = jnp.asarray(x)

        if x.shape[-1] != 3:
            raise ValueError(
                f"Expected input shape (..., 3), but received {x.shape}."
            )

        for _ in range(self.num_hidden_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.tanh(x)

        x = nn.Dense(1)(x)
        x = nn.tanh(x)

        return x.squeeze(axis=-1)


# ---------------------------------------------------------------------
# Basic phase-field evaluation and derivatives
# ---------------------------------------------------------------------


def evaluate_phi(phi_fn, points):
    """Evaluate φ at points with shape (..., 3)."""
    points = jnp.asarray(points)

    if points.shape[-1] != 3:
        raise ValueError(
            f"Expected point shape (..., 3), but received {points.shape}."
        )

    return phi_fn(points)


def _phi_scalar(phi_fn, point):
    """Evaluate φ at one point and return a scalar."""
    return phi_fn(point[None, :]).reshape(())


def grad_phi(phi_fn, points):
    """Compute ∇φ at points.

    Parameters
    ----------
    phi_fn
        Function mapping points of shape (N, 3) to values of shape (N,).
    points
        Array with shape (N, 3).

    Returns
    -------
    Array with shape (N, 3).
    """
    points = jnp.asarray(points).reshape(-1, 3)
    scalar_fn = lambda point: _phi_scalar(phi_fn, point)
    return jax.vmap(jax.grad(scalar_fn))(points)


def hessian_phi(phi_fn, points):
    """Compute the Hessian ∇²φ at points.

    Returns an array with shape (N, 3, 3).
    """
    points = jnp.asarray(points).reshape(-1, 3)
    scalar_fn = lambda point: _phi_scalar(phi_fn, point)
    hessian_fn = jax.hessian(scalar_fn)
    return jax.vmap(hessian_fn)(points)


def laplacian_phi(phi_fn, points):
    """Compute Δφ at points."""
    hessian = hessian_phi(phi_fn, points)
    return jnp.trace(hessian, axis1=-2, axis2=-1)


# ---------------------------------------------------------------------
# Geometric quantities
# ---------------------------------------------------------------------


def mean_curvature(
    phi_fn,
    points,
    numerical_eps=DEFAULT_NUMERICAL_EPS,
):
    """Compute the mean curvature of the level sets of φ.

    The convention used here is

        H = 1/2 ∇ · (∇φ / |∇φ|).

    The sign depends on the orientation induced by ∇φ.
    """
    gradient = grad_phi(phi_fn, points)
    hessian = hessian_phi(phi_fn, points)

    grad_norm = jnp.sqrt(
        jnp.sum(gradient**2, axis=-1) + numerical_eps**2
    )
    laplacian = jnp.trace(hessian, axis1=-2, axis2=-1)

    grad_hessian_grad = jnp.einsum(
        "ni,nij,nj->n",
        gradient,
        hessian,
        gradient,
    )

    return 0.5 * (
        laplacian / grad_norm
        - grad_hessian_grad / grad_norm**3
    )


def calc_grad_s_H_point(
    phi_fn,
    point,
    normal_eps=DEFAULT_NUMERICAL_EPS,
    curvature_eps=DEFAULT_NUMERICAL_EPS,
):
    """Compute the surface gradient ∇ₛH at one point."""

    scalar_fn = lambda x: _phi_scalar(phi_fn, x)
    gradient_fn = jax.grad(scalar_fn)
    hessian_fn = jax.hessian(scalar_fn)

    def curvature_fn(x):
        gradient = gradient_fn(x)
        hessian = hessian_fn(x)

        grad_norm = jnp.sqrt(
            jnp.dot(gradient, gradient) + curvature_eps**2
        )
        laplacian = jnp.trace(hessian)
        grad_hessian_grad = jnp.dot(
            gradient,
            hessian @ gradient,
        )

        return 0.5 * (
            laplacian / grad_norm
            - grad_hessian_grad / grad_norm**3
        )

    gradient = gradient_fn(point)
    grad_norm = jnp.sqrt(
        jnp.dot(gradient, gradient) + normal_eps**2
    )
    normal = gradient / grad_norm

    curvature_gradient = jax.grad(curvature_fn)(point)

    projection = (
        jnp.eye(3, dtype=point.dtype)
        - jnp.outer(normal, normal)
    )

    return projection @ curvature_gradient


def calc_grad_s_H(
    phi_fn,
    points,
    normal_eps=DEFAULT_NUMERICAL_EPS,
    curvature_eps=DEFAULT_NUMERICAL_EPS,
):
    """Compute the surface gradient ∇ₛH at multiple points."""
    points = jnp.asarray(points).reshape(-1, 3)

    point_fn = lambda point: calc_grad_s_H_point(
        phi_fn,
        point,
        normal_eps=normal_eps,
        curvature_eps=curvature_eps,
    )

    return jax.vmap(point_fn)(points)


def calc_delta_s_H(
    phi_fn,
    points,
    normal_eps=DEFAULT_NUMERICAL_EPS,
    curvature_eps=DEFAULT_NUMERICAL_EPS,
):
    """Compute the surface Laplacian ΔₛH.

    This requires derivatives of φ up to fourth order and can therefore
    be computationally expensive.
    """
    points = jnp.asarray(points).reshape(-1, 3)

    scalar_fn = lambda point: _phi_scalar(phi_fn, point)
    gradient_fn = jax.grad(scalar_fn)

    def surface_gradient_fn(point):
        return calc_grad_s_H_point(
            phi_fn,
            point,
            normal_eps=normal_eps,
            curvature_eps=curvature_eps,
        )

    gradient = jax.vmap(gradient_fn)(points)
    grad_norm = jnp.sqrt(
        jnp.sum(gradient**2, axis=-1, keepdims=True)
        + normal_eps**2
    )
    normal = gradient / grad_norm

    jacobian = jax.vmap(jax.jacfwd(surface_gradient_fn))(points)

    identity = jnp.eye(3, dtype=points.dtype)
    projection = (
        identity[None, :, :]
        - normal[:, :, None] * normal[:, None, :]
    )

    return jnp.einsum("nij,nji->n", projection, jacobian)


def _cofactor_3x3(matrix):
    """Compute the cofactor matrix of batched 3×3 matrices."""
    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    c = matrix[..., 0, 2]
    d = matrix[..., 1, 0]
    e = matrix[..., 1, 1]
    f = matrix[..., 1, 2]
    g = matrix[..., 2, 0]
    h = matrix[..., 2, 1]
    i = matrix[..., 2, 2]

    return jnp.stack(
        [
            e * i - f * h,
            f * g - d * i,
            d * h - e * g,
            c * h - b * i,
            a * i - c * g,
            b * g - a * h,
            b * f - c * e,
            c * d - a * f,
            a * e - b * d,
        ],
        axis=-1,
    ).reshape(matrix.shape)


def gaussian_curvature(
    phi_fn,
    points,
    numerical_eps=DEFAULT_NUMERICAL_EPS,
):
    """Compute Gaussian curvature of the level sets of φ.

    For an implicit surface,

        K = (∇φᵀ cof(∇²φ) ∇φ) / |∇φ|⁴.
    """
    gradient = grad_phi(phi_fn, points)
    hessian = hessian_phi(phi_fn, points)
    cofactor = _cofactor_3x3(hessian)

    numerator = jnp.einsum(
        "ni,nij,nj->n",
        gradient,
        cofactor,
        gradient,
    )

    grad_norm_sq = (
        jnp.sum(gradient**2, axis=-1)
        + numerical_eps**2
    )

    return numerator / grad_norm_sq**2


# ---------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------


def loss_data(
    phi_fn,
    data,
    epsilon=DEFAULT_EPSILON,
):
    """Match the diffuse-interface energy density to edge labels."""
    points = data["points"]
    labels = data["label"]

    phi = evaluate_phi(phi_fn, points)
    gradient = grad_phi(phi_fn, points)

    grad_norm_sq = jnp.sum(gradient**2, axis=-1)

    energy_density = (
        epsilon**2 * grad_norm_sq
        + 0.5 * (phi**2 - 1.0) ** 2
    )

    return jnp.mean((energy_density - labels) ** 2)


def loss_phys(
    phi_fn,
    data,
    epsilon=DEFAULT_EPSILON,
):
    """Allen–Cahn equilibrium residual loss."""
    points = data["points"]

    phi = evaluate_phi(phi_fn, points)
    laplacian = laplacian_phi(phi_fn, points)

    residual = (
        laplacian
        - (phi**2 - 1.0) * phi / epsilon**2
    )

    return jnp.mean(residual**2)


def loss_sign(phi_fn, data):
    """Supervise the phase-field sign using labels ±1."""
    points = data["points"]
    labels = data["label"]

    phi = evaluate_phi(phi_fn, points)
    return jnp.mean((phi - labels) ** 2)


def loss_curv(phi_fn, data):
    """Penalize spatial variation of mean curvature."""
    points = data["points"]
    surface_gradient = calc_grad_s_H(phi_fn, points)

    return jnp.mean(
        jnp.sum(surface_gradient**2, axis=-1)
    )


# ---------------------------------------------------------------------
# Integral phase-field quantities
# ---------------------------------------------------------------------


def _integral_mean(values, domain_volume):
    """Monte Carlo approximation of an integral over the domain."""
    return domain_volume * jnp.mean(values)


def phase_volume(
    phi_fn,
    points,
    domain_volume=8.0,
):
    """Approximate the volume of the φ = +1 phase."""
    phi = evaluate_phi(phi_fn, points)
    indicator = 0.5 * (1.0 + phi)

    return _integral_mean(indicator, domain_volume)


def phase_surface(
    phi_fn,
    points,
    epsilon,
    domain_volume=8.0,
):
    """Approximate the interface surface area."""
    phi = evaluate_phi(phi_fn, points)
    gradient = grad_phi(phi_fn, points)

    grad_norm_sq = jnp.sum(gradient**2, axis=-1)

    density = (
        epsilon * grad_norm_sq
        + (1.0 - phi**2) ** 2 / (2.0 * epsilon)
    )
    density *= 3.0 / (4.0 * jnp.sqrt(2.0))

    return _integral_mean(density, domain_volume)


def phase_bend(
    phi_fn,
    points,
    epsilon,
    kappa,
    domain_volume=8.0,
):
    """Approximate diffuse-interface bending energy."""
    phi = evaluate_phi(phi_fn, points)
    laplacian = laplacian_phi(phi_fn, points)

    chemical_potential = (
        epsilon * laplacian
        - (phi**2 - 1.0) * phi / epsilon
    )

    density = (
        3.0
        * kappa
        * chemical_potential**2
        / (4.0 * jnp.sqrt(2.0) * epsilon)
    )

    return _integral_mean(density, domain_volume)