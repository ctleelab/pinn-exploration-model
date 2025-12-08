"""
landscape.py

Visualize 1D/2D loss landscapes of a PINN trained with data (L1) and physics (L2) terms.
Includes:
- pytree utility functions
- filter-normalized direction generation
- loss evaluation and contour plotting
- (placeholders for Hessian and gradient correlation analysis)
"""

# ============================================================
# 1. Imports
# ============================================================
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import tree_util as jtu
from tqdm import tqdm

# ============================================================
# 2. Pytree utilities
# ============================================================
def tree_l2norm(tree):
    return jnp.sqrt(sum([jnp.vdot(x, x).real for x in jtu.tree_leaves(tree)]))

def tree_dot(a, b):
    return sum([jnp.vdot(x, y).real for x, y in zip(jtu.tree_leaves(a),
                                                     jtu.tree_leaves(b))])

def tree_scale(tree, s):
    return jtu.tree_map(lambda x: x * s, tree)

def tree_add(a, b):
    return jtu.tree_map(lambda x, y: x + y, a, b)

def random_like_tree(params, key):
    leaves, treedef = jtu.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    rnd = [jax.random.normal(k, x.shape) for k, x in zip(keys, leaves)]
    return jtu.tree_unflatten(treedef, rnd)

# ============================================================
# 3. Build filter-normalized directions
# ============================================================
def filter_normalize_per_tensor(params, direction, eps=1e-12):
    def norm_match(p, d):
        nd = jnp.linalg.norm(d)
        np_ = jnp.linalg.norm(p)
        return jnp.zeros_like(d) if nd == 0 else d * (np_ / (nd + eps))
    return jtu.tree_map(norm_match, params, direction)

def gram_schmidt_pair_global(d1, d2, eps=1e-12):
    denom = tree_dot(d1, d1) + eps
    proj = tree_scale(d1, tree_dot(d2, d1) / denom)
    d2o = tree_add(d2, tree_scale(proj, -1.0))
    return d1, d2o

def build_landscape_directions(params, rng):
    k1, k2 = jax.random.split(rng)
    u1 = random_like_tree(params, k1)
    u2 = random_like_tree(params, k2)
    d1 = filter_normalize_per_tensor(params, u1)
    d2 = filter_normalize_per_tensor(params, u2)
    d1, d2 = gram_schmidt_pair_global(d1, d2)
    d2 = filter_normalize_per_tensor(params, d2)
    return d1, d2

# ============================================================
# 4. Loss evaluation helpers (adapt for your setup)
# ============================================================
def make_phi_fn(state, params):
    # return lambda x: state.apply_fn({'params': params}, x.reshape(-1, 3))
    return lambda x: state.apply_fn(params, x.reshape(-1, 3))

def eval_L1(state, params, cryoET_data, loss_data):
    phi_fn = make_phi_fn(state, params)
    Ld = loss_data(phi_fn, cryoET_data, state.threshold)
    return state.lambda_1 * Ld

def eval_L2(state, params, x_train, loss_physics):
    phi_fn = make_phi_fn(state, params)
    Lp = loss_physics(phi_fn, x_train)
    return state.lambda_2 * Lp

def eval_L_full(state, params, cryoET_data, x_train, loss_data, loss_physics):
    phi_fn = make_phi_fn(state, params)
    Ld = loss_data(phi_fn, cryoET_data, state.threshold)
    Lp = loss_physics(phi_fn, x_train)
    return state.lambda_1 * Ld + state.lambda_2 * Lp

# ============================================================
# 5. Landscape evaluation + plotting
# ============================================================
def add_directions(params, d1, d2, alpha, beta):
    return jtu.tree_map(lambda p, u, v: p + alpha * u + beta * v, params, d1, d2)

def compute_landscape(state, params_star, d1, d2, alphas, betas,
                      cryoET_data, x_train, loss_data, loss_physics,
                      which="L_full", show_progress=True):
    alphas = np.asarray(alphas)
    betas  = np.asarray(betas)
    Z = np.zeros((len(alphas), len(betas)), dtype=np.float64)

    # Select loss function
    if which == "L1":
        loss_eval = lambda p: eval_L1(state, p, cryoET_data, loss_data)
    elif which == "L2":
        loss_eval = lambda p: eval_L2(state, p, x_train, loss_physics)
    else:
        loss_eval = lambda p: eval_L_full(state, p, cryoET_data, x_train, loss_data, loss_physics)

    loss_eval_jit = jax.jit(loss_eval)

    alpha_iter = tqdm(alphas, desc="alpha") if show_progress else alphas

    for i, a in enumerate(alpha_iter):
        for j, b in enumerate(betas):
            p_ab = add_directions(params_star, d1, d2, a, b)
            Z[i, j] = float(loss_eval_jit(p_ab))
    return Z



def plot_contour(alphas, betas, Z, title):
    plt.figure(figsize=(6,5))
    CS = plt.contourf(alphas, betas, Z.T, levels=50)
    plt.colorbar(CS, label="Loss")
    plt.xlabel("α")
    plt.ylabel("β")
    plt.title(title)
    plt.tight_layout()

def plot_contour_log(alphas, betas, Z, title):
    import matplotlib.pyplot as plt
    Zlog = np.log1p(np.maximum(Z - np.nanmin(Z), 0.0))  # shift & log(1+)
    plt.figure(figsize=(6,5))
    CS = plt.contourf(alphas, betas, Zlog.T, levels=50)
    plt.colorbar(CS, label="log(1 + shifted loss)")
    plt.xlabel("α"); plt.ylabel("β"); plt.title(title); plt.tight_layout()

# ============================================================
# 6. (Optional) Hessian and Gradient Correlation placeholders
# ============================================================
# def compute_hessian_spectrum(state, params, cryoET_data, x_train, loss_data, loss_physics):
#     # Placeholder for next step
#     pass

def compute_gradient_correlation(state, params, cryoET_data, x_train, loss_data, loss_physics):
    # Placeholder for next step
    pass

def max_relative_change(params, d1, d2, alpha, beta, eps=1e-12):
    def rel(p, u, v):
        num = jnp.linalg.norm(alpha * u + beta * v)
        den = jnp.linalg.norm(p) + eps
        return num / den
    # Compute per-tensor ratios
    rels = jtu.tree_leaves(jtu.tree_map(rel, params, d1, d2))
    return float(jnp.max(jnp.stack(rels)))

# # ============================================================
# # 7. Example main entry point
# # ============================================================
# if __name__ == "__main__":
#     # Example usage (pseudo-code):
#     # state, params, cryoET_data, x_train, loss_data, loss_physics = load_your_stuff()
#     rng = jax.random.PRNGKey(0)
#     d1, d2 = build_landscape_directions(params, rng)

#     alphas = np.linspace(-0.8, 0.8, 33)
#     betas  = np.linspace(-0.8, 0.8, 33)

#     Z_L1 = compute_landscape(state, params, d1, d2, alphas, betas,
#                              cryoET_data, x_train, loss_data, loss_physics, which="L1")
#     Z_Lfull = compute_landscape(state, params, d1, d2, alphas, betas,
#                                 cryoET_data, x_train, loss_data, loss_physics, which="L_full")

#     plot_contour(alphas, betas, Z_L1, "L1 (Data term)")
#     plot_contour(alphas, betas, Z_Lfull, "L1 + λ L2 (Full)")
#     plt.show()



# ============================================================
# Hessian spectrum analysis
# ============================================================

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax.flatten_util import ravel_pytree
import numpy as np


def loss_scalar(params, state, cryoET_data, x_train, loss_data, loss_physics, which="L1"):
    phi_fn = make_phi_fn(state, params)
    Ld = loss_data(phi_fn, cryoET_data, state.threshold)
    if which == "L1":
        return state.lambda_1 * Ld
    Lp = loss_physics(phi_fn, x_train)
    return state.lambda_1 * Ld + state.lambda_2 * Lp


def make_hvp_fn(params_star, state, cryoET_data, x_train, loss_data, loss_physics, which="L1"):
    # Flatten/unflatten helpers
    flat0, unflatten = ravel_pytree(params_star)

    # Scalar loss accepting *pytree* params
    def f_pytree(params):
        return loss_scalar(params, state, cryoET_data, x_train, loss_data, loss_physics, which)

    # Gradient wrt pytree params
    grad_f = jax.grad(f_pytree)

    # Build HVP on the *flat* space: v (R^D) -> H v (R^D)
    def hvp_flat(v_flat):
        v_tree = unflatten(v_flat)
        # jvp of grad f in direction v
        _, hvp_tree = jax.jvp(grad_f, (params_star,), (v_tree,))
        hvp_flat, _ = ravel_pytree(hvp_tree)
        return hvp_flat

    return hvp_flat, flat0.size


def lanczos(hvp, dim, k=30, rng_key=0, reorth=True, eps=1e-12):
    """
    hvp: function R^dim -> R^dim
    Returns: evals (k,), evecs (k, dim), and tridiagonal T (alpha, beta)
    """
    key = jax.random.PRNGKey(rng_key)
    q = jax.random.normal(key, (dim,))
    q = q / (jnp.linalg.norm(q) + 1e-12)

    Q = []
    alpha = []
    beta = []

    q_prev = jnp.zeros_like(q)
    for i in range(k):
        # v = H q_i
        v = hvp(q)
        # alpha_i
        a = jnp.dot(q, v)
        alpha.append(a)

        # v <- v - alpha_i q - beta_{i-1} q_{i-1}
        v = v - a * q - (beta[-1] * q_prev if i > 0 else 0.0)

        if reorth and len(Q) > 0:
            # Modified Gram–Schmidt re-orthogonalization
            for qj in Q:
                v = v - jnp.dot(v, qj) * qj

        b = jnp.linalg.norm(v)
        beta.append(b)

        Q.append(q)
        q_prev, q = q, jnp.where(b > eps, v / b, jnp.zeros_like(v))
        if b < eps:  # early termination
            break

    # Build tridiagonal T
    alpha = jnp.array(alpha)
    beta = jnp.array(beta[:-1])  # last beta not used on diagonal below last row

    T = jnp.diag(alpha) + jnp.diag(beta, 1) + jnp.diag(beta, -1)
    # Ritz eigenpairs of T
    evals, eigvecs_T = jnp.linalg.eigh(T)

    # Map Ritz vectors back to R^dim (optional; can be large)
    Qmat = jnp.stack(Q, axis=1)  # dim x m
    evecs = Qmat @ eigvecs_T      # dim x m
    return np.array(evals), np.array(evecs.T), np.array(alpha), np.array(beta)


def hessian_spectrum_topk(params_star, state, cryoET_data, x_train,
                          loss_data, loss_physics, which="L1",
                          k=30, rng_key=0):

    hvp, dim = make_hvp_fn(params_star, state, cryoET_data, x_train,
                           loss_data, loss_physics, which)
    evals, evecs, alpha, beta = lanczos(hvp, dim, k=k, rng_key=rng_key, reorth=True)
    # Sort descending by curvature magnitude
    idx = np.argsort(evals)[::-1]
    return evals[idx], evecs[idx]


# ============================================================
# Calculating full dense Hessian
# ============================================================

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

def compute_hessian_spectrum(params_star, state, cryoET_data, x_train, loss_data, loss_physics, topk=None):
    """
    Compute the Hessian of eval_L_full around params_star and return its eigenvalues/eigenvectors.

    Args:
        params_star: PyTree of trained parameters
        state: training state object containing lambda_1, lambda_2, threshold, etc.
        cryoET_data, x_train, loss_data, loss_physics: same as in eval_L_full
        topk (int or None): if set, return only the largest `topk` eigenvalues/vectors

    Returns:
        eigenvalues, eigenvectors
            eigenvalues: (P,) or (topk,)
            eigenvectors: (P, P) or (P, topk)
    """
    # 1. Flatten params into a vector
    theta_star, unravel = ravel_pytree(params_star)

    # 2. Define scalar loss with flat input
    def flat_loss(theta_flat):
        params = unravel(theta_flat)
        phi_fn = make_phi_fn(state, params)
        Ld = loss_data(phi_fn, cryoET_data, state.threshold)
        Lp = loss_physics(phi_fn, x_train)
        return state.lambda_1 * Ld + state.lambda_2 * Lp

    # 3. Compute full dense Hessian
    H_fn = jax.jit(jax.hessian(flat_loss))
    H = H_fn(theta_star)

    # 4. Symmetrize (to remove tiny numerical asymmetry)
    H = 0.5 * (H + H.T)

    # 5. Eigen decomposition
    eigvals, eigvecs = jnp.linalg.eigh(H)

    # 6. Optionally, return only top-k
    if topk is not None:
        eigvals = eigvals[-topk:]
        eigvecs = eigvecs[:, -topk:]

    return eigvals, eigvecs


# ============================================================
# Calculation using HVP (Hessian–vector product) + Krylov
# ============================================================


import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

def topk_hessian_eigs_(params_star, state, cryoET_data, x_train, loss_data, loss_physics,
                      k=10, block=None, iters=80, tol=1e-6, key=jax.random.PRNGKey(0)):
    """
    Top-k eigenpairs of the Hessian via block subspace iteration + HVPs.
    Returns (eigvals[k], eigvecs[P,k]) where eigvecs are in the flattened param space.
    """
    theta_star, unravel = ravel_pytree(params_star)

    # Flat scalar loss
    def flat_loss(theta_flat):
        params = unravel(theta_flat)
        phi_fn = make_phi_fn(state, params)
        Ld = loss_data(phi_fn, cryoET_data, state.threshold)
        Lp = loss_physics(phi_fn, x_train)
        return state.lambda_1 * Ld + state.lambda_2 * Lp

    # HVP on a block of vectors [P, b]
    @jax.jit
    def hvp_block(theta, V):
        def hv_single(v):
            return jax.jvp(jax.grad(flat_loss), (theta,), (v,))[1]
        return jax.vmap(hv_single, in_axes=1, out_axes=1)(V)

    P = theta_star.size
    b = block or max(k, min(k+4, k*2))  # small oversampling is helpful

    # Init block with Gaussian vectors and orthonormalize
    V = jax.random.normal(key, (P, b))
    V, _ = jnp.linalg.qr(V, mode="reduced")

    prev_top = None
    for _ in range(iters):
        W = hvp_block(theta_star, V)        # apply H
        T = V.T @ W                         # [b,b] Rayleigh-Ritz
        vals, Q = jnp.linalg.eigh(T)        # ascending
        V = V @ Q                           # Ritz vectors
        V, _ = jnp.linalg.qr(V, mode="reduced")

        top_vals = vals[-k:]
        if prev_top is not None and jnp.max(jnp.abs(top_vals - prev_top)) < tol:
            break
        prev_top = top_vals

    # Final extraction
    W = hvp_block(theta_star, V)
    T = V.T @ W
    vals, Q = jnp.linalg.eigh(T)
    V = V @ Q
    # return vals[-k:], V[:, -k:]

    # Sort in descending order
    idx = jnp.argsort(vals)[::-1]
    vals = vals[idx]
    V = V[:, idx]


    eigvals = vals[:k]
    eigvecs = V[:, :k]

    # SANITY CHECK 
    # (a) Orthonormality of eigenvectors
    I_approx = eigvecs.T @ eigvecs
    print("‖VᵀV - I‖:", float(jnp.linalg.norm(I_approx - jnp.eye(I_approx.shape[0]))))

    # (b) Rayleigh quotient matches λ
    Hv = hvp_block(theta_star, eigvecs)
    rq = jnp.sum(eigvecs * Hv, axis=0)  # since columns are ~unit-norm, this ~ vᵀHv
    print("max |rq - λ|:", float(jnp.max(jnp.abs(rq - eigvals))))

    # (c) Cosine between Hv and λ v (should be ~1)
    cos = jnp.sum(Hv * (eigvecs * eigvals), axis=0) / (
          jnp.linalg.norm(Hv, axis=0) * jnp.linalg.norm(eigvecs * eigvals, axis=0) + 1e-12)
    print("min cosine(Hv, λv):", float(jnp.min(cos)))


    # Return top-k largest
    return vals[:k], V[:, :k]    

# Usage:
# eigvals, eigvecs = topk_hessian_eigs(params_star, state, cryoET_data, x_train, loss_data, loss_physics, k=10)




import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

def topk_hessian_eigs(params_star, state, cryoET_data, x_train, loss_data, loss_physics,
                      k=10, block=None, iters=200, tol=1e-6, key=jax.random.PRNGKey(0),
                      use_float64=True):
    """
    Top-k (largest) eigenpairs of the Hessian via block subspace iteration + HVPs.
    Returns (eigvals[k], eigvecs[P,k], abs_residuals[k], rel_residuals[k]).
    - In-situ residuals are computed from W,V,T to avoid re-evaluating H with a different closure.
    - Results are sorted in DESCENDING order (largest -> smaller).
    """
    theta_star, unravel = ravel_pytree(params_star)
    if use_float64:
        theta_star = theta_star.astype(jnp.float64)

    # Flat scalar loss (deterministic: ensure eval mode / fixed batch on your side)
    def flat_loss(theta_flat):
        params = unravel(theta_flat)
        phi_fn = make_phi_fn(state, params)
        Ld = loss_data(phi_fn, cryoET_data, state.threshold)
        Lp = loss_physics(phi_fn, x_train)
        # return state.lambda_1 * Ld + state.lambda_2 * Lp
        return state.lambda_1 / 1e6 * Ld + state.lambda_2 / 1e6 * Lp

    # HVP on a block [P, b]
    @jax.jit
    def hvp_block(theta, V):
        def hv_single(v):
            return jax.jvp(jax.grad(flat_loss), (theta,), (v,))[1]
        Hv = jax.vmap(hv_single, in_axes=1, out_axes=1)(V)
        return Hv.astype(theta.dtype)

    P = theta_star.size
    b = block or max(k, min(k + 8, k * 2))  # a bit more oversampling helps

    # Init block (Gaussian) and orthonormalize
    V = jax.random.normal(key, (P, b), dtype=theta_star.dtype)
    V, _ = jnp.linalg.qr(V, mode="reduced")

    prev_top = None
    for _ in range(iters):
        # Apply H
        W = hvp_block(theta_star, V)     # [P, b]
        # Rayleigh-Ritz small eigenproblem
        T = V.T @ W                      # [b, b] symmetric
        vals, Q = jnp.linalg.eigh(T)     # ascending

        # Update Ritz vectors and re-orthonormalize
        V = V @ Q                        # Ritz vectors basis
        V, _ = jnp.linalg.qr(V, mode="reduced")

        # Convergence check on current top-k (still ascending here)
        top_vals = vals[-k:]
        if prev_top is not None and jnp.max(jnp.abs(top_vals - prev_top)) < tol:
            break
        prev_top = top_vals

    # Final extraction (use in-situ quantities to compute residuals consistently)
    W = hvp_block(theta_star, V)         # H V
    T = V.T @ W                          # small symmetric
    vals, Q = jnp.linalg.eigh(T)         # ascending
    V_ritz = V @ Q                       # Ritz vectors in full space

    # In-situ Ritz residuals: r_i = || H v_i - λ_i v_i ||
    # Use WQ as H(VQ) and VQ * vals to avoid fresh HVP calls.
    WQ = W @ Q                           # [P, b]
    R_block = WQ - V_ritz * vals         # broadcast λ over columns
    abs_res = jnp.linalg.norm(R_block, axis=0)      # (b,)
    # Since columns of V_ritz are orthonormal (up to QR / eigh numerics), ‖v_i‖≈1:
    rel_res = abs_res / (jnp.abs(vals) + 1e-12)

    # Sort DESCENDING and take top-k
    idx = jnp.argsort(vals)[::-1]
    vals = vals[idx][:k]
    V_ritz = V_ritz[:, idx][:, :k]
    abs_res = abs_res[idx][:k]
    rel_res = rel_res[idx][:k]


    # SANITY CHECK 
    eigvals = vals
    eigvecs = V_ritz
    # (a) Orthonormality of eigenvectors
    I_approx = eigvecs.T @ eigvecs
    print("‖VᵀV - I‖:", float(jnp.linalg.norm(I_approx - jnp.eye(I_approx.shape[0]))))

    # (b) Rayleigh quotient matches λ
    Hv = hvp_block(theta_star, eigvecs)
    rq = jnp.sum(eigvecs * Hv, axis=0)  # since columns are ~unit-norm, this ~ vᵀHv
    print("max |rq - λ|:", float(jnp.max(jnp.abs(rq - eigvals))))

    # (c) Cosine between Hv and λ v (should be ~1)
    cos = jnp.sum(Hv * (eigvecs * eigvals), axis=0) / (
          jnp.linalg.norm(Hv, axis=0) * jnp.linalg.norm(eigvecs * eigvals, axis=0) + 1e-12)
    print("min cosine(Hv, λv):", float(jnp.min(cos)))


    return vals, V_ritz, abs_res, rel_res




def make_flat_loss(params_template, state, cryoET_data, x_train, loss_data, loss_physics):
    _, unravel = ravel_pytree(params_template)
    def flat_loss(theta_flat):
        params = unravel(theta_flat)
        phi_fn = make_phi_fn(state, params)
        Ld = loss_data(phi_fn, cryoET_data, state.threshold)
        Lp = loss_physics(phi_fn, x_train)
        return state.lambda_1 * Ld + state.lambda_2 * Lp
    return flat_loss

def ritz_residuals(params_star, state, cryoET_data, x_train, loss_data, loss_physics,
                   eigvals, eigvecs, eps=1e-12):
    """
    Compute Ritz residuals for columns of eigvecs (flattened-space eigenvectors).
    eigvals: (k,), eigvecs: (P, k)
    Returns: residuals (k,), relative_residuals (k,)
    """
    theta_star, _ = ravel_pytree(params_star)
    flat_loss = make_flat_loss(params_star, state, cryoET_data, x_train, loss_data, loss_physics)

    # HVP on a block [P, k]
    @jax.jit
    def hvp_block(theta, V):
        def hv_single(v):
            return jax.jvp(jax.grad(flat_loss), (theta,), (v,))[1]
        return jax.vmap(hv_single, in_axes=1, out_axes=1)(V)

    Hv = hvp_block(theta_star, eigvecs)                # [P, k]
    R = Hv - eigvecs * eigvals                         # broadcast λ over columns
    res = jnp.linalg.norm(R, axis=0)                   # (k,)

    vnorm = jnp.linalg.norm(eigvecs, axis=0) + eps
    rel_res = res / ((jnp.abs(eigvals) + eps) * vnorm) # (k,)
    return res, rel_res



# ============================================================
# Gradient correlation analysis
# ============================================================

def grad_pytree_flatten_norm(grad_tree):
    # Flatten pytree and compute flat array + norm
    leaves, _ = jtu.tree_flatten(grad_tree)
    flat = jnp.concatenate([jnp.ravel(x) for x in leaves])
    norm = jnp.linalg.norm(flat) + 1e-12
    return flat, norm

def gradient_correlation(params, state, cryoET_data, x_train, loss_data, loss_physics):
    # Compute grads of each loss
    def f_L1(p):
        phi_fn = lambda x: state.apply_fn({'params': p}, x.reshape(-1, 3))
        return loss_data(phi_fn, cryoET_data, state.threshold)

    def f_L2(p):
        phi_fn = lambda x: state.apply_fn({'params': p}, x.reshape(-1, 3))
        return loss_physics(phi_fn, x_train)

    g1 = jax.grad(f_L1)(params)
    g2 = jax.grad(f_L2)(params)

    # Flatten and compute cosine similarity
    g1_flat, n1 = grad_pytree_flatten_norm(g1)
    g2_flat, n2 = grad_pytree_flatten_norm(g2)

    cosine = jnp.dot(g1_flat, g2_flat) / (n1 * n2)
    return float(cosine), float(n1), float(n2)





