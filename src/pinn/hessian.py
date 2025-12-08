import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import Callable, Tuple, List, Optional

# ------------------------- PyTree helpers -------------------------
def tree_dot(a, b):
    la, _ = jtu.tree_flatten(a)
    lb, _ = jtu.tree_flatten(b)
    return sum(jnp.vdot(x, y) for x, y in zip(la, lb))

def tree_norm(a):
    return jnp.sqrt(tree_dot(a, a))

def tree_add(*trees):
    return jtu.tree_map(lambda *xs: sum(xs), *trees)

def tree_scaled(t, alpha):
    return jtu.tree_map(lambda x: alpha * x, t)

def random_like_tree(treeref, key):
    leaves, treedef = jtu.tree_flatten(treeref)
    keys = jax.random.split(key, len(leaves))
    rnd = [jax.random.normal(k, x.shape) for k, x in zip(keys, leaves)]
    return jtu.tree_unflatten(treedef, rnd)

@jax.jit
def _normalize(v):
    n = tree_norm(v) + 1e-12
    return jtu.tree_map(lambda x: x / n, v), n

# --------------------- Factory for curvature ops -------------------
def build_curvature_tools(
    compute_losses: Callable[[any], jnp.ndarray]
):
    """
    Returns (HVP, lanczos_topk, hutchinson_trace) specialized to your compute_losses(params).
    - compute_losses(params): scalar loss (no side-effects; deterministic)
    """

    # HVP via JVP of grad
    def _hvp(params, v):
        grad_fn = jax.grad(compute_losses)                    # ∇L(params)
        # JVP of grad along v -> H v
        return jax.jvp(grad_fn, (params,), (v,))[1]

    HVP = jax.jit(_hvp)

    # Lanczos top-k eigenpairs
    def lanczos_topk(
        params,
        key: jax.Array,
        k: int = 10,
        m: Optional[int] = None,
        reorth: bool = True,
    ) -> Tuple[jnp.ndarray, List[any], jnp.ndarray, jnp.ndarray]:
        """
        Returns (evals, vecs, residuals, cosines)
        - evals: (k,) top eigenvalues (descending)
        - vecs : list of k eigenvector pytrees (same structure as params)
        - residuals: ||H v - λ v|| for each pair (size k)
        - cosines  : cos(Hv, v) ~ 1 when v is close to an eigenvector
        """
        if m is None:
            m = 2 * k + 6

        def orth(v, basis):
            if not reorth or len(basis) == 0:
                return v
            for u in basis:
                vu = tree_dot(v, u)
                v = tree_add(v, tree_scaled(u, -vu))
            return v

        q = random_like_tree(params, key)
        q, _ = _normalize(q)
        Q = []                 # Lanczos basis (pytrees)
        alphas, betas = [], []
        beta = 0.0
        prev = None

        for _ in range(m):
            w = HVP(params, q)                     # w = H q_j
            if prev is not None:
                w = tree_add(w, tree_scaled(prev, -beta))  # -β_{j-1} q_{j-1}
            alpha = tree_dot(q, w)                 # α_j
            w = tree_add(w, tree_scaled(q, -alpha))# orth to current q
            w = orth(w, Q)                         # full re-orthogonalize
            beta = tree_norm(w)                    # β_j

            Q.append(q)
            alphas.append(alpha)
            betas.append(beta)

            if beta < 1e-10:
                break
            prev = q
            q = tree_scaled(w, 1.0 / beta)

        # Build symmetric tridiagonal T
        T = jnp.diag(jnp.array(alphas))
        if len(betas) >= 2:
            off = jnp.array(betas[:-1])
            T = T.at[jnp.arange(len(off)), jnp.arange(1, len(off)+1)].set(off)
            T = T.at[jnp.arange(1, len(off)+1), jnp.arange(len(off))].set(off)

        # Small dense eigendecomp
        evals, evecs = jnp.linalg.eigh(T)        # ascending
        idx = jnp.argsort(evals)[::-1][:k]
        evals = evals[idx]
        evecs = evecs[:, idx]

        # Lift Ritz vectors back to parameter space: v_i = Q @ evecs[:, i]
        def lincomb(coeffs, basis):
            return tree_add(*[tree_scaled(b, c) for b, c in zip(basis, coeffs)])
        vecs = [lincomb(evecs[:, i], Q) for i in range(evals.shape[0])]

        # Diagnostics: residuals and cosines
        residuals = []
        cosines = []
        for lam, v in zip(evals, vecs):
            Hv = HVP(params, v)
            r = tree_add(Hv, tree_scaled(v, -lam))
            residuals.append(jnp.sqrt(tree_dot(r, r)))

            v_norm = tree_norm(v)
            Hv_norm = tree_norm(Hv)
            cos = tree_dot(Hv, tree_scaled(v, 1.0/(v_norm + 1e-12))) / (Hv_norm + 1e-12)
            cosines.append(cos)

        return evals, vecs, jnp.stack(residuals), jnp.stack(cosines)

    # Hutchinson trace estimator
    def hutchinson_trace(
        params,
        num_probes: int = 16,
        key: jax.Array = jax.random.PRNGKey(0),
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        keys = jax.random.split(key, num_probes)
        vals = []
        for kk in keys:
            z = random_like_tree(params, kk)
            z, _ = _normalize(z)
            Hz = HVP(params, z)
            vals.append(tree_dot(z, Hz))
        vals = jnp.stack(vals)
        return jnp.mean(vals), jnp.std(vals)

    return HVP, lanczos_topk, hutchinson_trace

