import jax.numpy as jnp

# cryoET_data.shape == (Z, Y, X) 
# model/function.shape == (x, y, z)

# 
def axes_from_cryo_shape(shape, lo=-1.0, hi=1.0):
    Z, Y, X = shape
    x = jnp.linespace(lo, hi, X)
    y = jnp.linespace(lo, hi, Y)
    z = jnp.linespace(lo, hi, Z)
    return x, y, z 

def phi_on_cryo_grid_xyz(phi_fn, shape, lo=-1.0, hi=1.0):
    x, y, z = axes_from_cryo_shape(shape, lo, hi)               
    Xg, Yg, Zg = jnp.meshgrid(x, y, z, indexing="ij")           # (X, Y, Z)
    pts = jnp.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], -1)   # (N, 3) = (x,y,z)
    phi_xyz = phi_fn(pts).reshape(Xg.shape)                     # (X, Y, Z)
    return phi_xyz, (Xg, Yg, Zg), (x, y, z)