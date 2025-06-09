import matplotlib.pyplot as plt
from skimage import measure
import jax
import jax.numpy as jnp
import numpy as np
from pinn.model import PINN, laplacian_phi, grad_phi, hessian_phi
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pinn.model import phase_volume, phase_surface
from matplotlib.colors import LinearSegmentedColormap


