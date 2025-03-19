import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np

# ======= 1️⃣ Generate Signed Distance Function (SDF) =======
def generate_sdf(grid_size=64, radius=0.5):
    """Generate a signed distance function (SDF) for a sphere in a [-1,1]^3 space."""
    x, y, z = jnp.meshgrid(
        jnp.linspace(-1, 1, grid_size),
        jnp.linspace(-1, 1, grid_size),
        jnp.linspace(-1, 1, grid_size),
        indexing="ij"
    )

    # Compute SDF for a sphere
    sdf_values = jnp.sqrt(x**2 + y**2 + z**2) - radius

    # Flatten grid points for training
    grid_points = jnp.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    sdf_values = sdf_values.ravel()  # Flatten the SDF values

    return grid_points, sdf_values.reshape(grid_size, grid_size, grid_size)

# ======= 2️⃣ Define the Neural Network Model =======
import flax.linen as nn

class SDFNet(nn.Module):
    """Simple fully connected neural network to learn SDF."""
    
    @nn.compact  # ✅ Required to define layers inside __call__()
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)  # Output single SDF value
        return x.squeeze()   # Ensure scalar output


# ======= 3️⃣ Training Functions =======
def train_step(state, x, y, model):
    """Single training step using gradient descent."""

    def loss_fn(params):
        pred = model.apply(params, x)  # ✅ Correctly applies the model
        mse_loss = jnp.mean((pred - y) ** 2)
        sign_loss = jnp.mean(jnp.abs(jnp.sign(pred) - jnp.sign(y)))
        # return mse_loss + 0.1 * sign_loss
        return mse_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


# ✅ Tell JAX that `model` is static
train_step = jax.jit(train_step, static_argnames=["model"])



# ======= 4️⃣ Main Training Loop =======
from flax.training import train_state

def train_sdf_model():
    """Train NN to learn SDF of a sphere."""
    # Generate Training Data
    grid_size = 64
    grid_points, true_sdf = generate_sdf(grid_size=grid_size, radius=0.5)

    # Initialize Model
    model = SDFNet()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 3)))  # Initialize with dummy input
    optimizer = optax.adam(1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # Select random training points
    train_idx = np.random.choice(grid_points.shape[0], 10000, replace=False)
    x_train = grid_points[train_idx]
    y_train = true_sdf.ravel()[train_idx]
    # x_train = grid_points
    # y_train = true_sdf.ravel()

    # Training Loop
    num_steps = 500
    for step in range(num_steps):
        state, loss = train_step(state, x_train, y_train, model)  # ✅ Correct order!
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss:.6f}")

    return state, model, grid_points, true_sdf


# ======= 5️⃣ Run Training and Visualize Results =======
if __name__ == "__main__":
    # Train the model
    state, model, grid_points, true_sdf = train_sdf_model()

    # Compute NN-predicted SDF on full grid
    predicted_sdf = model.apply(state.params, grid_points).reshape(true_sdf.shape)

    # Select a middle slice at Z=0
    slice_idx = true_sdf.shape[2] // 2

    # Plot results
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].imshow(true_sdf[:, :, slice_idx].T, cmap="bwr", origin="lower", vmin=-1, vmax=1)
    axs[0].set_title("True SDF (Slice at Z=0)")
    axs[0].set_xlabel("X-axis")
    axs[0].set_ylabel("Y-axis")
    cbar1 = fig.colorbar(axs[0].images[0], ax=axs[0])
    cbar1.set_label("Signed Distance")

    axs[1].imshow(predicted_sdf[:, :, slice_idx].T, cmap="bwr", origin="lower", vmin=-1, vmax=1)
    axs[1].set_title("NN-Predicted SDF (Slice at Z=0)")
    axs[1].set_xlabel("X-axis")
    axs[1].set_ylabel("Y-axis")
    cbar2 = fig.colorbar(axs[1].images[0], ax=axs[1])
    cbar2.set_label("Signed Distance")

    plt.tight_layout()
    plt.show()

    # Print Min/Max Values
    print(f"True SDF Min: {true_sdf.min()}, Max: {true_sdf.max()}")
    print(f"NN-Predicted SDF Min: {predicted_sdf.min()}, Max: {predicted_sdf.max()}")
