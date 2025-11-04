# === Lightweight checkpoint I/O: no Orbax, no layout/sharding headaches ===
import os, re, io, json, numpy as np, jax
from flax.serialization import to_bytes, from_bytes

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_params_bytes(path, state):
    # Materialize to host; training still runs on GPU.
    state_cpu = jax.device_get(jax.tree_util.tree_map(lambda x: x, state))
    data = to_bytes(state_cpu)  # msgpack bytes
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)  # atomic rename

def load_state_bytes(path, dummy_state):
    with open(path, "rb") as f:
        data = f.read()
    # Restore into dummy_state’s structure/shapes/dtypes
    return from_bytes(dummy_state, data)

def save_loss_npz(path, loss_obj: dict):
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    safe = {k: np.asarray(v) for k, v in loss_obj.items()}
    tmp = path + ".tmp"   # e.g., ".../loss_000000.npz.tmp"

    # Write EXACTLY to tmp (avoid np.savez adding another .npz)
    with open(tmp, "wb") as f:
        np.savez(f, **safe)

    os.replace(tmp, path)  # atomic rename to ".../loss_000000.npz"

def load_loss_npz(path):
    if not os.path.exists(path):
        return None
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}

def find_last_state_path(ckpt_dir):
    last_path = None
    last_step = -1
    pat = re.compile(r"state_(\d{6})\.msgpack$")
    for fn in os.listdir(ckpt_dir):
        m = pat.match(fn)
        if m:
            s = int(m.group(1))
            if s > last_step:
                last_step = s
                last_path = os.path.join(ckpt_dir, fn)
    return last_path, last_step