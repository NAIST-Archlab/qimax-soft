import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

# Number of devices
num_devices = 5
print(f"Number of devices: {num_devices}")

# Create example BCOO batched sparse matrix
batch_size = num_devices  # To fully utilize all devices
shape = (batch_size, 5, 5)
nnz = 3  # Number of non-zero entries per matrix

# Random indices and values for sparse matrix
indices = jax.random.randint(jax.random.PRNGKey(0), (batch_size, nnz, 2), 0, 5)
values = jax.random.normal(jax.random.PRNGKey(1), (batch_size, nnz))

# Create batched BCOO sparse tensor
bcoo_tensor = BCOO((values, indices), shape=shape)

# `pmap` to parallelize summation across devices
def sum_sparse_matrix(tensor):
    return tensor.sum(axis=1)  # Summing along axis 1

# Parallel mapping with pmap
parallel_sum_result = jax.pmap(sum_sparse_matrix)(bcoo_tensor)

# Convert to dense for verification (optional)
dense_result = jax.pmap(lambda x: x.todense())(parallel_sum_result)
print(dense_result)
