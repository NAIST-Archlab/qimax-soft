import jax
import jax.numpy as jnp
import numpy as np
import time
from jax import config
config.update("jax_platform_name", "cpu")

# Tạo 100 ma trận 100x100 ngẫu nhiên
n_matrices = 40000  # Số lượng ma trận
matrix_size = 20  # Kích thước mỗi ma trận

key = jax.random.PRNGKey(0)
matrices = jax.random.normal(key, (n_matrices, matrix_size, matrix_size))


def multiply_naive(ms):
    res = ms[0]
    for j in range(1, n_matrices):
        res = res + ms[j]
    return res

# Hàm thực hiện nhân ma trận cho đến khi còn 1 ma trận duy nhất
@jax.jit
def reduce_matrices(matrices):
    """Nhân liên tiếp các ma trận cho đến khi chỉ còn 1 ma trận."""
    while matrices.shape[0] > 1:
        if matrices.shape[0] % 2 != 0:
            last_matrix = matrices[-1:]
            m1 = matrices[::2][:-1]
        else:
            last_matrix = None
            m1 = matrices[::2]
        m2 = matrices[1::2]

        # Nhân các cặp song song
        vectorized_multiply = jax.vmap(lambda a, b: a + b)
        matrices = vectorized_multiply(m1, m2)
        if last_matrix is not None:
            #matrices = jnp.concatenate([matrices, last_matrix], axis=0)
            matrices = jnp.append(matrices, last_matrix, axis=0)
    return matrices[0]


import time

start = time.time()
final_matrix = reduce_matrices(matrices)
print("Shape", final_matrix[0][0])
end = time.time()
print(f"Thời gian thực hiện: {end - start:.6f} giây")

start = time.time()
final_matrix = multiply_naive(matrices)
print(final_matrix[0][0])
print(f"Nave: {time.time() - start:.6f}")