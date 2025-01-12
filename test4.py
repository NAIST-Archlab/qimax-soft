import jax
import jax.numpy as jnp
import numpy as np
import time

# Tạo 100 ma trận 100x100 ngẫu nhiên
n_matrices = 12  # Số lượng ma trận
matrix_size = 2**n_matrices  # Kích thước mỗi ma trận
if n_matrices % 2 != 0:
    n_matrices -= 1  # Đảm bảo số lượng ma trận là chẵn

key = jax.random.PRNGKey(0)
matrices = jax.random.normal(key, (n_matrices, matrix_size, matrix_size))

# Hàm nhân các cặp ma trận
def multiply_matrices(m1, m2):
    return m1 @ m2


# Hàm thực hiện nhân ma trận cho đến khi còn 1 ma trận duy nhất
@jax.jit
def multiply_until_one_matrix(matrices):
    while matrices.shape[0] > 1:
        # Nếu số lượng ma trận lẻ, bỏ ma trận cuối cùng
        if matrices.shape[0] % 2 != 0:
            matrices = matrices[:-1]
        
        # Chia thành các cặp (x0, x1), (x2, x3), ...
        x1 = matrices[::2]
        x2 = matrices[1::2]
        
        # Nhân các cặp song song
        vectorized_multiply = jax.vmap(multiply_matrices)
        matrices = vectorized_multiply(x1, x2)
    return matrices[0]

# Đo thời gian thực hiện
start = time.time()

final_matrix = multiply_until_one_matrix(matrices)
end = time.time()
print(final_matrix[0][0])
print(f"Thời gian thực hiện: {end - start:.6f} giây")
