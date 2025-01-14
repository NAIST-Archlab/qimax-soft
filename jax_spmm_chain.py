import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

# Tạo 100 ma trận 100x100 ngẫu nhiên
n_matrices = 32  # Số lượng ma trận
matrix_size = 1000  # Kích thước mỗi ma trận

matrices = [np.random.randn(matrix_size, matrix_size) for _ in range(n_matrices)]


def matrix_multiply(m1, m2):
    """Hàm nhân hai ma trận."""
    return np.matmul(m1, m2)


def reduce_matrices_multithread(matrices):
    """Nhân các cặp ma trận song song cho đến khi chỉ còn 1 ma trận."""
    while len(matrices) > 1:
        results = []

        # Nếu số ma trận lẻ, giữ lại ma trận cuối cùng
        last_matrix = None
        if len(matrices) % 2 != 0:
            last_matrix = matrices.pop()

        # Chia thành các cặp (x0 * x1), (x2 * x3), ...
        pairs = [(matrices[i], matrices[i + 1]) for i in range(0, len(matrices), 2)]

        # Thực hiện nhân các cặp ma trận song song
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda p: matrix_multiply(p[0], p[1]), pairs))

        # Thêm ma trận cuối cùng vào nếu cần
        if last_matrix is not None:
            results.append(last_matrix)

        matrices = results

    return matrices[0]


# Đo thời gian thực hiện
start_time = time.time()
final_result = reduce_matrices_multithread(matrices)
end_time = time.time()

print(f"Thời gian thực hiện multithreading: {end_time - start_time} giây.")