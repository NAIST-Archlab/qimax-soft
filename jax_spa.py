from jax.experimental import sparse
import jax.numpy as jnp
import numpy as np
import time
import jax
from scipy.sparse import random as sparse_random
from scipy.sparse import csr_matrix
from sojo import stabilizer



w1 = stabilizer.PauliWord(0.5, 'xiizyyzixxxxxizizixxxxxxxxx')
w2 = stabilizer.PauliWord(0.707, 'xiizyyzixxxxyyyyzixxxxxxxxx')
w3 = stabilizer.PauliWord(-0.07, 'xiizyyziyxxyyyyyzixxxxxxxxx')
w4 = stabilizer.PauliWord(-0.04, 'xiizyyziyxxyyyyyzixxxxxxxxx')
p = stabilizer.PauliTerm([w1,w2, w3, w4])

def add_matrices(m1, m2):
    return m1 + m2
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
        vectorized_multiply = jax.vmap(add_matrices)
        matrices = vectorized_multiply(x1, x2)
        x1 = None
        x2 = None
    return matrices[0]


begin = time.time()

p.words[0].to_pc()
for word in p.words[1:]:
    word.to_pc()
matrix = p.to_matrix('csr')

print('Prepare csr', time.time() - begin)

begin = time.time()
matrix = p.to_matrix('coo')

print('Prepare3', time.time() - begin)


begin = time.time()
M_scipy = matrix
MM_scipy = M_scipy @ M_scipy
print('SciPy', time.time() - begin)

# begin = time.time()
# matrix1 = matrix @ matrix
# print("Naive", time.time() - begin)

@jax.jit
def spMM():
    M_sp = sparse.BCOO.from_scipy_sparse(M_scipy)
    MM = M_sp @ M_sp
begin = time.time()
spMM()
print('JAX', time.time() - begin)
