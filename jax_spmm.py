from jax.experimental import sparse
import jax.numpy as jnp
import numpy as np
import time
import jax
from scipy.sparse import random as sparse_random
from scipy.sparse import csr_matrix
from sojo import stabilizer


begin = time.time()
w1 = stabilizer.PauliWord(0.5, 'xiizyyzixxxxxizizixxxxxxxxx')
w2 = stabilizer.PauliWord(0.707, 'xiizyyzixxxxyyyyzixxxxxxxxx')

p = stabilizer.PauliTerm([w1,w2])
print('Prepare1', time.time() - begin)
begin = time.time()
matrix = p.to_matrix('csr')

print('Prepare2', time.time() - begin)

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
