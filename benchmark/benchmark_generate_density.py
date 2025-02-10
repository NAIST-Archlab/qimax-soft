from qimax.stabilizer import StabilizerGenerator, PauliTerm, PauliWord
from qimax.utils import generate_pauli_combination
import numpy as np
import time

paulis = generate_pauli_combination(3)
result = {key: [np.random.rand()] for key in paulis}
s = PauliTerm(result)

begin = time.time()
s.to_matrix_with_i_jax()
print(time.time() - begin)