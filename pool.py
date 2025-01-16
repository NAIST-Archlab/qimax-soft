import time
from sojo.stabilizer import StabilizerGenerator
import numpy as np

num_qubits = 3
num_layers = 10
params = np.random.uniform(0, 2 * np.pi, 3 * num_qubits * num_layers)
stb = StabilizerGenerator(num_qubits)
stb.map_parallel('rx', 0, params[0])
# k = 0
# for j in range(num_qubits):
#     stb.map_parallel('rx', j, params[k], )
#     stb.map_parallel('ry', j, params[k+ 1],)
#     stb.map_parallel('rz', j, params[k+ 2], )
#     k += 3

# dm = stb.generate_density_matrix_by_generator_jax()
