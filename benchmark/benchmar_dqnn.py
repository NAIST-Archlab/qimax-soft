import time
from sojo.stabilizer import StabilizerGenerator
import numpy as np


def wchain_xyz_sojo(num_qubits, num_layers):
    params = np.random.uniform(0, 2 * np.pi, 3 * num_qubits * num_layers)
    stb = StabilizerGenerator(num_qubits)
    for l in range(num_layers):
        for j in range(num_qubits - 1):
            stb.cx([j, j + 1])
        stb.cx([num_qubits - 1, 0])
        k = 0
        for j in range(num_qubits):
            stb.rx(params[k * l], j)
            stb.ry(params[k * l + 1], j)
            stb.rz(params[k * l + 2], j)
            k += 3

    dm = stb.generate_density_matrix_by_generator_jax()
    return dm

import qiskit
import qiskit.quantum_info as qi

def wchain_xyz_qiskit(num_qubits, num_layers):
    params = np.random.uniform(0, 2 * np.pi, 3 * num_qubits * num_layers)
    qc = qiskit.QuantumCircuit(num_qubits)
    for l in range(num_layers):
        for j in range(num_qubits - 1):
            qc.cx(j, j + 1)
        qc.cx(num_qubits - 1, 0)
        k = 0
        for j in range(num_qubits):
            qc.rx(params[k * l], j)
            qc.ry(params[k * l + 1], j)
            qc.rz(params[k * l + 2], j)
            k += 3

    sv = qi.Statevector.from_instruction(qc).data
    dm = np.outer(sv, sv.conj())
    return dm


num_qubits = 6
num_layers = 10

start = time.time()
dm_sojo = wchain_xyz_sojo(num_qubits, num_layers)
end = time.time()
print(f"SOJO: {end - start}")

start = time.time()
dm_qiskit = wchain_xyz_qiskit(num_qubits, num_layers)
end = time.time()
print(f"QISKIT: {end - start}")