from sojo.stabilizer import StabilizerGenerator
import numpy as np
import qiskit
import qiskit.quantum_info as qi
import time

num_qubits = 3
num_gates = 8000


stb = StabilizerGenerator(num_qubits)
gates = np.random.choice(['h','cx','s', 'rx', 'ry', 'rz'], num_gates)
for gate in gates:
    index = np.random.choice(num_qubits)
    if gate == 'h':
        stb.h(index)
    elif gate == 'cx':
        target = np.random.choice([x for x in range(num_qubits) if x != index])
        stb.cx([index, target])
    elif gate == 's':
        stb.s(index)
    elif gate == 'rx':
        angle = np.random.uniform(0, 2*np.pi)
        stb.rx(angle, index)
    elif gate == 'ry':
        angle = np.random.uniform(0, 2*np.pi)
        stb.ry(angle, index)
    elif gate == 'rz':
        angle = np.random.uniform(0, 2*np.pi)
        stb.rz(angle, index)
start = time.time()
dm = stb.generate_density_matrix_by_generator_jax()
print("Generate matrix", time.time() - start)

start = time.time()
qc = qiskit.QuantumCircuit(num_qubits)
gates = np.random.choice(['h','cx','s', 'rx', 'ry', 'rz'], num_gates)
for gate in gates:
    index = np.random.choice(num_qubits)
    if gate == 'h':
        qc.h(index)
    elif gate == 'cx':
        target = np.random.choice([x for x in range(num_qubits) if x != index])
        qc.cx(index, target)
    elif gate == 's':
        qc.s(index)
    elif gate == 'rx':
        angle = np.random.uniform(0, 2*np.pi)
        qc.rx(angle, index)
    elif gate == 'ry':
        angle = np.random.uniform(0, 2*np.pi)
        qc.ry(angle, index)
    elif gate == 'rz':
        angle = np.random.uniform(0, 2*np.pi)
        qc.rz(angle, index)

dm = qi.DensityMatrix(qc)
print(time.time() - start)