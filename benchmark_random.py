from sojo.stabilizer import StabilizerGenerator
import numpy as np
import qiskit
import qiskit.quantum_info as qi
import time

num_qubits = 6
num_layers = 1000
start = time.time()
qc = qiskit.QuantumCircuit(num_qubits)
for k in range(num_layers):
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    qc.cx(num_qubits - 1, 0)
    for i in range(num_qubits):
        qc.rx(np.random.rand(), i)
        qc.ry(np.random.rand(), i)
        qc.rz(np.random.rand(), i)

dm = qi.DensityMatrix(qc)
print(time.time() - start)