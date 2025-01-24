from sojo.pstabilizer import PStabilizer
from sojo.instructor import Instructor
import numpy as np
import qiskit
import qiskit.quantum_info as qi
import time

num_qubits = 4
num_layers = 1000

start = time.time()
ins = Instructor(num_qubits)
for k in range(num_layers):
	for i in range(num_qubits - 1):
		ins.append('cx', [i, i + 1])
	ins.append('cx', [num_qubits - 1, 0])
	for i in range(num_qubits):
		ins.append('rx', i, np.random.rand())
		ins.append('ry', i, np.random.rand())
		ins.append('rz', i, np.random.rand())

ins.operatoring()
ins.create_lut_noncx(save_npy = False)
print(time.time() - start)

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