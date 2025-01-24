from sojo.pstabilizer import PStabilizer
from sojo.instructor import Instructor
from sojo.stabilizer import StabilizerGenerator
import numpy as np
import qiskit
import qiskit.quantum_info as qi
import time

num_qubits = 3
num_layers = 1000
avg_times = 10

times_lut = []
times_mapping = []
times_mp = []
times_qiskit = []

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
print("Create LUT", time.time() - start)


start = time.time()
pstabilizer = PStabilizer(0, num_qubits)
pstabilizer.map(ins)
print("Mapping", time.time() - start)


stb = StabilizerGenerator(num_qubits)
for i in range(num_layers):
    for i in range(num_qubits - 1):
        stb.cx([i, i + 1])
    stb.cx([num_qubits - 1, 0])
    for i in range(num_qubits):
        stb.rx(np.random.rand(), i)
        stb.ry(np.random.rand(), i)
        stb.rz(np.random.rand(), i)
start = time.time()
dm = stb.generate_density_matrix_by_generator_naive()
print("Generate matrix", time.time() - start)








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
print("Qiskit", time.time() - start)