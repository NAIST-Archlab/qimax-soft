from gqimax2.instructor import Instructor
import numpy as np

num_qubits = 4
num_layers = 10
ins = Instructor(num_qubits)
for k in range(num_layers):
	for i in range(num_qubits - 1):
		ins.append('cx', [i, i + 1])
	ins.append('cx', [num_qubits - 1, 0])
	for i in range(num_qubits):
		ins.append('rx', i, 1)
		ins.append('ry', i, 2)
		ins.append('rz', i, 3)
ins.run()