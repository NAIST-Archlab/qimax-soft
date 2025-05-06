from gqimax2.instructor import Instructor
import numpy as np


for num_qubits in range(2, 16):
	num_layers = 2
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

	lenght = []
	for lambdas in ins.lambdass:
		lenght.append(len(lambdas))
	print(f"{num_qubits} qubits, 2 layers: ", np.average(lenght))