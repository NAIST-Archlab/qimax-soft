from qimax.stabilizer import StabilizerGenerator, PauliTerm, PauliWord
import numpy as np

def ghz_state(num_qubits):
	stb = StabilizerGenerator(num_qubits)
	stb.h(0)
	for i in range(0, num_qubits - 1):
		stb.cx([i, i + 1])
	return stb.stabilizers

import time

times = []
for num_qubits in range(2, 1000):
	start = time.time()
	stb = ghz_state(num_qubits)
	end = time.time()
	print(f"Num qubits: {num_qubits}, Time taken: {end - start:.4f} seconds")
	times.append(end - start)

np.savetxt("ghz_times_qimaxv1.txt", times)