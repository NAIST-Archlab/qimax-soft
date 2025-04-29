
import pennylane as qml
import pandas as pd
import qiskit.quantum_info
num_qubitss = range(2, 16)
num_layers = 2
num_repeatss = [10, 100, 1000, 10000, 100000]

def benchmark_qml_pennylane(num_qubits, num_layers, num_repeats):
	import time

	start = time.time()
	dev = qml.device("lightning.gpu", wires=num_qubits)
	@qml.qnode(dev)
	def circuit():
		for k in range(num_layers):
			
			for i in range(num_qubits):
				qml.RX(1, wires=i)
				qml.RY(2, wires=i)
				qml.RZ(3, wires=i)
			for _ in range(num_repeats):
				for i in range(num_qubits - 1):
					qml.CNOT(wires=[i, i + 1])
				qml.CNOT(wires=[num_qubits - 1, 0])
		return qml.state()

	state = circuit()
	time_taken = time.time() - start
	return time_taken

benchmark_qml_pennylane(2, 2, 1000)

for num_qubits in num_qubitss:
	for num_repeats in num_repeatss:
		print(num_qubits, num_repeats)
		time_takens = []
		for _ in range(5):
			time_taken = benchmark_qml_pennylane(num_qubits, num_layers, num_repeats)
			time_takens.append(time_taken)
		time_taken = sum(time_takens)/5

		if 'results_df' not in locals():
			results_df = pd.DataFrame(columns=['num_qubits', 'num_repeats', 'time_taken'])

		# Append the current result to the DataFrame
		results_df = pd.concat([results_df, pd.DataFrame({'num_qubits': [num_qubits], 'num_repeats': [num_repeats], 'time_taken': [time_taken]})], ignore_index=True)
		results_df.to_csv('time_num_layers2_cxxyz_pennylane.csv', index=False)