from qimax.stabilizer import StabilizerGenerator
import pandas as pd
num_qubitss = range(2, 16)
num_layers = 2
num_repeatss = [10, 100, 1000, 10000, 100000]

def benchmark_qml_gqimax(num_qubits, num_layers, num_repeats):
	import time

	start = time.time()
	stb = StabilizerGenerator(num_qubits)
	for k in range(num_layers):
		for _ in range(num_repeats):
			for i in range(num_qubits):
				stb.map('rx', i, 1)
				stb.map('ry', i, 2)
				stb.map('rz', i, 3)
		for i in range(num_qubits - 1):
			stb.map('cx', [i, i + 1])
		stb.map('cx', [num_qubits - 1, 0])
	time_taken = time.time() - start
	return time_taken

benchmark_qml_gqimax(2, 2, 1000)

for num_qubits in num_qubitss:
	for num_repeats in num_repeatss:
		print(num_qubits, num_repeats)
		time_takens = []
		for _ in range(10):
			time_taken = benchmark_qml_gqimax(num_qubits, num_layers, num_repeats)
			time_takens.append(time_taken)
		time_taken = sum(time_takens)/10
		if 'results_df' not in locals():
			results_df = pd.DataFrame(columns=['num_qubits', 'num_repeats', 'time_taken'])

		# Append the current result to the DataFrame
		results_df = pd.concat([results_df, pd.DataFrame({'num_qubits': [num_qubits], 'num_repeats': [num_repeats], 'time_taken': [time_taken]})], ignore_index=True)
		results_df.to_csv('time_num_layers2_xyzcx_qimaxv1.csv', index=False)