from gqimax2.instructor import Instructor
import pandas as pd
num_qubitss = range(1, 11)
num_layers = 2
num_repeatss = [10, 100, 1000, 10000, 100000]

def benchmark_qml_gqimax(num_qubits, num_layers, num_repeats):
	import time

	start = time.time()
	ins = Instructor(num_qubits)
	for k in range(num_layers):
		for _ in range(num_repeats):
			for i in range(num_qubits):
				ins.append('rx', i, 1)
				ins.append('ry', i, 2)
				ins.append('rz', i, 3)
		for i in range(num_qubits - 1):
			ins.append('cx', [i, i + 1])
		ins.append('cx', [num_qubits - 1, 0])
	ins.run()
	time_taken = time.time() - start
	return time_taken

for num_qubits in num_qubitss:
	for num_repeats in num_repeatss:
		time_taken = benchmark_qml_gqimax(num_qubits, num_layers, num_repeats)

		if 'results_df' not in locals():
			results_df = pd.DataFrame(columns=['num_qubits', 'num_repeats', 'time_taken'])

		# Append the current result to the DataFrame
		results_df = pd.concat([results_df, pd.DataFrame({'num_qubits': [num_qubits], 'num_repeats': [num_repeats], 'time_taken': [time_taken]})], ignore_index=True)
results_df.to_csv('time_num_layers2_xyzcx.csv', index=False)