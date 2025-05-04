from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import GraphState
import qiskit
import networkx as nx
from qimax.stabilizer import StabilizerGenerator, PauliTerm, PauliWord
import numpy as np
def create_circuit(num_qubits: int, degree: int = 2) -> QuantumCircuit:
    """Returns a quantum circuit implementing a graph state.

    Arguments:
        num_qubits: number of qubits of the returned quantum circuit
        degree: number of edges per node
    """
    q = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(q, name="graphstate")

    g = nx.random_regular_graph(degree, num_qubits)
    a = nx.convert_matrix.to_numpy_array(g)
    qc.compose(GraphState(a), inplace=True)
    return qc.decompose(gates_to_decompose="graph_state")


import time

times = []
for num_qubits in range(2, 1000):
	qc = create_circuit(num_qubits)
	qc_trans = qiskit.transpile(qc, basis_gates = ['h', 's', 'cx', 'rx', 'ry', 'rz'], optimization_level=3)
	gate_list = []
	for instruction in qc_trans.data:
		gate = instruction.operation.name
		wires = [qubit._index for qubit in instruction.qubits]
		params = instruction.operation.params
		gate_list.append((gate, wires, params))
	start = time.time()
	stb = StabilizerGenerator(num_qubits)
	for gate in gate_list:
		if gate[0] in ['h', 's', 'cx']:
			if len(gate[1]) == 2:
				stb.map(gate[0], gate[1])
			else:
				stb.map(gate[0], gate[1][0])
		else:
			stb.map(gate[0], gate[1], gate[2])
	end = time.time()
	print(f"Num qubits: {num_qubits}, Time taken: {end - start:.4f} seconds")
	times.append(end - start)

np.savetxt("graph_times_qimaxv1.txt", times)
