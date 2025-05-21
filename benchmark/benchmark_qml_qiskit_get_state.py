import qiskit
import pennylane as qml
import pandas as pd
import qiskit
from qiskit_aer import Aer
num_qubitss = range(2, 16)
num_layers = 2
num_repeatss = [10, 100, 1000, 10000, 100000]
import time
def benchmark_qml_qiskit(num_qubits, num_layers):
    start = time.time()
    qc = qiskit.QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(1, i, )
        qc.ry(2, i)
        qc.rz(3, i)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    qc.cx(num_qubits - 1, 0)
    qc.save_statevector()
    simulator = Aer.get_backend('aer_simulator_statevector', device = 'GPU')
    qc = qiskit.transpile(qc, simulator)
    result = simulator.run(qc).result()
    statevector = result.get_statevector(qc)
    print(statevector)

benchmark_qml_qiskit(2, 2)
