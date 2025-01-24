from sojo.pstabilizer import PStabilizer
from sojo.instructor import Instructor
from sojo.stabilizer import StabilizerGenerator
import numpy as np
import qiskit
import qiskit.quantum_info as qi
import time



times_lut = []
times_mapping = []
times_mp = []
times_qiskit = []
num_layers = 50
def benchmark(num_qubits, num_repeat, avg_times):
    for _ in range(avg_times):
        ############################
        ##### -Create LUT -- #####
        ############################
        start = time.time()
        ins = Instructor(num_qubits)
        for k in range(num_layers):
            for i in range(num_qubits - 1):
                ins.append('cx', [i, i + 1])
            ins.append('cx', [num_qubits - 1, 0])
            for i in range(num_qubits):
                for _ in range(num_repeat):
                    ins.append('rx', i, np.random.rand())
                    ins.append('ry', i, np.random.rand())
                    ins.append('rz', i, np.random.rand())

        ins.operatoring()
        ins.create_lut_noncx(save_npy = False)
        times_lut.append(time.time() - start)
        ############################
        ##### -Mapping -- #####
        ############################

        start = time.time()
        pstabilizer = PStabilizer(0, num_qubits)
        pstabilizer.map(ins)
        times_mapping.append(time.time() - start)
        ############################
        # Matrix converting and producting #
        ############################

        stb = StabilizerGenerator(num_qubits)
        for i in range(num_layers):
            for i in range(num_qubits - 1):
                stb.cx([i, i + 1])
            stb.cx([num_qubits - 1, 0])
            for i in range(num_qubits):
                for _ in range(num_repeat):
                    stb.rx(np.random.rand(), i)
                    stb.ry(np.random.rand(), i)
                    stb.rz(np.random.rand(), i)
        start = time.time()
        dm = stb.generate_density_matrix_by_generator_naive()
        times_mp.append(time.time() - start)
        ############################
        ##### ---- Qiskit ---- #####
        ############################
        start = time.time()
        qc = qiskit.QuantumCircuit(num_qubits)
        for k in range(num_layers):
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
            qc.cx(num_qubits - 1, 0)
            for i in range(num_qubits):
                for _ in range(num_repeat):
                    qc.rx(np.random.rand(), i)
                    qc.ry(np.random.rand(), i)
                    qc.rz(np.random.rand(), i)

        dm = qi.DensityMatrix(qc)
        times_qiskit.append(time.time() - start)
    print("Num qubits", num_qubits, "Num repeats", num_repeat)
    print("LUT", np.mean(times_lut), np.std(times_lut))
    print("Mapping", np.mean(times_mapping), np.std(times_mapping))
    print("Matrix product", np.mean(times_mp), np.std(times_mp))
    print("Qiskit", np.mean(times_qiskit), np.std(times_qiskit))

    np.savetxt(f"./data/times_lut_{num_qubits}qubit_{num_repeat}repeatlayer.csv", times_lut, delimiter=",")
    np.savetxt(f"./data/times_mapping_{num_qubits}qubit_{num_repeat}repeatlayer.csv", times_mapping, delimiter=",")
    np.savetxt(f"./data/times_mp_{num_qubits}qubit_{num_repeat}repeatlayer.csv", times_mp, delimiter=",")
    np.savetxt(f"./data/times_qiskit_{num_qubits}qubit_{num_repeat}repeatlayer.csv", times_qiskit, delimiter=",")

avg_times = 10
qubits = range(2, 8)
for num_qubits in qubits:
    for num_repeat in range(10, 110, 10):
        benchmark(num_qubits, num_repeat, avg_times)