from .instructor import Instructor
import numpy as np
def sample1(num_qubits):
    ins = Instructor(num_qubits)
    ins.append("h", 0)
    ins.append("rx", 1, 0.78)
    # ins.append("h", 2)
    # ins.append("h", 0)
    # ins.append("cx", [0, 1])
    # ins.append("h", 2)
    # ins.append("h", 2)
    # ins.append("ry", 0, 0.56)
    # ins.append("cx", [1, 2])
    # ins.append("h", 1)
    # ins.append("h", 3)
    # ins.append("h", 3)
    # ins.append("h", 3)
    # ins.append("h", 3)
    # ins.append("h", 0)
    # ins.append("h", 1)
    # ins.append("h", 2)
    # ins.append("h", 3)
    # ins.append("h", 0)
    # ins.append("h", 2)
    # ins.append("cx", [1, 3])
    return ins

def sample3():
    ins = Instructor(2)
    ins.append('ry',0, np.pi/3)
    ins.append('rx',0, np.pi/2)
    ins.append('cx', [0,1])
    ins.append('t', 0)
    ins.append('cx',[0,1])
    ins.append('h',0)
    ins.append('rz', 0, np.pi/4)
    ins.append('rx',1, np.pi/3)
    return ins
def sample2(num_qubits, num_layers):
    ins = Instructor(num_qubits)
    for k in range(num_layers):
        for i in range(num_qubits - 1):
            ins.append('cx', [i, i + 1])
        ins.append('cx', [num_qubits - 1, 0])
        for i in range(num_qubits):
            ins.append('rx', i, 1)
            ins.append('ry', i, 2)
            ins.append('rz', i, 3)

    return ins

