import numpy as np
class Instructor:
    def __init__(self, num_qubits):
        self.operators = []
        self.operator = []
        self.operator_temp = []
        self.xoperator = []
        self.xoperator_temp = []
        self.xoperators = []
        self.instructors = []
        self.num_qubits = num_qubits
        self.barriers = [0] * self.num_qubits

    def append(self, gate, index, param=0):
        self.instructors.append([gate, index, param])

    def operatoring(self):
        self.barriers = [0] * self.num_qubits
        while len(self.instructors) > 0:
            gate, index, _ = self.instructors[0]
            
            is_break = False
            if gate == "cx":
                self.barriers[index[0]] += 1
                self.barriers[index[1]] += 1
                if sum(self.barriers) >= self.num_qubits and np.all(self.barriers) and len(self.operator) > 0:
                    self.operator_temp.append(self.instructors.pop(0))
                    is_break = True
                else:
                    self.xoperator.append(self.instructors.pop(0))
            else:
                if self.barriers[index] == 0:
                    self.operator.append(self.instructors.pop(0))
                else:
                    self.operator_temp.append(self.instructors.pop(0))
            if is_break:
                self.operators.append(self.operator)
                self.instructors =  self.operator_temp + self.instructors
                self.xoperators.append(self.xoperator)
                self.operator = []
                self.operator_temp = []
                self.xoperator = []
                self.barriers = [0] * self.num_qubits
                is_break = False
        if len(self.operator) > 0:
            self.operators.append(self.operator)
        if len(self.operator_temp) > 0:
            self.operators.append(self.operator_temp)
        if len(self.xoperator) > 0:
            self.xoperators.append(self.xoperator)
        return 
num_qubits = 3
ins = Instructor(num_qubits)
for i in range(num_qubits - 1):
    ins.append('cx', [i, i + 1])
ins.append('cx', [num_qubits - 1, 0])
for i in range(num_qubits):
    ins.append('h', i)
for i in range(num_qubits - 1):
    ins.append('cx', [i, i + 1])
ins.append('cx', [num_qubits - 1, 0])
for i in range(num_qubits):
    ins.append('h', i)
ins.operatoring()