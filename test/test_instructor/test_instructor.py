class Instructor:
    def __init__(self, num_qubits):
        self.operators = []
        self.operator = []
        self.operator_temp = []
        self.xoperator = []
        self.xoperators = []
        self.instructors = []
        self.num_qubits= num_qubits
        self.barriers = [0]*num_qubits

    def append(self, gate, index, param = 0):
        self.instructors.append([gate, index, param])
    def operatoring(self):
        while(len(self.instructors)>0):
            gate, index, _ = self.instructors[0]
            sum_barrier = sum(self.barriers)
            if  sum_barrier >= self.num_qubits:
                self.operators.append(self.operator)
                self.operator = []
                self.instructors = self.operator_temp + self.instructors
                self.barriers = [0]*self.num_qubits
            if gate == 'cx':
                self.barriers[index[0]] += 1
                self.barriers[index[1]] += 1
                self.xoperator.append(self.instructors[0])
                self.instructors.pop(0)
            else:
                if len(self.xoperator) > 0 and sum_barrier >= self.num_qubits:
                    self.xoperators.append(self.xoperator)
                    self.xoperator = []
                if self.barriers[index] == 0:
                    self.operator.append(self.instructors[0])
                    self.instructors.pop(0)
                else:
                    self.operator_temp.append(self.instructors.pop(0))
ins = Instructor(3)
ins.append('h', 0)
ins.append('h', 1)
ins.append('h', 2)
ins.append('h', 0)
ins.append('cx', [0,1])
ins.append('h', 2)
ins.append('h', 2)
ins.append('h', 0)
ins.append('cx', [1,2])
ins.append('h', 1)
ins.operatoring()