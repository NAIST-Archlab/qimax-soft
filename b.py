import numpy as np
from collections import deque
class Instructor:
    """List of instructors
    """
    def __init__(self, num_qubits):
        self.operators = []
        self.operator = []
        self.xoperator_begin = []
        self.xoperators = []
        self.instructors = []
        self.num_qubits = num_qubits
        self.is_cx_first = False
        self.orders = []


    def append(self, gate, index, param=0):
        """Add an instructor to the list instructors

        Args:
            gate (_type_): _description_
            index (_type_): _description_
            param (int, optional): _description_. Defaults to 0.
        """
        self.instructors.append((gate, index, param))


    def operatoring(self):
        instructors = deque(self.instructors)
        
        self.is_cx_first = instructors[0][0] == "cx" if instructors else False
        if self.is_cx_first:
            self.xoperator_begin.append([])
            while instructors and instructors[0][0] == "cx":
                self.xoperator_begin[0].append(instructors.popleft())
        
        self.xbarriers = [0] * self.num_qubits
        self.barriers = [0] * self.num_qubits
        
        while instructors:
            gate, index, param = instructors.popleft()
            if gate == 'cx':
                location = max(self.barriers[index[0]], self.barriers[index[1]])
                if location >= len(self.xoperators):
                    self.xoperators.append([(gate, index, param)])
                else:
                    self.xoperators[location].append((gate, index, param))
                if self.barriers[index[0]] >= self.xbarriers[index[0]]:
                    self.xbarriers[index[0]] += 1
                if self.barriers[index[1]] >= self.xbarriers[index[1]]:
                    self.xbarriers[index[1]] += 1
            else:
                location = self.xbarriers[index]
                if location >= len(self.operators):
                    self.operators.append([[] for _ in range(self.num_qubits)])
                self.operators[location][index].append((gate, index, param))
                if self.xbarriers[index] > self.barriers[index]:
                    self.barriers[index] += 1
        
        if self.is_cx_first:
            self.xoperators = self.xoperator_begin + self.xoperators
        self.instructors = list(instructors) 

# ins = Instructor(3)
# ins.append('rx', 0, 0.5)
# ins.append('cx', [1,2], 0)
# ins.append('rx', 0, 0.5)
# ins.append('rx', 1, 0)
# ins.append('rx', 2, 0)
# ins.append('cx', [1,2], 0)
# ins.operatoring()


ins = Instructor(3)
ins.append('cx', [0, 1])
ins.append('rx', 0, 0.5)
ins.append('rx', 1, 0.5)
ins.append('cx', [0, 1])
ins.append('cx', [0, 1])
ins.append('cx', [1, 2])
ins.append('rx', 0, 0.5)
ins.append('cx', [0, 1])
ins.operatoring()

# ins = Instructor(4)
# ins.append("cx", [0, 1])
# ins.append("h", 0)
# ins.append("rx", 1, 0.78)
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
# ins.append("cx", [2,3])



# print(len(ins.operators))
# print(len(ins.xoperators))
print(ins.operators)
print(ins.xoperators)