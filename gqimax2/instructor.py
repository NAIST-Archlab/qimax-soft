

from collections import deque

import cupy as cp
from .utils import create_word_zj
from gqimax2.mapper import construct_lut_noncx, map_noncx, map_cx
class Instructor:
    """List of instructors
    """
    def __init__(self, num_qubits):
        self.operators = []
        self.xoperator_begin = []
        self.xoperators = []
        self.instructors = []
        self.orders = []
        self.num_qubits = num_qubits
        self.is_cx_first = False
        self.lut = None
        self.lut_indicesss = None # Support lut
        self.lambdass = [
            cp.array([1], dtype=cp.float32) 
            for _ in range(num_qubits)
        ]

        self.indicesss = [
            cp.array([create_word_zj(num_qubits,j)])
            for j in range(num_qubits)
        ]


    def run(self):
        import time
        start1 = time.time()
        self.operatoring()
        # print('operating ...:', time.time() - start1)
        # print("Operatoring finished!")
        
        for j, order in enumerate(self.orders):
            k = j // 2
            if order == 0:
                start1 = time.time()
                self.lambdass, self.indicesss = map_noncx(self.lambdass, self.indicesss, self.lut[k], self.lut_indicesss[k])
                # print('map noncx ...:', time.time() - start1)
            else:
                start1 = time.time()
                for _, cnot_indices, _ in self.xoperators[k]:
                    self.lambdass, self.indicesss = map_cx(self.lambdass, self.indicesss, cnot_indices[0], cnot_indices[1])  
                # print('map cx ...:', time.time() - start1)
        return
    

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
        self.orders = create_zip_chain(len(self.operators), len(self.xoperators), self.is_cx_first)
        import time
        start1 = time.time()
        self.to_lut()
        # print('to_lut ...:', time.time() - start1)
        return

    def to_lut(self):
        r"""
            First, diving instructors into K non-cx operators and K+1/K-1/K cx-operator,
        
            Noneed LUT for cx-operators, utilizing the non-cx LUT (size K x n x 3 x 4)
        """
        # Thanks to the updated operatoring method, the operators
        # are already grouped by qubits, no need to group them again
        # operators (ragged tensor): K x n x ?, each element is an tuple (gate, index, param)
        self.lut, self.lut_indicesss = construct_lut_noncx(self.operators, self.num_qubits)
        return
    
def create_zip_chain(num_operators, num_xoperators, is_cx_first):
    total = num_operators + num_xoperators
    result = [0] * total
    step = 2
    start = 0 if is_cx_first else 1
    for i in range(start, min(total, num_xoperators * 2), step):
        result[i] = 1
    return result