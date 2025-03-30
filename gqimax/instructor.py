import numpy as np
import cupy as cp
from numpy import sin, cos, sqrt
from .utils import index_to_word, char_to_weight, create_zip_chain
from .mapper import construct_lut_noncx
class Instructor:
    """List of instructors
    """
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
        self.is_cx_first = False
        self.orders = []
        self.LUT = None

    def append(self, gate, index, param=0):
        """Add an instructor to the list instructors

        Args:
            gate (_type_): _description_
            index (_type_): _description_
            param (int, optional): _description_. Defaults to 0.
        """
        self.instructors.append((gate, index, param))
    def create_lut_noncx(self, save_npy = False):
        self.LUT = instructor_to_lut(self)
        if save_npy:
            np.save("LUT.npy", self.LUT)
    
    def operatoring(self):
        """Construct operators from the list of operators and list of xoperators
        """
        if self.instructors[0][0] == "cx":
            self.is_cx_first = True
        self.barriers = [0] * self.num_qubits
        while len(self.instructors) > 0:
            gate, index, _ = self.instructors[0]

            is_break = False
            if gate == "cx":
                self.barriers[index[0]] += 1
                self.barriers[index[1]] += 1
                if sum(self.barriers) >= self.num_qubits and np.all(self.barriers):
                    if len(self.instructors) > 1:
                        if self.instructors[1][0] != "cx":
                            is_break = True
                _, index, _ = self.instructors.pop(0)
                self.xoperator.append(index)
            else:
                if self.barriers[index] == 0:
                    self.operator.append(self.instructors.pop(0))
                else:
                    self.operator_temp.append(self.instructors.pop(0))
            if is_break:
                if len(self.operator) > 0:
                    self.operators.append(self.operator)
                self.instructors = self.operator_temp + self.instructors
                if len(self.xoperator) > 0:
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
        self.orders = create_zip_chain(len(self.operators), len(self.xoperators), self.is_cx_first)
        return

def instructor_to_lut(ins: Instructor):
    """First, diving instructors into k non-cx operators and k+1/k-1/k cx-operator,
    the, utilizing the lut (size k x n x 4 x 4)"""
    grouped_instructorss = group_instructorss_by_qubits(ins.operators, ins.num_qubits)
    LUT = construct_lut_noncx(grouped_instructorss, ins.num_qubits)
    return LUT

def group_instructorss_by_qubits(instructors: list, num_qubits: int) -> list:
    """Group instructors by qubits
    Example: [['h', 0, 0], ['rx', 1, 0], ['h', 1, 0], ['ry', 0, 0]]
    -> [[['h', 0, 0], ['ry', 0, 0]], [['h', 1, 0], ['rx', 1, 0]]]

    Args:
        instructors (list): list of instructors
        num_qubits (int)

    Returns:
        list of list of n instructors: _description_
    """
    grouped_instructors = []
    for sublist in instructors:
        groups = {i: [] for i in range(num_qubits)}
        for instructor in sublist:
            index = instructor[1]
            groups[index].append(instructor)
        grouped_instructors.append([groups[i] for i in range(num_qubits)])
    return grouped_instructors

