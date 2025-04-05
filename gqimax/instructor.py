

from collections import deque
from .mapper import construct_lut_noncx
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
        self.lut = None


    def append(self, gate, index, param=0):
        """Add an instructor to the list instructors

        Args:
            gate (_type_): _description_
            index (_type_): _description_
            param (int, optional): _description_. Defaults to 0.
        """
        self.instructors.append((gate, index, param))
    def operatoring(self):
        if self.instructors[0][0] == "cx":
            self.is_cx_first = True
        else:
            self.is_cx_first = False
        if self.is_cx_first:
            self.xoperator_begin.append([])
            while(True):
                gate, index, param = self.instructors.pop(0)
                self.xoperator_begin[0].append((gate, index, param))
                if len(self.instructors) == 0 or self.instructors[0][0] != "cx":
                    break
        self.xbarriers = [0] * self.num_qubits
        self.barriers = [0] * self.num_qubits
        for (gate, index, param) in self.instructors:
            if gate == 'cx':
                location = max(self.barriers[index[0]], self.barriers[index[1]])
                ###### --- Append to the ragged matrix of xoperators --- #####
                if location >= len(self.xoperators):
                    self.xoperators.append([(gate, index, param)])
                else:
                    self.xoperators[location].append((gate, index, param))
                ##############################################################
                if self.barriers[index[0]] >= self.xbarriers[index[0]]:
                    self.xbarriers[index[0]] += 1
                if self.barriers[index[1]] >= self.xbarriers[index[1]]:
                    self.xbarriers[index[1]] += 1
            else:
                location = self.xbarriers[index]
                ###### --- Append to the ragged matrix of operators --- ######
                if location >= len(self.operators):
                    self.operators.append([[] for _ in range(self.num_qubits)])
                self.operators[location][index].append((gate, index, param))
                ##############################################################
                if self.xbarriers[index] > self.barriers[index]:
                    self.barriers[index] += 1
  
        if self.is_cx_first:
            self.xoperators = self.xoperator_begin + self.xoperators
        return

    def instructor_to_lut(self):
        """First, diving instructors into K non-cx operators and K+1/K-1/K cx-operator,
        Noneed LUT for cx-operators
        , utilizing the non-cx LUT (size K x n x 3 x 4)"""
        # Thanks to the updated operatoring method, the operators
        # are already grouped by qubits, no need to group them again
        # operators (ragged tensor): K x n x ?, each element is an tuple (gate, index, param)
        self.lut = construct_lut_noncx(self.operators, self.num_qubits)
        returnxn

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

