import numpy as np
from numpy import sin, cos, sqrt
from .utils import index_to_word, char_to_weight, create_zip_chain

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


def mapper_noncx(character: str, instructors: list) -> np.ndarray:
    """Map a single Pauliword to list by multiple instructors
    Related to construct_LUT_noncx.
    Example: X -> [0, 1, 0, 0] -- h --> [0, 0, -1, 0] = -Y
    Args:
        character (str): I, X, Y or Z
        instructors (list)
    Returns:
        np.ndarray
    """
    weights = char_to_weight(character)
    for gate, _, param in instructors:
        I, A, B, C = weights
        if gate == "h":
            weights = np.array([I, C, -B, A])
        if gate == "s":
            weights = np.array([I, -B, A, C])
        if gate == "t":
            weights = np.array([I, (A - B) / sqrt(2), (A + B) / sqrt(2), C])
        if gate == "rx":
            weights = np.array(
                [I, A, B * cos(param) - C * sin(param), B * sin(param) + C * cos(param)]
            )
        if gate == "ry":
            weights = np.array(
                [I, A * cos(param) + C * sin(param), B, C * cos(param) - A * sin(param)]
            )
        if gate == "rz":
            weights = np.array(
                [I, A * cos(param) - B * sin(param), B * cos(param) + A * sin(param), C]
            )
    return weights




def construct_lut_noncx(grouped_instructorss, num_qubits: int):
    """grouped_instructorss has size k x n x [?], with k is number of non-cx layer, n is number of qubits,
    ? is the number of instructor (depend on each operator).
    lut has size k x n x 3 x 4, with 3 is the number of Pauli (ignore I), 4 for weights
    Args:
        grouped_instructorss (_type_): group by qubits
        num_qubits (int): _description_

    Returns:
        _type_: _description_
    """
    k = len(grouped_instructorss)
    lut = np.zeros((k, num_qubits, 3, 4))
    characters = ["x", "y", "z"] # Ignore I because [?]I = I
    for k in range(k):
        for j in range(num_qubits):
            for i in range(3):
                lut[k][j][i] = mapper_noncx(characters[i], grouped_instructorss[k][j])
    return lut

def instructor_to_lut(ins: Instructor):
    """First, diving instructors into k non-cx operators and k+1/k-1/k cx-operator,
    the, utilizing the lut (size k x n x 4 x 4)"""
    grouped_instructorss = group_instructorss_by_qubits(ins.operators, ins.num_qubits)
    LUT = construct_lut_noncx(grouped_instructorss, ins.num_qubits)
    return LUT

def weightss_to_lambda(weightss: np.ndarray, lambdas) -> np.ndarray:
    """A sum of transformed word (a matrix 4^n x n x 4) to list
    Example for a single transformed word (treated as 1 term): 
        k*[[1,2,3,4], [1,2,3,4]] 
            -> k*[1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12, 4, 8, 12, 16]
    Example for this function (sum, 2 qubits, 3 term):
        [[[1,2,3,4], [1,2,3,4]], [[1,2,3,4], [1,2,3,4]], [[1,2,3,4], [1,2,3,4]]] 
            -> [ 3.  6.  9. 12.  6. 12. 18. 24.  9. 18. 27. 36. 12. 24. 36. 48.]
    Args:
        weightss (np.ndarray): _description_

    Returns:
        np.ndarray: lambdas
    """
    num_qubits = weightss.shape[1]
    new_lambdas = np.zeros((4**num_qubits))
    for j, weights in enumerate(weightss):
        combinations = np.array(np.meshgrid(*weights)).T.reshape(-1, len(weights))
        new_lambdas += lambdas[j]*np.prod(combinations, axis=1)
    # This lambdas is still in the form of 4^n x 1, 
    # we need to ignore 0 values in the next steps
    # In the worst case, there is no 0 values.
    return new_lambdas