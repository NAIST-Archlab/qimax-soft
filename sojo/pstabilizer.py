import numpy as np
from numpy import sin, cos, sqrt


class Instructor:
    """List of instructors
    """
    def __init__(self, num_qubits):
        self.clusters = []
        self.cluster = []
        self.cluster_temp = []
        self.xcluster = []
        self.xcluster_temp = []
        self.xclusters = []
        self.instructors = []
        self.num_qubits = num_qubits
        self.barriers = [0] * self.num_qubits
        self.is_cx_first = False

    def append(self, gate, index, param=0):
        """Add an instructor to the list instructors

        Args:
            gate (_type_): _description_
            index (_type_): _description_
            param (int, optional): _description_. Defaults to 0.
        """
        self.instructors.append((gate, index, param))

    def clustering(self):
        """Construct clusters from the list of clusters and list of xclusters
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

                self.xcluster.append(self.instructors.pop(0))
            else:
                if self.barriers[index] == 0:
                    self.cluster.append(self.instructors.pop(0))
                else:
                    self.cluster_temp.append(self.instructors.pop(0))
            if is_break:
                if len(self.cluster) > 0:
                    self.clusters.append(self.cluster)
                self.instructors = self.cluster_temp + self.instructors
                if len(self.xcluster) > 0:
                    self.xclusters.append(self.xcluster)
                self.cluster = []
                self.cluster_temp = []
                self.xcluster = []
                self.barriers = [0] * self.num_qubits
                is_break = False
        if len(self.cluster) > 0:
            self.clusters.append(self.cluster)
        if len(self.cluster_temp) > 0:
            self.clusters.append(self.cluster_temp)
        if len(self.xcluster) > 0:
            self.xclusters.append(self.xcluster)
        return


def char_to_weight(character: str) -> np.ndarray:
    """X -> [0, 1, 0, 0] = 0*I + 1*X + 0*Y + 0*Z

    Args:
        character (str): _description_

    Returns:
        np.ndarray: _description_
    """
    if character == "x":
        return np.array([0, 1, 0, 0])
    if character == "y":
        return np.array([0, 0, 1, 0])
    if character == "z":
        return np.array([0, 0, 0, 1])
    if character == "i":
        return np.array([1, 0, 0, 0])


def char_to_index(character: str) -> int:
    """I,X,Y,Z -> 0,1,2,3

    Args:
        character (str): _description_

    Returns:
        int: _description_
    """
    if character == "i":
        return 0
    if character == "x":
        return 1
    if character == "y":
        return 2
    if character == "z":
        return 3


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




def index_to_word(index: int, num_qubits: int) -> str:
    """Convert a index to coressponding word
    Example: index 4, 2 (qubits) -> IX
    Example: index 255, 4 (qubits) -> ZZZZ
    Args:
        index (int): An integer from 0 to 4**num_qubits - 1
        num_qubits (int)

    Raises:
        ValueError: index out of bounds

    Returns:
        str: word
    """
    char_map = ['i', 'x', 'y', 'z']
    word = []
    if index < 0 or index >= 4**num_qubits:
        raise ValueError('Index out of bounds. must from 0 to 4**num_qubits - 1')
    for _ in range(num_qubits):
        word.append(char_map[index % 4])
        index //= 4
    return ''.join(reversed(word))

def construct_lut_noncx(grouped_instructorss, num_qubits: int):
    """grouped_instructorss has size k x n x [?], with k is number of non-cx layer, n is number of qubits,
    ? is the number of instructor (depend on each cluster).
    lut has size k x n x 4 x 4, with 4 is the number of Pauli, 4 for weights
    Args:
        grouped_instructorss (_type_): group by qubits
        num_qubits (int): _description_

    Returns:
        _type_: _description_
    """
    k = len(grouped_instructorss)
    lut = np.zeros((k, num_qubits, 4, 4))
    print(lut.shape)
    characters = ["i", "x", "y", "z"]
    for k in range(k):
        for j in range(num_qubits):
            for i in range(4):
                lut[k][j][i] = mapper_noncx(characters[i], grouped_instructorss[k][j])
    return lut

def instructor_to_lut(ins: Instructor):
    """First, diving instructors into k non-cx clusters and k+1/k-1/k cx-cluster,
    the, utilizing the lut (size k x n x 4 x 4)"""
    grouped_instructorss = group_instructorss_by_qubits(ins.clusters, ins.num_qubits)
    LUT = construct_lut_noncx(grouped_instructorss, ins.num_qubits)
    return LUT

def word_to_index(word: str) -> int:
    """Convert word to corresponding index in array of 4^n
    Example: IX -> 1*4^1 + 0*4^0 = 4
    Example: IIII -> 0, ZZZZ = 4^4 - 1 = 255
    Args:
        word (str): Pauliword

    Returns:
        int: index
    """
    index = 0
    for char in word:
        index = index * 4 + char_to_index(char)
    return index

def generate_pauli_combination(num_qubits: int) -> np.ndarray:
    """Generater 4^num_qubits Pauli combinations 
    for a given number of qubits.
    Args:
        num_qubits (int): _description_

    Returns:
        np.ndarray: _description_
    """
    combinations = []
    for i in range(0, 4**num_qubits):
        combinations.append(index_to_word(i, num_qubits	))
    return combinations

def weightss_to_lambda(weightss: np.ndarray, lambdas = None) -> np.ndarray:
    """A sum of transformed word (a matrix 4^n x n x 4) to list
    Example for a single transformed word (treated as 1 term): 
        [[1,2,3,4], [1,2,3,4]] 
            -> [1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12, 4, 8, 12, 16]
    Example for this function (sum, 2 qubits, 3 term):
        [[[1,2,3,4], [1,2,3,4]], [[1,2,3,4], [1,2,3,4]], [[1,2,3,4], [1,2,3,4]]] 
            -> [ 3.  6.  9. 12.  6. 12. 18. 24.  9. 18. 27. 36. 12. 24. 36. 48.]
    Args:
        weightss (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    if lambdas is None:
        lambdas = np.ones(weightss.shape[0])
    num_qubits = weightss.shape[1]
    new_lambdas = np.zeros((4**num_qubits))
    for j, weights in enumerate(weightss):
        combinations = np.array(np.meshgrid(*weights)).T.reshape(-1, len(weights))
        new_lambdas += lambdas[j]*np.prod(combinations, axis=1)
    return new_lambdas
