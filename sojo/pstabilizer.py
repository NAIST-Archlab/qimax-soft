import numpy as np
from numpy import sin, cos, sqrt


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


def construct_lut_noncx(instructorsss, num_qubits):
    """instructorss has size k x n x [?], with k is number of non-cx layer, n is number of qubits,
    ? is the number of instructor.
    lut has size k x n x 4 x 4, with 4 is the number of Pauli, 4 for weights
    Args:
        instructorsss (_type_): _description_
        num_qubits (_type_): _description_

    Returns:
        _type_: _description_
    """
    k = len(instructorsss)
    lut = np.zeros((k, num_qubits, 4, 4))
    print(lut.shape)
    characters = ["i", "x", "y", "z"]
    for k in range(k):
        for j in range(num_qubits):
            for i in range(4):
                lut[k][j][i] = mapper_noncx(characters[i], instructorsss[k][j])
    return lut


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
