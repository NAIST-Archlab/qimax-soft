from numpy import sqrt, cos, sin
import numpy as np

def pauli_to_weight(character: str) -> np.ndarray:
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
    weights = pauli_to_weight(character)
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

char = 'x'
instructor = [
    ('h', 0, 0),
    ('rx', 0, np.pi/2),
    ('ry', 0, np.pi/3),
    ('rz', 0, np.pi/4)
]


print(mapper_noncx(char, instructor))