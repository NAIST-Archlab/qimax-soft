import numpy as np

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