import cupy as cp

def index_to_word(index: int, num_qubits: int) -> str:
    """Convert a index to coressponding word
    Example: index 4, 2 (qubits) -> IX
    Example: index 255, 4 (qubits) -> ZZZZ
    """
    char_map = ['i', 'x', 'y', 'z']
    word = []
    for _ in range(num_qubits):
        word.append(char_map[index % 4])
        index //= 4
    return ''.join(reversed(word))

def index_to_indices(index: int, num_qubits: int) -> cp.ndarray:
    """Convert a index to coressponding word
    Example: index 4, 2 (qubits) -> IX --> [0, 1]
    Example: index 255, 4 (qubits) -> ZZZZ --> [3, 3, 3, 3]
    """
    words = index_to_word(index, num_qubits)
    indices = []
    for word in words:
        if word == "i":
            indices.append(0)
        elif word == "x":
            indices.append(1)
        elif word == "y":
            indices.append(2)
        elif word == "z":
            indices.append(3)
    
    return cp.array(indices)
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

def generate_pauli_combination(num_qubits: int) -> cp.ndarray:
    """Generater 4^num_qubits Pauli combinations 
    for a given number of qubits.
    Args:
        num_qubits (int): _description_

    Returns:
        cp.ndarray: _description_
    """
    combinations = []
    for i in range(0, 4**num_qubits):
        combinations.append(index_to_word(i, num_qubits	))
    return combinations

def char_to_weight(character: str) -> cp.ndarray:
    """X -> [0, 1, 0, 0] = 0*I + 1*X + 0*Y + 0*Z
    """
    if character == "i":
        return cp.array([1, 0, 0, 0], dtype=cp.float32)
    elif character == "x":
        return cp.array([0, 1, 0, 0], dtype=cp.float32)
    elif character == "y":
        return cp.array([0, 0, 1, 0], dtype=cp.float32)
    elif character == "z":
        return cp.array([0, 0, 0, 1], dtype=cp.float32)
    
def create_word_zj(j, num_qubits):
    if j < 0 or j >= num_qubits:
        raise ValueError('j out of bounds. must from 0 to num_qubits - 1')
    indices = [0] * num_qubits
    indices[j] = 3
    return cp.array(indices)






"""
Constants and functions for PauliComposer and PauliDecomposer classes.

See: https://arxiv.org/abs/2301.00560
"""

from numbers import Real


# Definition of some useful constants
PAULI_LABELS = ['I', 'X', 'Y', 'Z']
NUM2LABEL = {ind: PAULI_LABELS[ind] for ind in range(len(PAULI_LABELS))}

PAULI = {'I': cp.eye(2, dtype=cp.uint8),
         'X': cp.array([[0, 1], [1, 0]], dtype=cp.uint8),
         'Y': cp.array([[0, -1j], [1j, 0]], dtype=cp.complex64),
         'Z': cp.array([[1, 0], [0, -1]], dtype=cp.int8)}


def nbytes(size: int, n_items: int) -> Real:
    """Return number of bytes needed for a `n_items`-array of `size` bits."""
    # Bits/element * number of elements / 8 bits/byte
    n_bytes = size * n_items / 8
    return int(n_bytes) if n_bytes.is_integer() else n_bytes


def convert_bytes(n_bytes: Real) -> str:
    """Convert a number of bytes `n_bytes` into a manipulable quantity."""
    for unit in ['iB', 'kiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB']:
        if n_bytes < 1024:
            return '%4.2f %s' % (n_bytes, unit)
        n_bytes /= 1024
    return '%4.2f YiB' % (n_bytes)

