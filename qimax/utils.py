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
        index = index * 4 + pauli_to_index(char)
    return index

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


def pauli_to_index(character: str) -> int:
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

def create_word_zj(j, num_qubits):
    if j < 0 or j >= num_qubits:
        raise ValueError('j out of bounds. must from 0 to num_qubits - 1')
    return "i" * j + "z" + "i" * (num_qubits - j - 1)

def create_zip_chain(num_operators, num_xoperators, is_cx_first):
    """Create list 0,1,0,1,...
    If is_cx_first is True, then 1 is first, else 0 is first
    Args:
        n (_type_): _description_
        m (_type_): _description_
        is_cx_first (bool): _description_

    Returns:
        _type_: _description_
    """
    result = []
    while num_operators > 0 or num_xoperators > 0:
        if is_cx_first:
            if num_xoperators > 0:
                result.append(1)
                num_xoperators -= 1
            if num_operators > 0:
                result.append(0)
                num_operators -= 1
        else:   
            if num_operators > 0:
                result.append(0)
                num_operators -= 1
            if num_xoperators > 0:
                result.append(1)
                num_xoperators -= 1
    return result


"""
Constants and functions for PauliComposer and PauliDecomposer classes.

See: https://arxiv.org/abs/2301.00560
"""

import numpy as np
from numbers import Real


# Definition of some useful constants
PAULI_LABELS = ['I', 'X', 'Y', 'Z']
NUM2LABEL = {ind: PAULI_LABELS[ind] for ind in range(len(PAULI_LABELS))}

PAULI = {'I': np.eye(2, dtype=np.uint8),
         'X': np.array([[0, 1], [1, 0]], dtype=np.uint8),
         'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex64),
         'Z': np.array([[1, 0], [0, -1]], dtype=np.int8)}


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