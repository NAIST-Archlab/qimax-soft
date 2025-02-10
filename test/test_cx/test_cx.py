

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


def map_cx(index, control, target, num_qubits):  
    word = list(index_to_word(index, num_qubits))
    original = word.copy()
    char_control = word[control]
    char_target = word[target]
    lambdas = 1
    if char_control == 'i':
        if char_target == 'y':
            word[control] = 'z'
            word[target] = 'y'
        elif char_target == 'z':
            word[control] = 'z'
            word[target] = 'z'
    elif char_control == 'x':
        if char_target == 'i':
            word[control] = 'x'
            word[target] = 'x'
        elif char_target == 'x':
            word[control] = 'x'
            word[target] = 'i'
        elif char_target == 'y':
            word[control] = 'y'
            word[target] = 'z'
        elif char_target == 'z':
            lambdas = -1
            word[control] = 'y'
            word[target] = 'y'
    elif char_control == 'y':
        if char_target == 'i':
            word[control] = 'y'
            word[target] = 'x'
        elif char_target == 'x':
            word[control] = 'y'
            word[target] = 'y'
        elif char_target == 'y':
            word[control] = 'x'
            word[target] = 'z'
            lambdas = -1
        elif char_target == 'z':
            word[control] = 'x'
            word[target] = 'y'
    elif char_control == 'z':
        if char_target == 'y':
            word[control] = 'i'
            word[target] = 'y'
        elif char_target == 'z':
            word[control] = 'i'
            word[target] = 'z'
    return word_to_index(word)*lambdas

# Test for 3_0_2 file (num_qubits - control - target)
map_cx(4, 0, 2, 3)
