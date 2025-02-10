import numpy as np


def pauli_tabular(a, b):
    if a == 'i':
        return 1, b
    if b == 'i':
        return 1, a
    if a == b:
        return 1, 'i'
    if a == 'x':
        if b == 'y':
            return 1j, 'z'
        if b == 'z':
            return -1j, 'y'
    if a == 'y':
        if b == 'x':
            return -1j, 'z'
        if b == 'z':
            return 1j, 'x'
    if a == 'z':
        if b == 'x':
            return 1j, 'y'
        if b == 'y':
            return -1j, 'x'
    return 
        
def stabilizer_tabular(pauli, gate, param: float = 0):
    if pauli == 'i':
        return 1, 'i'
    if gate == 'h':
        if pauli == 'x':
            return 1, 'z'
        if pauli == 'y':
            return -1, 'y'
        if pauli == 'z':
            return 1, 'x'
    elif gate == 's':
        if pauli == 'x':
            return 1, 'y'
        elif pauli == 'y':
            return -1, 'x'
        elif pauli == 'z':
            return 1, 'z'
    elif gate == 't':
        if pauli == 'x':
            return [1/np.sqrt(2), 1/np.sqrt(2)], ['x','y']  # X -> (X + Y) / √2
        elif pauli == 'y':
            return [1/np.sqrt(2), -1/np.sqrt(2)], ['y','x']  # Y -> (Y - X) / √2
        elif pauli == 'z':
            return 1, 'z'  # Z -> Z
    elif gate == 'rx':  #
        if pauli == 'x':
            return 1, 'x'
        elif pauli == 'y':
            return [np.cos(param), np.sin(param)], ['y','z']
        elif pauli == 'z':
            return [np.cos(param), -np.sin(param)], ['z','y']
    elif gate == 'ry':
        if pauli == 'x':
            return [np.cos(param), -np.sin(param)], ['x','z']
        elif pauli == 'y':
            return 1, 'y'
        elif pauli == 'z':
            return [np.cos(param), np.sin(param)], ['z','x']
    elif gate == 'rz': 
        if pauli == 'x':
            return [np.cos(param), np.sin(param)], ['x','y']
        elif pauli == 'y':
            return [np.cos(param), -np.sin(param)], ['y','x']
        elif pauli == 'z':
            return 1, 'z'
        
    elif gate == 'cx':  # Controlled-X gate
        if pauli == 'ix':
            return 1, 'ix'
        elif pauli == 'xi':
            return 1, 'xx'
        elif pauli == 'iy':
            return 1, 'zy'
        elif pauli == 'yi':
            return 1, 'yx'
        elif pauli == 'iz':
            return 1, 'zz'
        elif pauli == 'zi':
            return 1, 'zi'
        elif pauli == 'xx':
            return 1, 'xi'
        elif pauli == 'xy':
            return 1, 'yz'
        elif pauli == 'xz':
            return -1, 'yy'
        elif pauli == 'yx':
            return 1, 'yi'
        elif pauli == 'yy':
            return -1, 'xz'
        elif pauli == 'yz':
            return 1, 'xy'
        elif pauli == 'zx':
            return 1, 'zx'
        elif pauli == 'zy':
            return 1, 'iy'
        elif pauli == 'zz':
            return 1, 'iz'
        elif pauli == 'ii':
            return 1, 'ii'
    else:
        return None