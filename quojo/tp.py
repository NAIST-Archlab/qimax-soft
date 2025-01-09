import numpy as np
def kr(pauli_string):
    def mapping(pauli):
        if pauli == 'i':
            return np.array([[1, 0], [0, 1]])
        elif pauli == 'x':
            return np.array([[0, 1], [1, 0]])
        elif pauli == 'y':
            return np.array([[0, -1j], [1j, 0]])
        elif pauli == 'z':
            return np.array([[1, 0], [0, -1]])
        else:
            raise ValueError('Invalid Pauli string.')
    kron_product = mapping(pauli_string[0])
    # Must acclerate this loop
    for pauli in pauli_string[1:]:
        kron_product = np.kron(kron_product, mapping(pauli))
    return kron_product