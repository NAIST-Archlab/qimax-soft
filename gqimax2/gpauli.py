
import cupy as cp

def mul_single_pauli(p1, p2):
    """Multiply two 1-qubit operators.
    """
    result = cp.zeros_like(p1, dtype=cp.int32)
    coeff = cp.ones_like(p1, dtype=cp.complex64)
    mask = p1 == 0
    result[mask] = p2[mask]
    mask = p2 == 0
    result[mask] = p1[mask]
    mask = (p1 == p2) & (p1 != 0)
    result[mask] = 0
    mask = (p1 == 1) & (p2 == 2)
    result[mask] = 3
    coeff[mask] = 1j
    mask = (p1 == 2) & (p2 == 1)
    result[mask] = 3
    coeff[mask] = -1j
    mask = (p1 == 1) & (p2 == 3)
    result[mask] = 2
    coeff[mask] = -1j
    mask = (p1 == 3) & (p2 == 1)
    result[mask] = 2
    coeff[mask] = 1j
    mask = (p1 == 2) & (p2 == 3)
    result[mask] = 1
    coeff[mask] = 1j
    mask = (p1 == 3) & (p2 == 2)
    result[mask] = 1
    coeff[mask] = -1j
    return result, coeff

def mul_pauli_word(index1, index2, num_qubits):
    """Multiply two encoded Pauli words.
    Example: With num_qubits = 2, II (0) * IX (1) = IX (1)

    Returns:
        The encoded Pauli word and its cofficient (+-1 or +-1j)
    """
    result_index = cp.zeros_like(index1, dtype=cp.int32)
    coefficient = cp.ones_like(index1, dtype=cp.complex64)
    for q in range(num_qubits):
        p1_digit = (index1 // (4 ** q)) % 4
        p2_digit = (index2 // (4 ** q)) % 4
        result_digit, coeff = mul_single_pauli(p1_digit, p2_digit)
        result_index += result_digit * (4 ** q)
        coefficient *= coeff
    return result_index, coefficient

def mul_pauli_term(lambdas_T1, indices_T1, lambdas_T2, indices_T2, num_qubits):
    """Multiply two Pauli terms.
    Example: (XX + ZZ) * (1YY + 2ZZ)
    Lambdas is the coefficient of the Pauli word.
    Returns:
        Indices and lambdas of the multiplication Pauli term.
    """
    lambdas_T1 = cp.array(lambdas_T1, dtype=cp.complex64)
    indices_T1 = cp.array(indices_T1, dtype=cp.int32)
    lambdas_T2 = cp.array(lambdas_T2, dtype=cp.complex64)
    indices_T2 = cp.array(indices_T2, dtype=cp.int32)
    i, j = cp.meshgrid(cp.arange(len(lambdas_T1)), cp.arange(len(lambdas_T2)), indexing='ij')
    i = i.flatten()
    j = j.flatten()
    lambda1 = lambdas_T1[i]
    lambda2 = lambdas_T2[j]
    index1 = indices_T1[i]
    index2 = indices_T2[j]
    result_index, coeff = mul_pauli_word(index1, index2, num_qubits)
    total_coeff = lambda1 * lambda2 * coeff
    total_coeff_real = cp.real(total_coeff)
    total_coeff_imag = cp.imag(total_coeff)
    # I don't know what going on below
    unique_indices, inverse_indices = cp.unique(result_index, return_inverse=True)
    lambdas_real = cp.zeros_like(unique_indices, dtype=cp.float32)
    cp.add.at(lambdas_real, inverse_indices, total_coeff_real)
    lambdas_imag = cp.zeros_like(unique_indices, dtype=cp.float32)
    cp.add.at(lambdas_imag, inverse_indices, total_coeff_imag)
    lambdas_new = lambdas_real + 1j * lambdas_imag
    return unique_indices.get(), lambdas_new.get()