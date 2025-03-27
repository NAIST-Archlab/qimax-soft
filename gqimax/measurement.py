import numpy as np


def single_measurement(lambdas, indices, k):
    # Find corresponding lambda of P = Z_k in a Pauli term
    zk_index = 3 * (4 ** k)
    match = np.where(indices == zk_index)[0]
    return lambdas[match[0]] if len(match) > 0 else 0
