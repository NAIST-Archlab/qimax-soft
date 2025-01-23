import numpy as np

def weightss_to_lambda(weightss: np.ndarray, lambdas) -> np.ndarray:
    """A sum of transformed word (a matrix 4^n x n x 4) to list
    Example for a single transformed word (treated as 1 term): 
        k*[[1,2,3,4], [1,2,3,4]] 
            -> k*[1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12, 4, 8, 12, 16]
    Example for this function (sum, 2 qubits, 3 term):
        [[[1,2,3,4], [1,2,3,4]], [[1,2,3,4], [1,2,3,4]], [[1,2,3,4], [1,2,3,4]]] 
            -> [ 3.  6.  9. 12.  6. 12. 18. 24.  9. 18. 27. 36. 12. 24. 36. 48.]
    Args:
        weightss (np.ndarray): _description_

    Returns:
        np.ndarray: lambdas
    """
    num_qubits = weightss.shape[1]
    new_lambdas = np.zeros((4**num_qubits))
    for j, weights in enumerate(weightss):
        combinations = np.array(np.meshgrid(*weights)).T.reshape(-1, len(weights))
        new_lambdas += lambdas[j]*np.prod(combinations, axis=1)
    # This lambdas is still in the form of 4^n x 1, 
    # we need to ignore 0 values in the next steps
    # In the worst case, there is no 0 values.
    return new_lambdas

weightss = np.load('weight_0.npy')
lambdas = np.load('lambdas_0.npy')
output = np.load('output_0.npy')
print(weightss_to_lambda(weightss, lambdas))
print(output)
