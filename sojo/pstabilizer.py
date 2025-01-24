import numpy as np
import polars as pl
from .instructor import Instructor, weightss_to_lambda
from .utils import word_to_index, char_to_index, index_to_word, create_word_zj
class PStabilizer:
    def __init__(self, j: int, num_qubits: int):
        """PStabilizer is a Pauli term
        I encode it as two lists: indices (encoded Pauli words) and lambdas

        Args:
            j (int): index of stabilizer in the stabilizer group (generator)
            num_qubits (int)
        """
        # Init this stabilizer is Z_j = I \otimes ... \otimes I \otimes Z \otimes I \otimes ... \
        self.num_qubits = num_qubits
        self.lambdas = np.array([1])
        self.indices = np.array([word_to_index(create_word_zj(j, num_qubits))])
        return
    def at(self, j: int | str):
        if type(j) == str:
            j = word_to_index(j)
        return self.lambdas[np.where(self.indices == j)[0]]
    def map(self, ins: Instructor):
        for j, order in enumerate(ins.orders):
            k = j // 2
            if order == 0:
                self.indices, self.lambdas = map_noncx(self.indices, self.lambdas, ins.LUT, k, self.num_qubits)
            else:
                self.indices, self.lambdas = map_cx(self.indices, self.lambdas, ins.xoperators[k], self.num_qubits)
        return
    def __str__(self) -> str:
        text = ""
        for i, index in enumerate(self.indices[:3]):
            text += (f"{np.round(self.lambdas[i],2)} * {index_to_word(index, self.num_qubits)} + ")
        text += (f"... + {np.round(self.lambdas[-1], 2)} * {index_to_word(self.indices[-1], self.num_qubits)}")
        return text
    

def map_noncx(indices: np.ndarray, lambdas: np.ndarray, LUT, k: int, num_qubits: int):
    """Given a list of indices and lambdas, return the indices and lambdas of the non-cx gates.
    Example:
        Initial: 1*IZ => indices = [3], lambdas = [1]
        Another: 2*XZ + 3*YY => indices = [7, 10], lambdas = [2, 3]
    Args:
        indices (np.ndarray)
        lambdas (np.ndarray)
        num_qubits (int)
        k: index of the operator
    Returns:
        np.ndarray, np.ndarray: mapped indices and lambdas
    """
    weightss = []
    for index in indices:
        weights = []
        word = index_to_word(index, num_qubits)
        for j, char in enumerate(word):
            i_in_lut = char_to_index(char) - 1
            if i_in_lut == -1:
                weights.append([1,0,0,0])
            else: 
                weights.append(LUT[k][j][i_in_lut])
        weightss.append(weights)
    # Weightss's now a 4-d tensor
    weightss = np.array(weightss)
    # Flattening the weightss into new lambdas, => Can be process effeciently on hardware
    lambdas = weightss_to_lambda(weightss, lambdas)
    # For most of the cases, lambdas will be sparse
    # In the worst case, it will be full dense,
    # and the below lines would be useless.
    indices = np.nonzero(lambdas)[0]
    lambdas = np.array(lambdas)[indices]
    return indices, lambdas

def map_cx(indices: np.ndarray, lambdas: np.ndarray, cxs: np.ndarray, num_qubits: int):
    """Mapping multple CX gates on a given indices and lambdas

    Args:
        indices (np.ndarray): Words after converting to integers
        lambdas (np.ndarray): Corresponding lambdas
        cxs (np.ndarray): List of [control, target] pairs from k^th CX-operator
        num_qubits (int)

    Returns:
        indices, lambdas
    """
    for control, target in cxs:
        # Access by indices
        # Example: df = [[0,1,2,3,4], [2,5,-3,4,1]]
        # I want to request element at [1,2,4], 
        # polors (pl) read saved df and return [5,-3,1]
        df = pl.read_csv(f'./sojo/db/{num_qubits}_{control}_{target}_cx.csv')
        selected_rows = df[indices]['out'].to_numpy()
        indices = np.abs(selected_rows)
        lambdas[np.where(selected_rows < 0)[0]] *= -1
    return indices, lambdas