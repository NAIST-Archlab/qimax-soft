import cupy as cp
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
        self.lambdas = cp.array([1])
        self.indices = cp.array([word_to_index(create_word_zj(j, num_qubits))])
        return
    def at(self, j: int | str):
        if type(j) == str:
            j = word_to_index(j)
        return self.lambdas[cp.where(self.indices == j)[0]]
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
        lambdas_cp = self.lambdas.get()[:3]
        indices_cp = self.indices.get()[:3]
        for i, index in enumerate(indices_cp):
            text += (f"{cp.round(lambdas_cp[i], 2)} * {index_to_word(index, self.num_qubits)} + ")
        text += (f"... + {cp.round(self.lambdas.get()[-1], 2)} * {index_to_word(self.indices.get()[-1], self.num_qubits)}")
        return text
    

def map_noncx(indices: cp.ndarray, lambdas: cp.ndarray, LUT, k: int, num_qubits: int):
    """Given a list of indices and lambdas, return the indices and lambdas of the non-cx gates.
    Example:
        Initial: 1*IZ => indices = [3], lambdas = [1]
        Another: 2*XZ + 3*YY => indices = [7, 10], lambdas = [2, 3]
    Args:
        indices (cp.ndarray)
        lambdas (cp.ndarray)
        num_qubits (int)
        k: index of the operator
    Returns:
        cp.ndarray, cp.ndarray: mapped indices and lambdas
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
    # Weightss's now a 4-d tensor, 4^n x n x 4
    weightss = cp.array(weightss)
    # Flattening the weightss into new lambdas, => Can be process effeciently on hardware
    lambdas = weightss_to_lambda(weightss, lambdas)
    # For most of the cases, lambdas will be sparse
    # In the worst case, it will be full dense,
    # and the below lines would be useless.
    indices = cp.nonzero(lambdas)[0]
    lambdas = lambdas[indices]
    return indices, lambdas

def map_cx(indices: cp.ndarray, lambdas: cp.ndarray, cxs: cp.ndarray, num_qubits: int):
    """Mapping multiple CX gates on a given indices and lambdas

    Args:
        indices (cp.ndarray): Words after converting to integers
        lambdas (cp.ndarray): Corresponding lambdas
        cxs (cp.ndarray): List of [control, target] pairs from k^th CX-operator
        num_qubits (int)

    Returns:
        indices, lambdas
    """
    for control, target in cxs:
        df = pl.read_csv(f'./qimax/db/{num_qubits}_{control}_{target}_cx.csv')
        out_array = cp.array(df['out'].to_numpy())
        selected_rows = out_array[indices]
        indices = cp.abs(selected_rows)
        lambdas[cp.where(selected_rows < 0)[0]] *= -1
    return indices, lambdas