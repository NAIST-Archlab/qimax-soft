import cupy as cp
import polars as pl
from .mapper import weightss_to_lambda
from .instructor import Instructor
from .utils import word_to_index, char_to_index, index_to_word, create_word_zj
from .mapper import map_noncx, map_cx

class PStabilizer:
    def __init__(self, j: int, num_qubits: int):
        """PStabilizer is a Pauli term
        I encode it as two lists: indices (encoded Pauli words) and lambdas

        Args:
            j (int): index of stabilizer in the stabilizer group (generator)
            num_qubits (int)
        """
        # Init n stabilizer 
        # Each stabilizer is Z_j = I \otimes ... \otimes I \otimes Z (j^th) \otimes I \otimes ... \
        self.num_qubits = num_qubits
        self.lambdas = cp.ones(num_qubits)
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
    
