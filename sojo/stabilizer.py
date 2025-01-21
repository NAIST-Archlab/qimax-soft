from .tabular import pauli_tabular, stabilizer_tabular
from .tp import kron_product
from .pc import PauliComposer, PauliDiagComposer
import concurrent.futures
import jax.numpy as jnp
import numpy as np
import jax
from jax.experimental.sparse import BCOO


class PauliWord:
    # Example: 2*ixz
    def __init__(self, scalar, word: str):
        self.scalar = scalar
        self.word = word

    def duplicate(self):
        p = PauliWord(self.scalar, self.word)
        return p

    def create_empty(self):
        p = PauliWord(self.scalar, self.word)
        p.word = "i" * len(self.word)
        return p

    def get(self, index):
        return self.word[index]

    def update(self, index, character):
        self.word = self.word[:index] + character + self.word[index + 1 :]

    def update_scalar(self, scalar):
        self.scalar = self.scalar * scalar

    def update_word(self, word):
        self.word = word

    def to_pc(self):
        characters = list(set(self.word))
        if "x" in characters or "y" in characters:
            return PauliComposer(self.word.upper(), self.scalar)
        else:
            return PauliDiagComposer(self.word.upper(), self.scalar)

    def to_matrix(self):
        matrix = self.scalar * kron_product(self.word)
        return matrix

    def multiply(self, other):
        # Multiply two PauliWords
        word_temp = self.create_empty()
        word_temp.scalar = self.scalar * other.scalar
        for i, _ in enumerate(self.word):
            scalar, character = pauli_tabular(self.word[i], other.word[i])
            word_temp.scalar = word_temp.scalar * scalar
            word_temp.word = word_temp.word[:i] + character + word_temp.word[i + 1 :]
        return word_temp

    def __str__(self):
        return str(self.scalar) + "*" + self.word


# class PauliTerm:
#     def __init__(self, words):
#         self.words: list[PauliWord] = words
#         self.num_qubits = len(words[0].word)
#     def __str__(self):
#         return ' + '.join([str(word) for word in self.words])
#     def to_matrix_naive(self):
#         # Return sum(P)
#         matrix = np.zeros((2**(self.num_qubits), 2**(self.num_qubits)), dtype=complex)
#         for word in self.words:
#             matrix += word.to_matrix()
#         return matrix
#     def reduce(self):
#         # Example: P = [1*zxx, 1*yzi, 1j*zxx] -- reduce() --> [(1+1j)*zxx, 1*yzi]
#         # Example: P = [1*zxx, 1*yzi, -1*zxx] -- reduce() --> [1*yzi]
#         element_sum = {}
#         for pauli_word in self.words:
#             scalar, string = pauli_word.scalar, pauli_word.word
#             if np.abs(scalar) < 10**(-10):
#                 scalar = 0
#             if string in element_sum:
#                 element_sum[string] += scalar
#             else:
#                 element_sum[string] = scalar

#         self.words = [PauliWord(value, key) for key, value in element_sum.items() if value != 0]
#         return
#     def to_matrix(self, mode):
#         # Return sum(P)
#         if mode == 'csr':
#             matrix = self.words[0].to_pc().to_csr()
#             for word in self.words[1:]:
#                 matrix = matrix + word.to_pc().to_csr()
#         else:
#             matrix = self.words[0].to_pc().to_coo()
#             for word in self.words[1:]:
#                 matrix = matrix + word.to_pc().to_coo()
#         return matrix
#     def multiply(self, other):
#         # Multiply two PauliTerms
#         new_terms = []
#         for i in self.words:
#             for j in other.words:
#                 new_terms.append(i.multiply(j))
#         return PauliTerm(new_terms)
#     def map(self, gate, index, param):
#         # Example: xii -- (h, 0) --> [z]ii
#         # Example: xxx -- (t, 0) --> [x']xx = [1/sqrt(2) (x + y)]xx
#         # Example: zxz -- (cx, [0,2]) --> [i]x[z]
#         # This function will map a Pauli term to another Pauli term
#         # Example: xii + xxx -- (h, 0) --> [z]ii + [z]xx
#         # Two keys: only act on coressponding qubits, and independence between pauli strings
#         num_words = len(self.words)
#         for j in range(num_words):
#             # Process on a single Pauli string

#             if gate == 'cx':
#                 # Index will be [control, target]
#                 # Word will be [word_control, word_target]
#                 out_scalar, output_word = stabilizer_tabular(
#                     self.words[j].get(index[0]) + self.words[j].get(index[1]), gate)
#                 self.words[j].update(index[0], output_word[0])
#                 self.words[j].update(index[1], output_word[1])
#                 self.words[j].update_scalar(out_scalar)
#             else:
#                 # Index will be just a scalar
#                 if gate in ['rx', 'ry', 'rz']:
#                     out_scalar, output_word = stabilizer_tabular(
#                         self.words[j].get(index), gate, param)
#                 else:
#                     out_scalar, output_word = stabilizer_tabular(
#                         self.words[j].get(index), gate)
#                 if type(output_word) == list:
#                     self.words.append(self.words[j].duplicate())
#                     self.words[j].update(index, output_word[0])
#                     self.words[j].update_scalar(out_scalar[0])
#                     self.words[-1].update(index, output_word[1])
#                     self.words[-1].update_scalar(out_scalar[1])
#                 else:
#                     self.words[j].update(index, output_word)
#                     self.words[j].update_scalar(out_scalar)
#         self.reduce()
#         return


class PauliTerm:
    """
    Present a Pauli term
    \tidle{P}_n=\sum_{i}\lambda_i P_i,
    where P_i is a Pauli word, \lambda_i is a complex number.
    """

    def __init__(self, words: dict[str, list[np.complex64]]):
        self.words: dict[str, list[np.complex64]] = words
        self.num_qubits = len(next(iter(words)))

    def __str__(self):
        return " + ".join(
            [str(np.round(v, 4)) + "*" + str(k) for k, v in self.words.items()]
        )

    def to_matrix_naive(self):
        result = None
        for word, scalar in self.words.items():
            if result is None:
                result = PauliComposer(word, scalar[0]).to_matrix()
            result = result + PauliComposer(word, scalar[0]).to_matrix()
        return result

    def to_matrix_with_i_naive(self):
        result = np.eye(2**self.num_qubits)
        for word, scalar in self.words.items():
            result = result + PauliComposer(word, scalar[0]).to_matrix()
        return result

    def to_matrix_jax(self):

        def zipped_list(rows, cols):
            return [list(item) for item in zip(rows, cols)]

        print("start to matrix")
        import time

        start = time.time()
        batch_indices = []
        batch_values = []
        for word, value in self.words.items():
            pc = PauliComposer(word, value[0])
            zip_list = zipped_list(pc.get_row(), pc.get_col())
            batch_indices.append(zip_list)
            batch_values.append(pc.get_value())
        batch_indices = jnp.array(batch_indices)
        batch_values = jnp.array(batch_values)
        batch_sparse_matrix = BCOO(
            (batch_values, batch_indices),
            shape=(len(self.words.items()), 2**pc.n, 2**pc.n),
        )
        jit_sum_bcoo: BCOO = jax.jit(lambda tensor: tensor.sum(axis=0))
        print("end to matrix", time.time() - start)
        return jit_sum_bcoo(batch_sparse_matrix).todense()

    def to_matrix_with_i_jax(self):
        self.words["i" * self.num_qubits] = [1]
        result = self.to_matrix_jax()
        self.words.pop("i" * self.num_qubits)
        return result

    def multiply(self, other):
        # Multiply two PauliTerms
        def multiply_two_word(word1, scalar1, word2, scalar2):
            # Multiply two PauliWords
            word_temp = "i" * len(word1)
            scalar_temp = scalar1 * scalar2
            for j, _ in enumerate(word1):
                scalar, character = pauli_tabular(word1[j], word2[j])
                scalar_temp *= scalar
                word_temp = word_temp[:j] + character + word_temp[j + 1 :]
            return word_temp, scalar_temp

        new_terms = {}
        for word1, scalar1 in self.words.items():
            for word2, scalar2 in other.words.items():
                word_temp, scalar_temp = multiply_two_word(
                    word1, scalar1[0], word2, scalar2[0]
                )
                new_terms[word_temp] = scalar_temp
        return PauliTerm(new_terms)

    def map(self, gate, index, param=0):
        """This function will map a Pauli term to another Pauli term
        # Example: xii -- (h, 0) --> [z]ii
        # Example: xxx -- (t, 0) --> [x']xx = [1/sqrt(2) (x + y)]xx
        # Example: zxz -- (cx, [0,2]) --> [i]x[z]
        # Example: xii + xxx -- (h, 0) --> [z]ii + [z]xx
        # Two keys: only act on coressponding qubits, and independence between pauli strings
        Args:
            gate (_type_): _description_
            index (_type_): _description_
            param (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """

        num_words = len(self.words)
        count = 0

        def update_word(word, index, character):
            return word[:index] + character + word[index + 1 :]
        for word_j, _ in list(self.words.items()):
            # Process on a single Pauli string
            if count > num_words:
                break
            count += 1

            if gate == "cx":
                # Index will be [control, target]
                # Word will be [word_control, word_target]
                out_scalar, output_word = stabilizer_tabular(
                    word_j[index[0]] + word_j[index[1]], gate
                )
                # Replace word_j with output_word
                # Example: xii -- (cx, [0,2]) --> [x]i[x]
                new_word = update_word(word_j, index[0], output_word[0])
                new_word = update_word(new_word, index[1], output_word[1])
                # Don't forget +- 1 factor from cx
                # If new_word is not in the dictionary, add it, and set [0] in the old word = 0
                # If new_word is in the dictionary, update the scalar and append at [-1] in the existance one
                # New_word = word_j in cnot mean there is no change, either scalar
                if new_word == word_j:
                    pass
                else:
                    if new_word in self.words:
                        self.words[new_word].append(self.words[word_j][0] * out_scalar)
                        self.words[word_j][0] = 0
                    else:
                        self.words[new_word] = [self.words[word_j][0] * out_scalar]
                        self.words[word_j][0] = 0
                # if abs(self.words[new_word]) < 10**(-10):
                #     self.words.pop(new_word)
            else:
                # Index will be just a scalar
                if gate in ["rx", "ry", "rz"]:
                    out_scalar, output_word = stabilizer_tabular(
                        word_j[index], gate, param
                    )
                else:
                    out_scalar, output_word = stabilizer_tabular(word_j[index], gate)
                if type(output_word) == list:
                    # One word turn to be two words,
                    # Only one index, 2 output words, 2 output scalars
                    # I append new words to the end of the list
                    # Example: xii -- (ry, 0) --> [x]ii + [z]ii
                    new_word_0 = word_j
                    new_word_1 = update_word(word_j, index, output_word[1])
                    if new_word_1 in self.words:
                        self.words[new_word_1].append(
                            self.words[word_j][0] * out_scalar[1]
                        )

                    else:
                        self.words[new_word_1] = [self.words[word_j][0] * out_scalar[1]]
                    self.words[word_j][0] *= out_scalar[0]
                else:
                    new_word = update_word(word_j, index, output_word)
                    if new_word in self.words:
                        self.words[new_word].append(self.words[word_j][0] * out_scalar)
                        self.words[word_j][0] = 0
                    else:
                        self.words[new_word] = [self.words[word_j][0] * out_scalar]
                        self.words[word_j][0] = 0
        # Reduce
        # Example: P = [1*zxx, 1*yzi, 1j*zxx] -- reduce() --> [(1+1j)*zxx, 1*yzi]
        # Example: P = [1*zxx, 1*yzi, -1*zxx] -- reduce() --> [1*yzi]
        self.words = {
            key: [s]
            for key, value in self.words.items()
            if abs(s := sum(value)) > 10 ** (-10)
        }
        return


def map_x(pc: PauliTerm, gate="rx", index=0, param=0) -> PauliTerm:
    """
    This function will map a Pauli term to another Pauli term
    Example: xii -- (h, 0) --> [z]ii
    Example: xxx -- (t, 0) --> [x']xx = [1/sqrt(2) (x + y)]xx
    Example: zxz -- (cx, [0,2]) --> [i]x[z]
    Example: xii + xxx -- (h, 0) --> [z]ii + [z]xx
    Two keys: only act on coressponding qubits, and independence between pauli strings

    Args:
        pc (PauliTerm): _description_
        gate (str, optional): _description_. Defaults to 'rx'.
        index (int, optional): _description_. Defaults to 0.
        param (int, optional): _description_. Defaults to 0.

    Returns:
        PauliTerm: _description_
    """
    words = pc.words
    num_words = len(words)
    count = 0

    def update_word(word, index, character):
        return word[:index] + character + word[index + 1 :]

    for word_j, _ in list(words.items()):
        # Process on a single Pauli string
        if count > num_words:
            break
        count += 1
        if gate == "cx":
            # Index will be [control, target]
            # Word will be [word_control, word_target]
            out_scalar, output_word = stabilizer_tabular(
                word_j[index[0]] + word_j[index[1]], gate
            )
            # Replace word_j with output_word
            # Example: xii -- (cx, [0,2]) --> [x]i[x]
            new_word = update_word(word_j, index[0], output_word[0])
            new_word = update_word(new_word, index[1], output_word[1])
            # Don't forget +- 1 factor from cx
            # If new_word is not in the dictionary, add it, and set [0] in the old word = 0
            # If new_word is in the dictionary,
            # update the scalar and append at [-1] in the existance one
            # New_word = word_j in cnot mean there is no change, either scalar
            if new_word == word_j:
                pass
            else:
                if new_word in words:
                    words[new_word].append(words[word_j][0] * out_scalar)
                    words[word_j][0] = 0
                else:
                    words[new_word] = [words[word_j][0] * out_scalar]
                    words[word_j][0] = 0
            # if abs(self.words[new_word]) < 10**(-10):
            #     self.words.pop(new_word)
        else:
            # Index will be just a scalar
            if gate in ["rx", "ry", "rz"]:
                out_scalar, output_word = stabilizer_tabular(word_j[index], gate, param)
            else:
                out_scalar, output_word = stabilizer_tabular(word_j[index], gate)
            if type(output_word) == list:
                # One word turn to be two words,
                # Only one index, 2 output words, 2 output scalars
                # I append new words to the end of the list
                # Example: xii -- (ry, 0) --> [x]ii + [z]ii
                # new_word_0 = word_j
                new_word_1 = update_word(word_j, index, output_word[1])
                if new_word_1 in words:
                    words[new_word_1].append(words[word_j][0] * out_scalar[1])

                else:
                    words[new_word_1] = [words[word_j][0] * out_scalar[1]]
                words[word_j][0] *= out_scalar[0]
            else:
                new_word = update_word(word_j, index, output_word)
                if new_word in words:
                    words[new_word].append(words[word_j][0] * out_scalar)
                    words[word_j][0] = 0
                else:
                    words[new_word] = [words[word_j][0] * out_scalar]
                    words[word_j][0] = 0
    # Reduce
    # Example: P = [1*zxx, 1*yzi, 1j*zxx] -- reduce() --> [(1+1j)*zxx, 1*yzi]
    # Example: P = [1*zxx, 1*yzi, -1*zxx] -- reduce() --> [1*yzi]
    return PauliTerm(
        {key: [s] for key, value in words.items() if abs(s := sum(value)) > 10 ** (-10)}
    )


class StabilizerGenerator:
    """It's just n-PauliTerm, where n is the number of qubits"""

    def __init__(self, num_qubits):
        # Example: if the system is 3 qubits, then the stabilizer group will be {Z0, Z1, Z2}
        # or {zii, izi, iiz}
        self.num_qubits = num_qubits
        init_words = [
            "i" * j + "z" + "i" * (num_qubits - j - 1) for j in range(num_qubits)
        ]
        init_stabilizers = [PauliTerm({word: [1]}) for word in init_words]
        self.stabilizers: list[PauliTerm] = init_stabilizers
        self.ps: list[PauliTerm] = []

    def __str__(self):
        string = "<"
        for i in self.stabilizers[:-1]:
            string += str(i) + "\n"
        return string + str(self.stabilizers[-1]) + ">"

    def map_parallel(self, gate: str, index, param=0):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            self.stabilizers = list(executor.map(map_x, self.stabilizers))
        return

    # def map_parallel(self, gate: str, index, param = 0):
    #     with concurrent.futures.ProcessPoolExecutor() as executor:
    #         self.stabilizers = list(executor.map(
    #             lambda stabilizer: map_parallel(stabilizer, gate, index, param),
    #             self.stabilizers))
    #     return

    def map(self, gate: str, index, param=0):
        for i, _ in enumerate(self.stabilizers):
            self.stabilizers[i].map(gate, index, param)
        return

    def generate_group(self):
        def get_subsets(n):
            from itertools import chain, combinations

            input_set = set(range(n))
            subsets = list(
                chain.from_iterable(
                    combinations(input_set, r) for r in range(len(input_set) + 1)
                )
            )
            result = [set(subset) if subset else set() for subset in subsets]
            return result

        sets = get_subsets(len(self.stabilizers))
        self.ps = []
        for indices in sets:
            p = PauliTerm({1, "i" * self.num_qubits})
            for j in indices:
                p = p.multiply(self.stabilizers[j])
            self.ps.append(p)
        return

    def generate_density_matrix_by_generator_naive(self):
        density_matrix = self.stabilizers[0].to_matrix_with_i_naive()
        for j in range(1, self.num_qubits):
            density_matrix = (
                density_matrix @ self.stabilizers[j].to_matrix_with_i_naive()
            )
        return density_matrix * (1 / (2**self.num_qubits))

    def generate_density_matrix_by_generator_jax(self):
        density_matrix = self.stabilizers[0].to_matrix_with_i_jax()
        for j in range(1, self.num_qubits):
            density_matrix = density_matrix @ self.stabilizers[j].to_matrix_with_i_jax()
        return density_matrix * (1 / (2**self.num_qubits))

    def generate_density_matrix_by_group_naive(self):
        if len(self.ps) == 0:
            self.generate_group()
        density_matrix = self.ps[0].to_matrix_naive()
        for j in range(1, len(self.ps)):
            density_matrix += self.ps[j].to_matrix_naive()
        return (1 / 2**self.num_qubits) * density_matrix


StabilizerGenerator.h = lambda self, index: self.map("h", index)
StabilizerGenerator.s = lambda self, index: self.map("s", index)
StabilizerGenerator.t = lambda self, index: self.map("t", index)
StabilizerGenerator.cx = lambda self, index: self.map("cx", index)
StabilizerGenerator.rx = lambda self, param, index: self.map("rx", index, param)
StabilizerGenerator.ry = lambda self, param, index: self.map("ry", index, param)
StabilizerGenerator.rz = lambda self, param, index: self.map("rz", index, param)
