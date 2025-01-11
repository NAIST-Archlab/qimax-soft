from .tabular import pauli_tabular, stabilizer_tabular
from .tp import kron_product
import numpy as np
from .pc import PauliComposer, PauliDiagComposer
from scipy import sparse
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
        p.word = 'i'*len(self.word)
        return p
    def get(self,index):
        return self.word[index]
    def update(self, index, character):
        self.word = self.word[:index] + character + self.word[index+1:]
    def update_scalar(self, scalar):
        self.scalar = self.scalar * scalar
    def update_word(self, word):
        self.word = word
    def to_pc(self):
        characters = list(set(self.word))
        if 'x' in characters or 'y' in characters:
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
            word_temp.word =  word_temp.word[:i] + character +  word_temp.word[i+1:]
        return word_temp
    def __str__(self):
        return str(self.scalar) + '*' + self.word

class PauliTerm:
    def __init__(self, words):
        self.words: list[PauliWord] = words
        self.num_qubits = len(words[0].word)
    def __str__(self):
        return ' + '.join([str(word) for word in self.words])
    def to_matrix_naive(self):
        # Return sum(P)
        matrix = np.zeros((2**(self.num_qubits), 2**(self.num_qubits)), dtype=complex)
        for word in self.words:
            matrix += word.to_matrix()
        return matrix
    def reduce(self):
        # Example: P = [1*zxx, 1*yzi, 1j*zxx] -- reduce() --> [(1+1j)*zxx, 1*yzi]
        # Example: P = [1*zxx, 1*yzi, -1*zxx] -- reduce() --> [1*yzi]
        element_sum = {}
        for pauli_word in self.words:
            scalar, string = pauli_word.scalar, pauli_word.word
            if np.abs(scalar) < 10**(-10):
                scalar = 0
            if string in element_sum:
                element_sum[string] += scalar
            else:
                element_sum[string] = scalar

        self.words = [PauliWord(value, key) for key, value in element_sum.items() if value != 0]
        return
    def to_matrix(self):
        # Return sum(P)
        matrix = self.words[0].to_pc().to_sparse()
        for word in self.words[1:]:
            matrix = matrix + word.to_pc().to_sparse()
        return matrix
    def multiply(self, other):
        # Multiply two PauliTerms
        new_terms = []
        for i in self.words:
            for j in other.words:
                new_terms.append(i.multiply(j))
        return PauliTerm(new_terms)
    def map(self, gate, index, param):
        # Example: xii -- (h, 0) --> [z]ii
        # Example: xxx -- (t, 0) --> [x']xx = [1/sqrt(2) (x + y)]xx
        # Example: zxz -- (cx, [0,2]) --> [i]x[z]
        # This function will map a Pauli term to another Pauli term
        # Example: xii + xxx -- (h, 0) --> [z]ii + [z]xx
        # Two keys: only act on coressponding qubits, and independence between pauli strings
        num_words = len(self.words)
        for j in range(num_words):
            # Process on a single Pauli string
            
            if gate == 'cx':
                # Index will be [control, target]
                # Word will be [word_control, word_target]
                out_scalar, output_word = stabilizer_tabular(
                    self.words[j].get(index[0]) + self.words[j].get(index[1]), gate)
                self.words[j].update(index[0], output_word[0])
                self.words[j].update(index[1], output_word[1])
                self.words[j].update_scalar(out_scalar)
            else:
                # Index will be just a scalar
                if gate in ['rx', 'ry', 'rz']:
                    out_scalar, output_word = stabilizer_tabular(
                        self.words[j].get(index), gate, param)
                else:
                    out_scalar, output_word = stabilizer_tabular(
                        self.words[j].get(index), gate)
                if type(output_word) == list:
                    self.words.append(self.words[j].duplicate())
                    self.words[j].update(index, output_word[0])
                    self.words[j].update_scalar(out_scalar[0])
                    self.words[-1].update(index, output_word[1])
                    self.words[-1].update_scalar(out_scalar[1])
                else:
                    self.words[j].update(index, output_word)
                    self.words[j].update_scalar(out_scalar) 
        self.reduce()
        return
class StabilizerGenerator:
    def __init__(self, num_qubits):
        # Example: if the system is 3 qubits, then the stabilizer group will be {Z0, Z1, Z2}
        # or {zii, izi, iiz}
        self.num_qubits = num_qubits
        init_words = ['i' * i + 'z' + 'i' * (num_qubits - i - 1) for i in range(num_qubits)]
        init_stabilizers = [PauliTerm([PauliWord(1, word)]) for word in init_words]
        self.stabilizers: list[PauliTerm] = init_stabilizers
        self.ps: list[PauliTerm] = []
    def __str__(self):
        string = '<'
        for i in self.stabilizers[:-1]:
            string += str(i) + '\n'
        return string + str(self.stabilizers[-1]) + '>'
    def reduce(self):
        for i, _ in enumerate(self.stabilizers):
            self.stabilizers[i].reduce()
        return
    def map(self, gate: str, index, param = 0):
        for i, _ in enumerate(self.stabilizers):
            self.stabilizers[i].map(gate, index, param)
        return
    def generate_p(self):
        def get_subsets(n):
            from itertools import chain, combinations
            input_set = set(range(n))
            subsets = list(chain.from_iterable(combinations(input_set, r) for r in range(len(input_set) + 1)))
            result = [set(subset) if subset else set() for subset in subsets] 
            return result
        sets = get_subsets(len(self.stabilizers))
        self.ps = []
        for indices in sets:
            p = PauliTerm([PauliWord(1, 'i'*self.num_qubits)])
            for j in indices:
                p = p.multiply(self.stabilizers[j])
            self.ps.append(p)
        return
    def generate_density_matrix_by_generator_naive(self):
        density_matrix = (np.eye(2**self.num_qubits) + self.stabilizers[0].to_matrix_naive())
        for stab in self.stabilizers[1:]:
            density_matrix = density_matrix @ (np.eye(2**self.num_qubits) + stab.to_matrix_naive())
        return density_matrix*(1/(2**self.num_qubits))    
    def generate_density_matrix_by_generator(self):
        density_matrix = (sparse.csr_matrix(np.eye(2**self.num_qubits)) + self.stabilizers[0].to_matrix())
        for j in range(1, self.num_qubits):
            density_matrix = density_matrix @ (sparse.csr_matrix(np.eye(2**self.num_qubits)) + self.stabilizers[j].to_matrix())
        return density_matrix*(1/(2**self.num_qubits))
    def generate_density_matrix(self):
        if len(self.ps) == 0:
            self.generate_p()
        density_matrix = np.zeros((2**self.num_qubits, 2**self.num_qubits), dtype=complex)
        for p in self.ps:
            density_matrix += p.to_matrix_naive()
        return (1/2**self.num_qubits)*density_matrix
StabilizerGenerator.h = lambda self, index: self.map('h', index)
StabilizerGenerator.s = lambda self, index: self.map('s', index)
StabilizerGenerator.t = lambda self, index: self.map('t', index)
StabilizerGenerator.cx = lambda self, index: self.map('cx', index)
StabilizerGenerator.rx = lambda self, param, index: self.map('rx', index, param)
StabilizerGenerator.ry = lambda self, param, index: self.map('ry', index, param)
StabilizerGenerator.rz = lambda self, param, index: self.map('rz', index, param)
