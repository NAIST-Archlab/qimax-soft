"""
Pauli(Diag)Composer class definition.

See: https://arxiv.org/abs/2301.00560

The below code from https://github.com/sebastianvromero/PauliComposer
"""

import cupy as cp
import cupyx.scipy.sparse as ss
from numbers import Number

BINARY = {'I': '0', 'X': '1', 'Y': '1', 'Z': '0'}

class PauliComposer:

    def __init__(self, entry: str, weight: Number = None):
        # Compute the number of dimensions for the given entry
        n = len(entry)
        self.n = n

        # Compute some helpful powers
        self.dim = 1<<n

        # Store the entry converting the Pauli labels into uppercase
        self.entry = entry.upper()
        self.paulis = list(set(self.entry))

        # (-i)**(0+4m)=1, (-i)**(1+4m)=-i, (-i)**(2+4m)=-1, (-i)**(3+4m)=i
        mat_ent = {0: 1, 1: -1j, 2: -1, 3: 1j}

        # Count the number of ny mod 4
        self.ny = self.entry.count('Y') & 3
        init_ent = mat_ent[self.ny]
        if weight is not None:
            # first non-zero entry
            init_ent *= weight
        self.init_entry = init_ent
        self.iscomplex = cp.iscomplex(init_ent)

        # Reverse the icput and its 'binary' representation
        rev_entry = self.entry[::-1]
        rev_bin_entry = ''.join([BINARY[ent] for ent in rev_entry])

        # Column of the first-row non-zero entry
        col_val = int(''.join([BINARY[ent] for ent in self.entry]), 2)

        # Initialize an empty (2**n x 3)-matrix (rows, columns, entries)
        # row = cp.arange(self.dim) 
        col = cp.empty(self.dim, dtype=cp.float32)
        # FIXME: storing rows and columns as cp.complex64 since NumPy arrays
        # must have the same data type for each entry. Consider using
        # pd.DataFrame?

        col[0] = col_val  # first column
        # The AND bit-operator computes more rapidly mods of 2**n. Check that:
        #    x mod 2**n == x & (2**n-1)
        if weight is not None:
            if self.iscomplex:
                ent = cp.full(self.dim, self.init_entry)
            else:
                ent = cp.full(self.dim, float(self.init_entry))
        else:
            if self.iscomplex:
                ent = cp.full(self.dim, self.init_entry, dtype=cp.complex64)
            else:
                ent = cp.full(self.dim, self.init_entry, dtype=cp.complex64)

        for ind in range(n):
            p = 1<<int(ind)  # left-shift of bits ('1' (1) << 2 = '100' (4))
            p2 = p<<1
            disp = p if rev_bin_entry[ind] == '0' else -p  # displacements
            col[p:p2] = col[0:p] + disp  # compute new columns
            # col[p:p2] = col[0:p] ^ p  # alternative for computing column

            # Store the new entries using old ones
            if rev_entry[ind] in ['I', 'X']:
                ent[p:p2] = ent[0:p]
            else:
                ent[p:p2] = -ent[0:p]

        self.col = col
        self.mat = ent
    def get_value(self):
        return self.mat
    def get_row(self):
        return cp.arange(self.dim)
    def get_col(self):
        return self.col
    def to_dict(self):
        return [(i, self.col[i], self.mat[i]) for i in range(self.dim)]
    def to_coo(self):
        self.row = cp.arange(self.dim)
        return ss.coo_matrix((self.mat, (self.row, self.col)),
                             shape=(self.dim, self.dim))
    
    def to_csr(self):
        self.row = cp.arange(self.dim)
        return ss.csr_matrix((self.mat, (self.row, self.col)),
                             shape=(self.dim, self.dim))

    def to_matrix(self):
        return self.to_csr().toarray()
    def add(self, pc2):
        self.to_csr() + pc2.to_csr()

class PauliDiagComposer:

    def __init__(self, entry: str, weight: Number = None):
        # Compute the number of dimensions for the given entry
        n = len(entry)
        self.n = n
        self.row = None
        # Compute some helpful powers
        self.dim = 1<<n

        # Store the entry converting the Pauli labels into uppercase
        self.entry = entry.upper()

        # Reverse the icput and its 'binary' representation
        rev_entry = self.entry[::-1]

        # FIXME: storing rows and columns as cp.complex64 since NumPy arrays
        # must have the same data type for each entry. Consider using
        # pd.DataFrame?

        # mat[:, 0] = mat[:, 1] = cp.arange(self.dim)  # rows, columns
        if weight is not None:
            # first non-zero entry
            mat = cp.full(self.dim, weight)
        else:
            mat = cp.ones(self.dim, dtype=cp.int8)

        for ind in range(n):
            p = 1<<int(ind)  # left-shift of bits ('1' (1) << 2 = '100' (4))
            p2 = p<<1
            # Store the new entries using old ones
            if rev_entry[ind] == 'I':
                mat[p:p2] = mat[0:p]
            else:
                mat[p:p2] = -mat[0:p]

        self.mat = mat
    def get_value(self):
        return self.mat
    def to_coo(self):
        self.row = cp.arange(self.dim)
        return ss.coo_matrix((self.mat, (self.row, self.row)),
                             shape=(self.dim, self.dim))
    def get_row(self):
        return cp.arange(self.dim)
    def get_col(self):
        return cp.arange(self.dim)
    def to_csr(self):
        self.row = cp.arange(self.dim)
        return ss.csr_matrix((self.mat, (self.row, self.row)),
                             shape=(self.dim, self.dim))

    def to_matrix(self):
        return self.to_csr().toarray()