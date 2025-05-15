[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-UNITARY%20FOUNDATION-brightgreen.svg?style=for-the-badge)](https://unitary.foundation)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/vutuanhai237/qoop)

qimax - an extended stabilizer formalism implemented on software side

Paper: https://arxiv.org/abs/2505.03307

Documentation: https://deepwiki.com/NAIST-Archlab/qimax-soft

**Utilities function used in three mode**: The Qimax v2 and v3 use different encoders. v2 encode Pauli matrix and summation of Pauli matrix as the same 4D vector; in specially, it considers a single Pauli matrix as a 1-term sum. Both modes encode the Pauli string as an integer using base 4. We summarize all other functions as:


- pauli_to_index($p$) and index_to_pauli($i$): Encode a Pauli $\{I, X, Y, Z\}$ as an integer $\{0, 1, 2, 3\}$ and vice versa.
- pauli_to_weight($p$) and weight_to_pauli($w$): Encode a Pauli $\{I, X, Y, Z\}$ as a one-hot vector ${e}_j$ and vice versa.
- word_to_index($\mathbb{P}$) and index_to_word($\mathbb{P}, n$): Encode a Pauli word as integer using base-4. On the other hand, we need an additional parameter \#Qubits.
- index_to_indices($i$) and indices_to_index($I$): Convert an encoded Pauli word (integer form) to array form and vice versa. For example: $27 \ (XYZ) \rightarrow [1, 2, 3]$.
- createZ_j(j, n): Return $n$-dim one-hot vector ${e}_j$.
- create_chain($K, K'$): Create the chain $[1, 0, 1, \ldots]$ if the instructions begin with a 1-qubit gate and vice versa.
- divide_instruction(instruction): Split the instruction into $\{U_{j,i}\}$ and $\{V_j\}$, $K$ and $K'$ are two corresponding first dimensional.
