import cupy as cp
import cupyx as cpx
from .kernel import broadcasted_multiplies_kernel, broandcast_base4_kernel, sum_distributions_kernel, construct_lut_noncx_kernel, map_cx_kernel, index_to_indices_kernel
from .constant import BLOCK_SIZE



#################################################################
# ---- map_noncx to map_cx section (weightsss_to_lambdass) ---- #
#################################################################

def weightsss_to_lambdass(lambdass, weightsss, indicesss):
    num_qubits = len(weightsss)
    lambdass_res = [None] * num_qubits
    indicess_res = [None] * num_qubits
    for i in range(num_qubits):
        lambdass_res[i], indicess_res[i] = sum_distributions(
            broadcasted_multiplies(lambdass[i], weightsss[i]),
            broadcasted_base4(indicesss[i])
        )
    return lambdass_res, indicess_res

def broadcasted_multiplies(lambdas, weightss):
    """
    Returns:
        weightss (k x ?): k arrays of weights
    """
    def prepare(weightss):
        k = len(weightss)
        dims = len(weightss[0])
        shapes = cp.array([len(arr) for arrays in weightss for arr in arrays], dtype=cp.int32)
        data = cp.concatenate([arr for arrays in weightss for arr in arrays])
        lengths = shapes.reshape(k, dims)
        offsets = cp.cumsum(cp.concatenate([cp.array([0], dtype=cp.int32), lengths.flatten()[:-1]]))
        sizes = cp.prod(lengths.reshape(k, dims), axis=1)
        offsets_result = cp.cumsum(cp.concatenate([cp.array([0], dtype=cp.int32), sizes[:-1]]))
        return shapes, data, offsets, offsets_result
    
    # - shapes: cp.ndarray (N * dims,) - save all k_i
    # - data: cp.ndarray - Flattened array of all arrays in weightss
    # - offsets: cp.ndarray (N * dims,) - Beginning indices
    # - offsets_result: cp.ndarray (N,) - Beginning indices of results
    shapes, data, offsets, offsets_result = prepare(weightss)
    k = lambdas.size
    dims = shapes.size // k
    total_size = int(offsets_result[-1] + cp.prod(shapes[-dims:]))
    result_cp = cp.zeros(total_size, dtype=cp.float32)
    grid_size = (total_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    broadcasted_multiplies_kernel(
        (grid_size,), (BLOCK_SIZE,),
        (lambdas, shapes, data, offsets, offsets_result, result_cp, k, dims, total_size)
    )
    
    weightss = cp.split(result_cp, offsets_result[1:].tolist())
    return weightss


def broadcasted_multiplies_base4(indicess):
    r'''
    Example: indicess = [[0, 2, 3], [1, 2, 3]] # (I + Y + Z)(X + Y + Z)
    => output = [0*4+1, 0*4+2, 0*4+3, 2*4+1, 2*4+2, 2*4+3, 3*4+1, 3*4+2, 3*4+3]
    => [1, 2, 3, 9, 10, 11, 13, 14, 15] # [IX, IY, IZ, YX, YY, YZ, ZX, ZY, ZZ]
    '''
    def prepare_data(indicess):
        num_qubits = len(indicess)        
        n_dim = len(indicess[0])   
        shapes = cp.array([len(index) for indices in indicess for index in indices], dtype=cp.int32)
        data = cp.concatenate([index for indices in indicess for index in indices])
        lengths = shapes.reshape(num_qubits, n_dim)
        offsets = cp.cumsum(cp.concatenate([cp.array([0], dtype=cp.int32), lengths.flatten()[:-1]]))
        sizes = cp.prod(lengths.reshape(num_qubits, n_dim), axis=1)
        offsets_result = cp.cumsum(cp.concatenate([cp.array([0], dtype=cp.int32), sizes[:-1]]))
        powers = cp.array([4 ** (n_dim - i - 1) for i in range(n_dim)], dtype=cp.int32)
        return shapes, data, offsets, offsets_result, powers, sizes
    # Below is from AI, I dont know what it means
    shapes, data, offsets, offsets_result, powers, sizes = prepare_data(indicess)
    num_qubits = len(indicess)
    n_dim = len(indicess[0])
    total_size = int(sizes.sum())
    output = cp.empty(total_size, dtype=cp.int32)
    grid_size = (total_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    broandcast_base4_kernel((grid_size,), (BLOCK_SIZE,),
                 (shapes, data, offsets, offsets_result, powers, output, num_qubits, n_dim, total_size))
    return cp.split(output, offsets_result[1:].tolist()) # k arrays of indices, named indicess



def broadcasted_base4(arrayss):
    results = []
    for arrays in arrayss:
        # Expand all arrays to shape (1, ..., N, ..., 1)
        reshaped = [a.reshape((1,) * i + (-1,) + (1,) * (len(arrays) - i - 1)) for i, a in enumerate(arrays)]
        # Broadcast to common shape
        broadcasted = cp.broadcast_arrays(*reshaped)
        # Stack along last axis to get combinations
        stacked = cp.stack(broadcasted, axis=-1)
        # Reshape to (num_combinations, num_features)
        results.append(stacked.reshape(-1, len(arrays)))
    return results


def sum_distributions(ragged_matrix, ragged_indices):
    """
    Sum values from ragged_matrix based on corresponding indices in ragged_indices.
    If indices are identical, their values are summed; otherwise, they are appended to the result.

    Args:
        ragged_indices (list of cp.ndarray): List of CuPy arrays containing multi-dimensional indices.
        ragged_matrix (list of cp.ndarray): List of CuPy arrays containing values to sum.

    Returns:
        tuple: (unique_indices, summed_values)
            - unique_indices: CuPy array of unique indices.
            - summed_values: CuPy array of summed values corresponding to unique indices.
    """
    # Concatenate all indices into a single array of shape (N, k)
    all_indices = cp.vstack(ragged_indices, dtype = cp.int8)
    
    # Concatenate all values into a single array of shape (N,)
    all_values = cp.concatenate(ragged_matrix, dtype = cp.float32)
    
    # Total number of index-value pairs
    N = all_indices.shape[0]
    
    # Sort indices lexicographically by rows
    sort_idx = cp.lexsort(all_indices.T[::-1])
    sorted_indices = all_indices[sort_idx]
    sorted_values = all_values[sort_idx]
    
    # Identify where rows differ to mark the start of each unique group
    diff = (sorted_indices[1:] != sorted_indices[:-1]).any(axis=1)
    is_new = cp.concatenate([cp.array([True]), diff])
    split_idx = cp.where(is_new)[0]
    
    # Extract unique indices
    unique_indices = sorted_indices[split_idx]
    
    # Compute sums for each group using cumulative sum
    cumsum = cp.cumsum(sorted_values)
    total_cumsum = cp.concatenate([cp.array([0]), cumsum], dtype=cp.float32)
    group_sums = total_cumsum[cp.concatenate([split_idx[1:], cp.array([N])])] - total_cumsum[split_idx]
    
    return group_sums, unique_indices




# --------------------------- #
# ---- map_noncx section ---- #
# --------------------------- #

def construct_lut_noncx(grouped_instructorss, num_qubits: int):
    def refactor_lut_noncx(lut):
        r'''
        Eliminate all zero weights in the lut and return the corresponding indices,
        it only run one times so I think it is ok to use CPU
        '''
        indicesss = [[[None for k in range(lut.shape[2])] for i in range(lut.shape[1])] for j in range(lut.shape[0])]
        lutsss = []
        for i, weightss in enumerate(lut):
            lutss = []
            for j, weights in enumerate(weightss):
                luts = []
                for k, weight in enumerate(weights):
                    non_zero_indices = cp.nonzero(weight)[0]
                    indicesss[i][j][k] = cp.array([w + 1 for w in non_zero_indices], dtype = cp.int8)
                    luts.append(weight[non_zero_indices])
                lutss.append(luts)
            lutsss.append(lutss)
        return lutsss, indicesss
    
    
    
    K = len(grouped_instructorss)
    
    instructors_flat = []
    offsets = [0]
    num_instructors = []
    gate_map = {"h": 0, "s": 1, "t": 2, "rx": 3, "ry": 4, "rz": 5}
    
    for k in range(K):
        for j in range(num_qubits):
            instructors = grouped_instructorss[k][j]
            num_ins = len(instructors)
            num_instructors.append(num_ins)
            for gate, _, param in instructors:
                instructors_flat.extend([float(gate_map[gate]), 0.0, float(param if param is not None else 0.0)])
            offsets.append(offsets[-1] + num_ins * 3)
    
    instructors_flat = cp.array(instructors_flat, dtype=cp.float32)
    offsets = cp.array(offsets[:-1], dtype=cp.int32)
    num_instructors = cp.array(num_instructors, dtype=cp.int32)
    
    lut = cp.zeros((K, num_qubits, 3, 3), dtype=cp.float32)
    lut_flat = lut.reshape(K * num_qubits * 3 * 3)
    
    total_elements = K * num_qubits * 3
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    construct_lut_noncx_kernel((grid_size,), (BLOCK_SIZE,), 
                              (instructors_flat, offsets, num_instructors, K, num_qubits, lut_flat))
    lutsss, indicesss = refactor_lut_noncx(lut.reshape(K, num_qubits, 3, 3))
    return lutsss, indicesss


def map_noncx(lambdass, indicesss, lut_at_k, indicesss_at_k):
    r'''
    indicesss = 
    
    [
		stb_0: [] (term 0) + [] (term-1) + ... + [] (term-k0),
			
   			Each term [] lambda x [CCC...C] (n char)
		
  		stb_1: [] (term 0) + [] (term-1) + ... + [] (term-k1),
		
  		...
		
  		stb_n-1: [] (term 0) + [] (term-1) + ... + [] (term-k0)
	]
    
    '''
    weightsss = []
    indicesss_out = []
    for j, indicess in enumerate(indicesss):  # stabilizer
        r'''Dealing with stabilizer j [] (term 0) + [] (term-1) + ... + [] (term-k0),
			# Each term [] = lambda x [CCC...C]
			After mapping, 
   			lambda x [CCC...C] => lambda [Ax + By + Cz] x [Ax + By + Cz] x ... x [Ax + By + Cz] (n times)
			=> indiess = [array([x,y,z]) x n]
        '''
        weightss = []
        indicess_out = []
        for k, indices in enumerate(indicess): # term k_th
            weights = []
            indices_out = []  
            for qubit, index in enumerate(indices): # qubit j_th
                if index == 0:
                    weights.append(cp.array([1], dtype=cp.float32))
                    indices_out.append(cp.array([0], dtype=cp.int8))
                else:
                    weights.append(lut_at_k[qubit][int(index) - 1])
                    indices_out.append(indicesss_at_k[qubit][int(index) - 1])
            weightss.append(weights)
            indicess_out.append(indices_out)
        weightsss.append(weightss)
        indicesss_out.append(indicess_out)
    import time
    start1 = time.time()
    w = weightsss_to_lambdass(lambdass, weightsss, indicesss_out)
    print('w2l ...:', time.time() - start1)
    return w


# def map_noncx(lambdass, indicesss, lut_at_k, indicesss_at_k):
#     weightsss = []
#     indicesss_out = []

#     for j, indicess in enumerate(indicesss):  # stabilizer
#         weightss = []
#         indicess_out = []

#         # Process each stabilizer (term-by-term)
#         for k, indices in enumerate(indicess):  # term k_th
#             # Mask for indices that are zero (no mapping needed for zero-index)
#             non_zero_mask = indices != 0
#             # Initialize weights and indices_out arrays
#             weights = cp.ones_like(indices, dtype=cp.float32)
#             indices_out = cp.zeros_like(indices, dtype=cp.int8)

#             # Apply LUT lookups only for non-zero indices
#             indices_out[non_zero_mask] = indicesss_at_k[j][k][indices[non_zero_mask] - 1]
#             weights[non_zero_mask] = lut_at_k[j][k][indices[non_zero_mask] - 1]

#             weightss.append(weights)
#             indicess_out.append(indices_out)

#         weightsss.append(weightss)
#         indicesss_out.append(indicess_out)

#     # Perform the final mapping step
#     w = weightsss_to_lambdass(lambdass, weightsss, indicesss_out)

#     return w




# # --------------------------- #
# # ----- map_cx section ------ #
# # --------------------------- #


def cuda_map_cx(indices, control, target):
    # if indices.dtype != cp.int8:
    #     raise TypeError("Indices must be of type int8")
    
    k, n = indices.shape
    indicess = cp.empty_like(indices)
    signss = cp.empty(k, dtype = cp.int8)
    grid_size = (k + BLOCK_SIZE - 1) // BLOCK_SIZE
    map_cx_kernel((grid_size,), (BLOCK_SIZE,), 
                  (indices, indicess, signss, k, n, control, target))

    return signss, indicess


def map_cx(lambdass, indicess, control, target):
    r"""
    --- First, I encode the n-qubit Pauli word as list of n int8 array
    
    Ex: XYI --> [1, 2, 0]
    
    For n-stabilizer, we have n x k words, so the encoded tensor will be n x k x n (ragged tensor)
    
    --- Next step, I flatten this tensor to 1D array (each element is still n-dim array)
    
    Flatten vector: [[0 1 2]
    [0 2 2]
    [0 3 2]
    [1 2 3]
    [2 3 1]
    [2 2 1]]
    
    --- Map this array to the new array using map_cx kernel
    
    Mapped flatten vector: [[0 1 2]
    [3 2 2]
    [3 3 2]
    [2 3 3]
    [1 2 1]
    [1 3 1]]
    
    Lambdas: [ 1  1  1  1  1 -1]
    
    --- Finally, I unflatten the mapped array to the original shape (ragged tensor)
    
    --- Obviously, this function requires starts variable (the start index of each row in the flatten vector)
    
    Out ragged tensor (with starts = [0,3,4]): 
    
    [array([[0, 1, 2],
        [3, 2, 2],
        [3, 3, 2]], dtype=int8), array([[2, 3, 3]], dtype=int8), array([[1, 2, 1],
        [1, 3, 1]], dtype=int8)]

    """

    def flatten_ragged_matrix_cupy(ragged_matrix):
        lengths = cp.array([len(row) for row in ragged_matrix])
        starts = cp.concatenate((cp.array([0]), cp.cumsum(lengths)))
        flatten_vector = cp.concatenate(ragged_matrix)
        return flatten_vector, starts[:-1]

    flatten_indicess, starts = flatten_ragged_matrix_cupy(indicess)
    flatten_lambdas_sign, mapped_flatten_indicess = cuda_map_cx(flatten_indicess, control, target)
    # print("mapped_flatten_indicess", mapped_flatten_indicess)
    # print("flatten_lambdas_sign", flatten_lambdas_sign)
    
    starts = starts[1:].tolist()
	# Convert flatten vector to ragged tensor
    ragged_indicess = cp.vsplit(mapped_flatten_indicess, starts)
    ragged_signss = cp.split(flatten_lambdas_sign, starts)
	# OP: lambdas_sign * ragged_lambdas
    # This operator can be implemented in CUDA kernel (in file notebook)
    # But I see there is no different between two methods
    return [cp.multiply(m1, m2) for m1, m2 in zip(lambdass, ragged_signss)], ragged_indicess



def map_indices_to_indicess(indicess):
    """
    --- First, I encode the n-qubit Pauli word (index) as list of n int8 array
    Ex: 0(III) --> [0, 0, 0]
    For n-stabilizer, we have n x k indicess, so the encoded tensor will be n x k x n (ragged tensor)
    Original tensor: [
		array([1, 2]),
		array([3]),
		array([4]),
	]
    --- Next step, I flatten this tensor to 1D array (each element is still n-dim array)
    Flatten vector: [1,2,3,4] and following starts (variable) = [0, 2, 3]
    --- Map this array to the new array using map_indices_to_weighted kernel
    Mapped flatten vector: [
        [0, 0, 1],
        [0, 0, 2],
        [0, 0, 3],
        [0, 1, 0],
    ] (n x k x n)
    
    --- Finally, I unflatten the mapped array to the original shape (ragged tensor)
    --- Obviously, this function requires starts variable (the start index of each row in the flatten vector)
    
    Output ragged tensor (with starts = [0, 2, 3]): [
        array([[0, 0, 1], [0, 0, 2]]),
		array([[0, 0, 3]]),
		array([[0, 1, 0]]),
    ]

    """
    def indices_to_indicess(indices, num_qubits):
        size = indices.size
        result = cp.empty((size, num_qubits), dtype=cp.int8)
        grid_size = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
        index_to_indices_kernel((grid_size,), (BLOCK_SIZE,), 
                            (indices, num_qubits, result, size))
        return result

    def flatten_ragged_matrix(ragged_matrix):
        lengths = cp.array([len(row) for row in ragged_matrix])
        starts = cp.concatenate((cp.array([0]), cp.cumsum(lengths)))
        flatten_vector = cp.concatenate(ragged_matrix)
        return flatten_vector, starts[:-1]
    num_qubits = len(indicess)
    flatten_vector, starts = flatten_ragged_matrix(indicess)
    mapped_flatten_vector = indices_to_indicess(flatten_vector, num_qubits)
    indicess = cp.vsplit(mapped_flatten_vector, starts[1:].tolist())
    return indicess