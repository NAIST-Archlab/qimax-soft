import cupy as cp
from .kernel import broadcasted_multiplies_kernel, broandcast_base4_kernel, sum_distributions_kernel, construct_lut_noncx_kernel, map_cx_kernel, index_to_indices_kernel
from .constant import BLOCK_SIZE



#################################################################
# ---- map_noncx to map_cx section (weightsss_to_lambdass) ---- #
#################################################################

def broadcasted_multiplies(lambdas, arrayss):
    """
    Returns:
        weightss (k x ?): k arrays of weights
    """
    def prepare(arrayss):
        k = len(arrayss)
        dims = len(arrayss[0])
        shapes = cp.array([len(arr) for arrays in arrayss for arr in arrays], dtype=cp.int32)
        data = cp.concatenate([arr for arrays in arrayss for arr in arrays])
        lengths = shapes.reshape(k, dims)
        offsets = cp.cumsum(cp.concatenate([cp.array([0], dtype=cp.int32), lengths.flatten()[:-1]]))
        sizes = cp.prod(lengths.reshape(k, dims), axis=1)
        offsets_result = cp.cumsum(cp.concatenate([cp.array([0], dtype=cp.int32), sizes[:-1]]))
        return shapes, data, offsets, offsets_result
    
    # - shapes: cp.ndarray (N * dims,) - save all k_i
    # - data: cp.ndarray - Flattened array of all arrays in arrayss
    # - offsets: cp.ndarray (N * dims,) - Beginning indices
    # - offsets_result: cp.ndarray (N,) - Beginning indices of results
    shapes, data, offsets, offsets_result = prepare(arrayss)
    num_qubits = lambdas.size
    dims = shapes.size // num_qubits
    total_size = int(offsets_result[-1] + cp.prod(shapes[-dims:]))
    result_cp = cp.zeros(total_size, dtype=cp.float32)
    grid_size = (total_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    broadcasted_multiplies_kernel(
        (grid_size,), (BLOCK_SIZE,),
        (lambdas, shapes, data, offsets, offsets_result, result_cp, num_qubits, dims, total_size)
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

def sum_distributions(weightss, indicess):
    r'''
    After broadcasting k term, we have k arrays of weights and k arrays of indices, then we need to sum them up
    Example: weightss = [1,2,3] [1,2,3] indicess = [0,2,3] [1,2,3] => new_weights = [1,1,4,6], indicess = [0,1,2,3]
    '''
    from gqimax2.constant import BLOCK_SIZE
    # Flattening all data
    flatten_indicess = cp.concatenate([cp.array(indices, dtype=cp.int32) for indices in indicess])
    flatten_weightss = cp.concatenate([cp.array(weights, dtype=cp.float32) for weights in weightss])
    n_elements = flatten_indicess.size
    if n_elements != flatten_weightss.size:
        raise ValueError(r'''
            These sizes from weights and indicess do not match!
            Make sure, example: weightss = [1,2,3] [1,2,3] indicess = [1,2,3] [1,2,3] is ok
            Each weight has a corresponding position.
        ''')
	# Below is from AI, I dont know what it means
    max_pos = int(cp.max(flatten_indicess))
    output = cp.zeros(max_pos + 1, dtype=cp.float32)
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    sum_distributions_kernel((grid_size,), (BLOCK_SIZE,),
                             (flatten_indicess, flatten_weightss, output, n_elements))

    non_zero_pos = cp.nonzero(output)[0]
    non_zero_values = output[non_zero_pos]
    return non_zero_pos, non_zero_values


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
                    indicesss[i][j][k] = cp.array([w + 1 for w in non_zero_indices])
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






def map_noncx(lambdass, indicesss, lut_at_k):
    weightsss = []
    for arr in indicesss:
        H, W = arr.shape
        j_indices = cp.tile(cp.arange(W), (H, 1)) 
        values = arr
        out = cp.empty((H, W, 3), dtype=cp.float32)
        mask0 = (values == 0) # Mask for I
        out[mask0] = cp.array([0], dtype=cp.float32)
        mask123 = ~mask0 # Mask for X,Y,Z
        j_valid = j_indices[mask123]
        val_valid = values[mask123] - 1
        out[mask123] = lut_at_k[j_valid, val_valid]
        weightsss.append(out)
    lambdass, indicesss = weightsss_to_lambdass(lambdass, weightsss)
    return lambdass, indicesss




# # --------------------------- #
# # ----- map_cx section ------ #
# # --------------------------- #


def cuda_map_cx(indices, control, target):
    if indices.dtype != cp.int8:
        raise TypeError("Indices must be of type int8")
    
    k, n = indices.shape
    indicess = cp.empty_like(indices, dtype=cp.int8)
    lambdas = cp.empty(k, dtype=cp.int8)
    grid_size = (k + BLOCK_SIZE - 1) // BLOCK_SIZE
    map_cx_kernel((grid_size,), (BLOCK_SIZE,), 
                  (indices, indicess, lambdas, k, n, control, target))

    return lambdas, indicess


def map_cx(lambdass, indicess, control, target):
    """
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
    
    Out ragged tensor (with starts = [0,3,4]): [array([[0, 1, 2],
        [3, 2, 2],
        [3, 3, 2]], dtype=int8), array([[2, 3, 3]], dtype=int8), array([[1, 2, 1],
        [1, 3, 1]], dtype=int8)]

    """

    def flatten_ragged_matrix_cupy(ragged_matrix):
        lengths = cp.array([len(row) for row in ragged_matrix])
        starts = cp.concatenate((cp.array([0]), cp.cumsum(lengths)))
        flatten_vector = cp.concatenate(ragged_matrix, dtype=cp.int8)
        return flatten_vector, starts[:-1]

    flatten_indicess, starts = flatten_ragged_matrix_cupy(indicess)
    flatten_lambdas_sign, mapped_flatten_indicess = cuda_map_cx(flatten_indicess, control, target)
    # print("mapped_flatten_indicess", mapped_flatten_indicess)
    # print("flatten_lambdas_sign", flatten_lambdas_sign)
    
    starts = starts[1:].tolist()
	# Convert flatten vector to ragged tensor
    ragged_indicess = cp.vsplit(mapped_flatten_indicess, starts)
    ragged_lambdas_sign = cp.split(flatten_lambdas_sign, starts)
	# OP: lambdas_sign * ragged_lambdas
    # This operator can be implemented in CUDA kernel (in file notebook)
    # But I see there is no different between two methods
    return [cp.multiply(m1, m2) for m1, m2 in zip(lambdass, ragged_lambdas_sign)], ragged_indicess



def map_indices_to_indicess(indicess):
    """
    --- First, I encode the n-qubit Pauli word (index) as list of n int8 array
    Ex: 0(III) --> [0, 0, 0]
    For n-stabilizer, we have n x k indices, so the encoded tensor will be n x k x n (ragged tensor)
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