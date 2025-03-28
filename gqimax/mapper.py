import cupy as cp


def mapper_noncx(character: str, instructors: list):
    """Map a single Pauliword to list by multiple instructors
    Related to construct_LUT_noncx.
    Example: X -> [0, 1, 0, 0] -- h --> [0, 0, -1, 0] = -Y
    Args:
        character (str): I, X, Y or Z
        instructors (list)
    """
    weights = char_to_weight(character)
    for gate, _, param in instructors:
        I, A, B, C = weights
        if gate == "h":
            weights = cp.array([I, C, -B, A])
        if gate == "s":
            weights = cp.array([I, -B, A, C])
        if gate == "t":
            weights = cp.array([I, (A - B) / sqrt(2), (A + B) / sqrt(2), C])
        if gate == "rx":
            weights = cp.array(
                [I, A, B * cos(param) - C * sin(param), B * sin(param) + C * cos(param)]
            )
        if gate == "ry":
            weights = cp.array(
                [I, A * cos(param) + C * sin(param), B, C * cos(param) - A * sin(param)]
            )
        if gate == "rz":
            weights = cp.array(
                [I, A * cos(param) - B * sin(param), B * cos(param) + A * sin(param), C]
            )
    return weights




def construct_lut_noncx(grouped_instructorss, num_qubits: int):
    """grouped_instructorss has size k x n x [?], with k is number of non-cx layer, n is number of qubits,
    ? is the number of instructor (depend on each operator).
    lut has size k x n x 3 x 4, with 3 is the number of Pauli (ignore I), 4 for weights
    Args:
        grouped_instructorss (_type_): group by qubits
        num_qubits (int): _description_

    Returns:
        _type_: _description_
    """
    K = len(grouped_instructorss)
    lut = cp.zeros((K, num_qubits, 3, 4))
    characters = ["x", "y", "z"] # Ignore I because [?]I = I
    for k in range(K):
        for j in range(num_qubits):
            for i in range(3):
                lut[k][j][i] = mapper_noncx(characters[i], grouped_instructorss[k][j])
    return lut




def weightss_to_lambda(weightss: cp.ndarray, lambdas: cp.ndarray) -> cp.ndarray:
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
    num_terms, num_qubits, _ = weightss.shape
    new_lambdas = cp.zeros(4**num_qubits)
    for j in range(num_terms):
        weights = weightss[j]
        products = weights[0]
        for k in range(1, num_qubits):
            products = cp.outer(products, weights[k]).ravel()
        new_lambdas += lambdas[j] * products
    # This lambdas is still in the form of 4^n x 1, 
    # we need to ignore 0 values in the next steps
    # In the worst case, there is no 0 values.
    return new_lambdas
















# I accelerate the CNOT gate on GPU using cupy
# With indices_array is a list of integers representing the Pauli words
# control and target are the indices of the control and target qubits
map_cx_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void map_cx(int* indices_array, int control, int target, int num_qubits, int* lambdas_array, int num_words) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_words) {
            int word_int = indices_array[idx];
            
            auto get_digit = [](int word_int, int k, int num_qubits) {
                return (word_int >> (2 * (num_qubits - 1 - k))) & 3;
            };
            
            auto set_digit = [](int word_int, int k, int new_digit, int num_qubits) {
                int shift = 2 * (num_qubits - 1 - k);
                word_int &= ~(3 << shift);  
                word_int |= (new_digit << shift);  
                return word_int;
            };
            
            int char_control = get_digit(word_int, control, num_qubits);
            int char_target = get_digit(word_int, target, num_qubits);
            
            int new_control = char_control;
            int new_target = char_target;

            
            if (char_control == 0) {  // 'i'
                if (char_target == 2) {  // 'y'
                    new_control = 3;  // 'z'
                    new_target = 2;   // 'y'
                } else if (char_target == 3) {  // 'z'
                    new_control = 3;  // 'z'
                    new_target = 3;   // 'z'
                }
            } else if (char_control == 1) {  // 'x'
                if (char_target == 0) {  // 'i'
                    new_control = 1;  // 'x'
                    new_target = 1;   // 'x'
                } else if (char_target == 1) {  // 'x'
                    new_control = 1;  // 'x'
                    new_target = 0;   // 'i'
                } else if (char_target == 2) {  // 'y'
                    new_control = 2;  // 'y'
                    new_target = 3;   // 'z'
                } else if (char_target == 3) {  // 'z'
                    lambdas_array[idx] = -1;
                    new_control = 2;  // 'y'
                    new_target = 2;   // 'y'
                }
            } else if (char_control == 2) {  // 'y'
                if (char_target == 0) {  // 'i'
                    new_control = 2;  // 'y'
                    new_target = 1;   // 'x'
                } else if (char_target == 1) {  // 'x'
                    new_control = 2;  // 'y'
                    new_target = 2;   // 'y'
                } else if (char_target == 2) {  // 'y'
                    new_control = 1;  // 'x'
                    new_target = 3;   // 'z'
                    lambdas_array[idx] = -1;
                } else if (char_target == 3) {  // 'z'
                    new_control = 1;  // 'x'
                    new_target = 2;   // 'y'
                }
            } else if (char_control == 3) {  // 'z'
                if (char_target == 2) {  // 'y'
                    new_control = 0;  // 'i'
                    new_target = 2;   // 'y'
                } else if (char_target == 3) {  // 'z'
                    new_control = 0;  // 'i'
                    new_target = 3;   // 'z'
                }
            }
            
            int new_word_int = set_digit(word_int, control, new_control, num_qubits);
            new_word_int = set_digit(new_word_int, target, new_target, num_qubits);
            
            indices_array[idx] = new_word_int;
        }
    }
''', 'map_cx')

def map_cx_big(indices_array, control, target, n, max_memory_bytes=8*1024*1024*1024):
    bytes_per_element = 4*12
    max_elements_per_chunk = max_memory_bytes // bytes_per_element
    
    num_words = len(indices_array)
    chunk_size = min(max_elements_per_chunk, num_words)
    
    num_chunks = (num_words + chunk_size - 1) // chunk_size

    result_full = cp.zeros(num_words, dtype=cp.int32)
    lambdas_full = cp.zeros(num_words, dtype=cp.int32)
    block_size = 256
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_words)
        chunk_words = end_idx - start_idx
        
        word_int_chunk = cp.array(indices_array[start_idx:end_idx], dtype=cp.int32)
        result_chunk = cp.zeros(chunk_words, dtype=cp.int32)
        lambdas_chunk = cp.zeros(chunk_words, dtype=cp.int32)
        
        grid_size = (chunk_words + block_size - 1) // block_size
        

        map_cx_kernel((grid_size,), (block_size,), 
                         (word_int_chunk, control, target, n, result_chunk, lambdas_chunk, chunk_words))

        result_full[start_idx:end_idx] = result_chunk
        lambdas_full[start_idx:end_idx] = lambdas_chunk
    
    return result_full, lambdas_full





# ------------------------------------- #
# ----- GPU version (not effecifient) - #
# ------------------------------------- #


# def get_digit(index, k, n):
#     return (index >> (2 * (n - 1 - k))) & 3

# def set_digit(index: int, k: int, new_digit: int, n: int) -> int:
#     shift = 2 * (n - 1 - k)
#     index &= ~(3 << shift)  # Xóa 2 bit tại vị trí k
#     index |= (new_digit << shift)  # Đặt giá trị mới
#     return index

# def apply_cnot(index, control, target, n):
#     char_control = get_digit(index, control, n)
#     char_target = get_digit(index, target, n)
    
#     lambdas = 1 
    
#     if char_control == 0:  # 'i'
#         if char_target == 2:  # 'y'
#             new_control = 3  # 'z'
#             new_target = 2   # 'y'
#         elif char_target == 3:  # 'z'
#             new_control = 3  # 'z'
#             new_target = 3   # 'z'
#         else:
#             new_control = char_control
#             new_target = char_target
#     elif char_control == 1:  # 'x'
#         if char_target == 0:  # 'i'
#             new_control = 1  # 'x'
#             new_target = 1   # 'x'
#         elif char_target == 1:  # 'x'
#             new_control = 1  # 'x'
#             new_target = 0   # 'i'
#         elif char_target == 2:  # 'y'
#             new_control = 2  # 'y'
#             new_target = 3   # 'z'
#         elif char_target == 3:  # 'z'
#             lambdas = -1
#             new_control = 2  # 'y'
#             new_target = 2   # 'y'
#     elif char_control == 2:  # 'y'
#         if char_target == 0:  # 'i'
#             new_control = 2  # 'y'
#             new_target = 1   # 'x'
#         elif char_target == 1:  # 'x'
#             new_control = 2  # 'y'
#             new_target = 2   # 'y'
#         elif char_target == 2:  # 'y'
#             new_control = 1  # 'x'
#             new_target = 3   # 'z'
#             lambdas = -1
#         elif char_target == 3:  # 'z'
#             new_control = 1  # 'x'
#             new_target = 2   # 'y'
#     elif char_control == 3:  # 'z'
#         if char_target == 2:  # 'y'
#             new_control = 0  # 'i'
#             new_target = 2   # 'y'
#         elif char_target == 3:  # 'z'
#             new_control = 0  # 'i'
#             new_target = 3   # 'z'
#         else:
#             new_control = char_control
#             new_target = char_target
    
#     # Cập nhật số nguyên
#     new_index = set_digit(index, control, new_control, n)
#     new_index = set_digit(new_index, target, new_target, n)
    
#     return new_index, lambdas

# for i in range(0, 4**12):
# 	(apply_cnot(i, 0, 3, 10))


