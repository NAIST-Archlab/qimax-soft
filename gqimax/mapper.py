import cupy as cp
from numpy import sin, cos, sqrt
from .utils import char_to_weight

def char_to_weight(character: str) -> cp.ndarray:
    if character == "i":
        return cp.array([1, 0, 0, 0], dtype=cp.float32)
    elif character == "x":
        return cp.array([0, 1, 0, 0], dtype=cp.float32)
    elif character == "y":
        return cp.array([0, 0, 1, 0], dtype=cp.float32)
    elif character == "z":
        return cp.array([0, 0, 0, 1], dtype=cp.float32)

# ------------------------------------- #
# ---- map_noncx to map_cx section ---- #
# ------------------------------------- #

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



# --------------------------- #
# ---- map_noncx section ---- #
# --------------------------- #


construct_lut_noncx_kernel = cp.RawKernel(r'''
extern "C" __global__
void construct_lut_noncx(const float* instructors_flat, const int* offsets, const int* num_instructors,
                         int K, int num_qubits, float* lut) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int total_elements = K * num_qubits * 3;
    
    if (idx < total_elements) {
        int k = idx / (num_qubits * 3);
        int j = (idx / 3) % num_qubits;
        int i = idx % 3;
        float weights[4];
        if (i == 0) {  // X
            weights[0] = 0.0f; weights[1] = 1.0f; weights[2] = 0.0f; weights[3] = 0.0f;
        } else if (i == 1) {  // Y
            weights[0] = 0.0f; weights[1] = 0.0f; weights[2] = 1.0f; weights[3] = 0.0f;
        } else {  // Z
            weights[0] = 0.0f; weights[1] = 0.0f; weights[2] = 0.0f; weights[3] = 1.0f;
        }
        
        int start_offset = offsets[k * num_qubits + j];
        int num_ins = num_instructors[k * num_qubits + j];
        
        for (int ins = 0; ins < num_ins; ins++) {
            int base_offset = start_offset + ins * 3;
            float gate_type = instructors_flat[base_offset];  
            float param = instructors_flat[base_offset + 2];  
            
            float I = weights[0], A = weights[1], B = weights[2], C = weights[3];
            if (gate_type == 0.0f) {  // h
                weights[0] = I; weights[1] = C; weights[2] = -B; weights[3] = A;
            } else if (gate_type == 1.0f) {  // s
                weights[0] = I; weights[1] = -B; weights[2] = A; weights[3] = C;
            } else if (gate_type == 2.0f) {  // t
                float sqrt2 = 1.41421356237f;
                weights[0] = I; 
                weights[1] = (A - B) / sqrt2; 
                weights[2] = (A + B) / sqrt2; 
                weights[3] = C;
            } else if (gate_type == 3.0f) {  // rx
                float cos_p = cosf(param), sin_p = sinf(param);
                weights[0] = I; 
                weights[1] = A; 
                weights[2] = B * cos_p - C * sin_p; 
                weights[3] = B * sin_p + C * cos_p;
            } else if (gate_type == 4.0f) {  // ry
                float cos_p = cosf(param), sin_p = sinf(param);
                weights[0] = I; 
                weights[1] = A * cos_p + C * sin_p; 
                weights[2] = B; 
                weights[3] = C * cos_p - A * sin_p;
            } else if (gate_type == 5.0f) {  // rz
                float cos_p = cosf(param), sin_p = sinf(param);
                weights[0] = I; 
                weights[1] = A * cos_p - B * sin_p; 
                weights[2] = B * cos_p + A * sin_p; 
                weights[3] = C;
            }
        }
        
        int lut_offset = (k * num_qubits * 3 + j * 3 + i) * 4;
        for (int w = 0; w < 4; w++) {
            lut[lut_offset + w] = weights[w];
        }
    }
}
''', 'construct_lut_noncx')

def construct_lut_noncx(grouped_instructorss, num_qubits: int):
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
    
    lut = cp.zeros((K, num_qubits, 3, 4), dtype=cp.float32)
    lut_flat = lut.reshape(K * num_qubits * 3 * 4)
    
    total_elements = K * num_qubits * 3
    block_size = 256
    grid_size = (total_elements + block_size - 1) // block_size
    
    construct_lut_noncx_kernel((grid_size,), (block_size,), 
                              (instructors_flat, offsets, num_instructors, K, num_qubits, lut_flat))
    
    return lut.reshape(K, num_qubits, 3, 4)






# --------------------------- #
# ----- map_cx section ------ #
# --------------------------- #

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

def map_cx(indices_array, control, target, n, max_memory_bytes=8*1024*1024*1024):
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


# def map_noncx(indices: cp.ndarray, lambdas: cp.ndarray, LUT, k: int, num_qubits: int):
#     """Given a list of indices and lambdas, return the indices and lambdas of the non-cx gates.
#     Example:
#         Initial: 1*IZ => indices = [3], lambdas = [1]
#         Another: 2*XZ + 3*YY => indices = [7, 10], lambdas = [2, 3]
#     Args:
#         indices (cp.ndarray)
#         lambdas (cp.ndarray)
#         num_qubits (int)
#         k: index of the operator
#     Returns:
#         cp.ndarray, cp.ndarray: mapped indices and lambdas
#     """
#     weightss = []
#     for index in indices:
#         weights = []
#         word = index_to_word(index, num_qubits)
#         for j, char in enumerate(word):
#             i_in_lut = char_to_index(char) - 1
#             if i_in_lut == -1:
#                 weights.append([1,0,0,0])
#             else: 
#                 weights.append(LUT[k][j][i_in_lut])
#         weightss.append(weights)
#     # Weightss's now a 4-d tensor, 4^n x n x 4
#     weightss = cp.array(weightss)
#     # Flattening the weightss into new lambdas, => Can be process effeciently on hardware
#     lambdas = weightss_to_lambda(weightss, lambdas)
#     # For most of the cases, lambdas will be sparse
#     # In the worst case, it will be full dense,
#     # and the below lines would be useless.
#     indices = cp.nonzero(lambdas)[0]
#     lambdas = lambdas[indices]
#     return indices, lambdas

# def map_cx(indices: cp.ndarray, lambdas: cp.ndarray, cxs: cp.ndarray, num_qubits: int):
#     """Mapping multiple CX gates on a given indices and lambdas

#     Args:
#         indices (cp.ndarray): Words after converting to integers
#         lambdas (cp.ndarray): Corresponding lambdas
#         cxs (cp.ndarray): List of [control, target] pairs from k^th CX-operator
#         num_qubits (int)

#     Returns:
#         indices, lambdas
#     """
#     for control, target in cxs:
#         df = pl.read_csv(f'./qimax/db/{num_qubits}_{control}_{target}_cx.csv')
#         out_array = cp.array(df['out'].to_numpy())
#         selected_rows = out_array[indices]
#         indices = cp.abs(selected_rows)
#         lambdas[cp.where(selected_rows < 0)[0]] *= -1
#     return indices, lambdas

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


