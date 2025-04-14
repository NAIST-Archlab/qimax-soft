import cupy as cp


###############################################
#### weightss_to_lambdas section ##############
###############################################
broadcasted_multiplies_kernel = cp.RawKernel(r'''
	extern "C" __global__
	void broadcasted_multiplies_kernel(
		const float* lambdas,     
		const int* shapes,        
		const float* data,        
		const int* offsets,       
		const int* offsets_result, 
		float* result,             
		int N,                     
		int dims,                  
		int total_size             
	) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total_size) return;

		int i = 0;
		while (i < N - 1 && idx >= offsets_result[i + 1]) i++;
		int local_idx = idx - offsets_result[i];

		int temp = local_idx;
		int indices[4]; 
		for (int j = dims - 1; j >= 0; j--) {
			int s = shapes[i * dims + j];
			indices[j] = temp % s;
			temp /= s;
		}

		float val = lambdas[i];
		for (int j = 0; j < dims; j++) {
			int offset = offsets[i * dims + j];
			int index = indices[j];
			val *= data[offset + index];
		}
		result[idx] = val;
	}
''', 'broadcasted_multiplies_kernel')

broandcast_base4_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void broandcast_base4_kernel(
        const int* shapes,        
        const int* data,          
        const int* offsets,        
        const int* offsets_result, 
        const int* powers,         
        int* output,               
        int N,                    
        int n_dim,               
        int total_size             
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_size) return;

        int i = 0;
        while (i < N - 1 && idx >= offsets_result[i + 1]) i++;
        int local_idx = idx - offsets_result[i];

        int temp = local_idx;
        int indices[3]; 
        for (int j = n_dim - 1; j >= 0; j--) {
            int s = shapes[i * n_dim + j];
            indices[j] = temp % s;
            temp /= s;
        }

        int acc = 0;
        for (int j = 0; j < n_dim; j++) {
            int offset = offsets[i * n_dim + j];
            int index = indices[j];
            acc += data[offset + index] * powers[j];
        }
        output[idx] = acc;
    }
''', 'broandcast_base4_kernel')

sum_distributions_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void sum_distributions(
        const int* positions,    
        const float* values,   
        float* output,         
        int n_elements          
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_elements) return;

        int pos = positions[idx];
        float val = values[idx];
        atomicAdd(&output[pos], val);
    }
''', 'sum_distributions')

###############################################
#### map-noncx section ##############
###############################################
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
        float weights[3];  // [A, B, C]

        if (i == 0) {  
            weights[0] = 1.0f; weights[1] = 0.0f; weights[2] = 0.0f; // X
        } else if (i == 1) {  
            weights[0] = 0.0f; weights[1] = 1.0f; weights[2] = 0.0f; // Y
        } else {  
            weights[0] = 0.0f; weights[1] = 0.0f; weights[2] = 1.0f; // Z
        }

        int start_offset = offsets[k * num_qubits + j];
        int num_ins = num_instructors[k * num_qubits + j];

        for (int ins = 0; ins < num_ins; ins++) {
            int base_offset = start_offset + ins * 3;
            float gate_type = instructors_flat[base_offset];  
            float param = instructors_flat[base_offset + 2];  

            float A = weights[0], B = weights[1], C = weights[2];
            if (gate_type == 0.0f) {  // h
                weights[0] = C;  weights[1] = -B; weights[2] = A;
            } else if (gate_type == 1.0f) {  // s
                weights[0] = -B; weights[1] = A;  weights[2] = C;
            } else if (gate_type == 2.0f) {  // t
                float sqrt2 = 1.41421356237f;
                weights[0] = (A - B) / sqrt2;
                weights[1] = (A + B) / sqrt2;
                weights[2] = C;
            } else if (gate_type == 3.0f) {  // rx
                float cos_p = cosf(param), sin_p = sinf(param);
                weights[0] = A;
                weights[1] = B * cos_p - C * sin_p;
                weights[2] = B * sin_p + C * cos_p;
            } else if (gate_type == 4.0f) {  // ry
                float cos_p = cosf(param), sin_p = sinf(param);
                weights[0] = A * cos_p + C * sin_p;
                weights[1] = B;
                weights[2] = C * cos_p - A * sin_p;
            } else if (gate_type == 5.0f) {  // rz
                float cos_p = cosf(param), sin_p = sinf(param);
                weights[0] = A * cos_p - B * sin_p;
                weights[1] = B * cos_p + A * sin_p;
                weights[2] = C;
            }
        }

        int lut_offset = (k * num_qubits * 3 + j * 3 + i) * 3;
        for (int w = 0; w < 3; w++) {
            lut[lut_offset + w] = weights[w];
        }
    }
}
''', 'construct_lut_noncx')

###############################################
#### map-cx section ##############
###############################################


map_cx_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void map_cx_kernel(
        const char* words_array,  
        char* new_words_array,     
        char* lambdas,            
        int k,                    
        int n,              
        int control,                
        int target                 
    ) {
        // Lookup tables
        const char new_control_table[4][4] = {
            {0, 0, 3, 3},
            {1, 1, 2, 2},
            {2, 2, 1, 1},
            {3, 3, 0, 0}
        };
        const char new_target_table[4][4] = {
            {0, 1, 2, 3},
            {1, 0, 3, 2},
            {1, 0, 3, 2},
            {0, 1, 2, 3}
        };
        const char lambda_table[4][4] = {
            {1, 1, 1, 1},
            {1, 1, 1, -1},
            {1, 1, -1, 1},
            {1, 1, 1, 1}
        };

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= k) return;

        int offset = idx * n;

        for (int i = 0; i < n; i++) {
            new_words_array[offset + i] = words_array[offset + i];
        }

        char control_val = words_array[offset + control];
        char target_val = words_array[offset + target];

        new_words_array[offset + control] = new_control_table[control_val][target_val];
        new_words_array[offset + target] = new_target_table[control_val][target_val];
        lambdas[idx] = lambda_table[control_val][target_val];
    }
''', 'map_cx_kernel')


# # -------------------------------------------- #
# # ----- map_indices_to_indicess section ------ #
# # -------------------------------------------- #

index_to_indices_kernel = cp.RawKernel(r'''
extern "C" __global__
    void index_to_indices(const long long* indices, int num_qubits, char* result, int size) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < size) {
            long long index = indices[tid];
            for (int q = 0; q < num_qubits; q++) {
                long long divisor = 1LL << (2 * (num_qubits - 1 - q));
                char digit = (index / divisor) % 4;
                result[tid * num_qubits + q] = digit;
            }
        }
    }
''', 'index_to_indices')