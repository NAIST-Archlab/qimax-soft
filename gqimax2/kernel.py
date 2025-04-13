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