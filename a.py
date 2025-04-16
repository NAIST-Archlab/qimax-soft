from gqimax2.mapper import broadcasted_multiplies, broadcasted_multiplies_base4, sum_distributions
from gqimax2.mapper import map_cx
from gqimax2.sample import sample1, sample2
import cupy as cp
import time
from gqimax2.mapper import broadcasted_multiplies, broadcasted_multiplies_base4, sum_distributions
from gqimax2.mapper import broadcasted_multiplies, broadcasted_base4, sum_distributions
num_qubits = 4
# ins = sample3()
def create_word_zj(num_qubits, j):
	lst = cp.zeros(num_qubits, dtype=cp.int32)
	lst[j] = 3
	return lst
def create_word_ii(num_qubits):
	lst = [cp.array([0], dtype=cp.int32) for _ in range(num_qubits)]
	return lst

ins = sample2(num_qubits, 3)
ins.operatoring()
from gqimax2.pstabilizers import PStabilizers


def weightsss_to_lambdass(lambdass, weightsss, indicesss):
    num_qubits = len(weightsss)
    lambdass_res = [None] * num_qubits
    indicess_res = [None] * num_qubits
    # print('lambda: ', lambdass)
    # print('index: ', indicesss)
    # print('weight: ', weightsss)
    for i in range(num_qubits):
        print(f'Flattening at {i} ...')
        print('(Input) weightsss[i]: ', weightsss[i])
        print('(Input) lambdass[i]: ', lambdass[i])
        print('(Input) indicesss[i]: ', indicesss[i])
        a = broadcasted_multiplies(lambdass[i], weightsss[i])
        b = broadcasted_base4(indicesss[i])
        print('(Output) weights: ', a)
        print('(Output) Index:', b)
        lambdass_res[i], indicess_res[i] = sum_distributions(
            a,b 
            )
        print(f"lambdass_res[{i}]: {lambdass_res[i]}")
        print(f"indicess_res[{i}]: {indicess_res[i]}")
    return lambdass_res, indicess_res

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
                print(f"k: {k}, qubit: {qubit}, index: {index}")
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
        print(f'stabilizer: {j}, term: {k}')
        print(f"weightsss: {weightss}")
        print(f"indicesss_out: {indicess_out}")
    return weightsss_to_lambdass(lambdass, weightsss, indicesss_out)


lambdass = [
    cp.array([1], dtype=cp.float32) 
    for _ in range(num_qubits)
]

indicesss = [
	[
		create_word_zj(num_qubits,j),
	]
 	for j in range(num_qubits)
]

for j, order in enumerate(ins.orders):
    k = j // 2
    if order == 0:
        lambdass, indicesss = map_noncx(lambdass, indicesss, ins.lut[k], ins.indicesss[k])
    else:
        for _, cnot_indices, _ in ins.xoperators[k]:
            print('lambdass: ', lambdass)
            print('indicesss: ', indicesss)
            lambdass, indicesss = map_cx(lambdass, indicesss, cnot_indices[0], cnot_indices[1])

print("Final")
print('lambdass: ', lambdass)
print('indicesss: ', indicesss)