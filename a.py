
from gqimax.mapper import map_cx, map_noncx
from gqimax.sample import sample2
from gqimax.mapper import weightss_to_lambdas, map_indices_to_indicess, map_noncx
import cupy as cp
import time

import cupy as cp

def weightsss_to_lambdass(lambdass: list, weightsss: list) -> cp.ndarray:
    n = len(weightsss)
    streams = [cp.cuda.Stream() for _ in range(n)]
    mapped_lambdass = [None] * n
    non_zeros_indicess = [None] * n
    for i in range(n):
        with streams[i]:
            new_lambdas, non_zeros_indices = weightss_to_lambdas(
                lambdass[i], weightsss[i]
            )
            mapped_lambdass[i] = new_lambdas
            non_zeros_indicess[i] = non_zeros_indices
    for stream in streams:
        stream.synchronize()
    
    return mapped_lambdass, map_indices_to_indicess(non_zeros_indicess)

num_qubits = 4
ins = sample2(num_qubits, 2)
ins.operatoring()
from gqimax.pstabilizers import PStabilizers
stb = PStabilizers(num_qubits)
lambdass = stb.lambdass
indicesss = stb.indicess

for j, order in enumerate(ins.orders):
    k = j // 2
    if order == 0:

        lambdass, indicesss = map_noncx(lambdass, indicesss, ins.lut[k])
        print("noncx", indicesss[0][:5])
    else:

        for _, indices, _ in ins.xoperators[k]:
            lambdass, indicesss = map_cx(lambdass, indicesss, indices[0], indices[1])
        print("cx", indicesss[0][:5])
