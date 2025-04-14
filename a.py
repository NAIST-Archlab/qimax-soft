from gqimax2.mapper import map_cx, map_noncx
from gqimax2.sample import sample1, sample2
from gqimax2.mapper import broadcasted_multiplies, broadcasted_multiplies_base4
import cupy as cp
import time
num_qubits = 4
ins = sample1(num_qubits)
ins.operatoring()
from gqimax2.pstabilizers import PStabilizers
stb = PStabilizers(num_qubits)
lambdass = stb.lambdass
indicesss = stb.indicess


def weightsss_to_lambdass(lambdass, weightsss, indicesss):
    num_qubits = len(weightsss)
    lambdass = [None] * num_qubits
    indicess = [None] * num_qubits
    for i in range(num_qubits):
        lambdass[i] = broadcasted_multiplies(lambdass[i], weightsss[i])
        indicess[i] = broadcasted_multiplies_base4(indicesss[i])
    return lambdass, indicess

def map_noncx(lambdass, indicesss, lut_at_k):
    weightsss = []
    for indicess in indicesss:
        weightss = []
        for qubit, indices in enumerate(indicess):
            weights = []
            for index in indices:
                if index == 0:
                    weights.append(cp.array([0], dtype=cp.float32))
                else: 
                    weights.append(lut_at_k[qubit][int(index) - 1])
            weightss.append(weights)
        weightsss.append(weightss)

    return weightsss_to_lambdass(lambdass, weightsss, indicesss)
for j, order in enumerate(ins.orders):
    k = j // 2
    if order == 0:
        print(indicesss)
        lambdass, indicesss = map_noncx(lambdass, indicesss, ins.lut[k])

    else:
        for _, indices, _ in ins.xoperators[k]:
            lambdass, indicesss = map_cx(lambdass, indicesss, indices[0], indices[1])