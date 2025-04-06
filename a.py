from gqimax.instructor import Instructor
from gqimax.mapper import map_cx, map_noncx
import numpy as np
from gqimax.sample import sample2

import time


num_qubits = 4
ins = sample2(num_qubits, 20)
ins.operatoring()
from gqimax.pstabilizers import PStabilizers
stb = PStabilizers(num_qubits)
lambdass = stb.lambdass
indicesss = stb.indicess



for j, order in enumerate(ins.orders):
    k = j // 2
    if order == 0:
        lambdass, indicesss = map_noncx(lambdass, indicesss, ins.lut[k])
    else:
        
        for _, indices, _ in ins.xoperators[k]:
            lambdass, indicesss = map_cx(lambdass, indicesss, indices[0], indices[1])
# print("Lambda", lambdass[0][:10])
# print("Indices", indicesss[0][:10])