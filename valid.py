from sojo.stabilizer import StabilizerGenerator, PauliTerm, PauliWord
import numpy as np

# Same as qiskit
stb = StabilizerGenerator(2)
stb.ry(np.pi/3, 0)
stb.rx(np.pi/2, 0)
stb.cx([0,1])
stb.t(0)
stb.cx([0,1])
stb.h(0)
stb.rz(np.pi/4, 0)
stb.rx(np.pi/3, 1)
for s in (stb.stabilizers):
    print(s)
# stb.generate_p()
# for p in (stb.ps):
#     print(p)
stb_density_matrix = stb.generate_density_matrix_by_generator_jax()
print(np.round(stb_density_matrix,2))