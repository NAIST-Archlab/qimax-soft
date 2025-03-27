from .pstabilizer import PStabilizer
from .instructor import Instructor
class PStabilizerGroup():
    def init(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.stabilizers = []*num_qubits
        for j in range(num_qubits):
            # Example: if j = 0, initialize stabilizer = Z0
            self.stabilizers.append(PStabilizer(j, num_qubits))
        return
    def map(self, ins: Instructor):
        for stabilizer in self.stabilizers:
            stabilizer.map(ins)
        return
    def generate_group(self):
        pass
    