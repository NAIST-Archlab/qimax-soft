import numpy as np
class Instructor:
    def __init__(self, num_qubits):
        self.clusters = []
        self.cluster = []
        self.cluster_temp = []
        self.xcluster = []
        self.xcluster_temp = []
        self.xclusters = []
        self.instructors = []
        self.num_qubits = num_qubits
        self.barriers = [0] * self.num_qubits

    def append(self, gate, index, param=0):
        self.instructors.append([gate, index, param])

    def clustering(self):
        self.barriers = [0] * self.num_qubits
        while len(self.instructors) > 0:
            gate, index, _ = self.instructors[0]
            
            is_break = False
            if gate == "cx":
                self.barriers[index[0]] += 1
                self.barriers[index[1]] += 1
                if sum(self.barriers) >= self.num_qubits and np.all(self.barriers) and len(self.cluster) > 0:
                    self.cluster_temp.append(self.instructors.pop(0))
                    is_break = True
                else:
                    self.xcluster.append(self.instructors.pop(0))
            else:
                if self.barriers[index] == 0:
                    self.cluster.append(self.instructors.pop(0))
                else:
                    self.cluster_temp.append(self.instructors.pop(0))
            if is_break:
                self.clusters.append(self.cluster)
                self.instructors =  self.cluster_temp + self.instructors
                self.xclusters.append(self.xcluster)
                self.cluster = []
                self.cluster_temp = []
                self.xcluster = []
                self.barriers = [0] * self.num_qubits
                is_break = False
        if len(self.cluster) > 0:
            self.clusters.append(self.cluster)
        if len(self.cluster_temp) > 0:
            self.clusters.append(self.cluster_temp)
        if len(self.xcluster) > 0:
            self.xclusters.append(self.xcluster)
        return 
num_qubits = 3
ins = Instructor(num_qubits)
for i in range(num_qubits - 1):
    ins.append('cx', [i, i + 1])
ins.append('cx', [num_qubits - 1, 0])
for i in range(num_qubits):
    ins.append('h', i)
for i in range(num_qubits - 1):
    ins.append('cx', [i, i + 1])
ins.append('cx', [num_qubits - 1, 0])
for i in range(num_qubits):
    ins.append('h', i)
ins.clustering()