class Instructor:
    def __init__(self, num_qubits):
        self.clusters = []
        self.cluster = []
        self.cluster_temp = []
        self.xcluster = []
        self.xclusters = []
        self.instructors = []
        self.num_qubits= num_qubits
        self.barriers = [0]*num_qubits

    def append(self, gate, index, param = 0):
        self.instructors.append([gate, index, param])
    def clustering(self):
        while(len(self.instructors)>0):
            gate, index, _ = self.instructors[0]
            sum_barrier = sum(self.barriers)
            if  sum_barrier >= self.num_qubits:
                self.clusters.append(self.cluster)
                self.cluster = []
                self.instructors = self.cluster_temp + self.instructors
                self.barriers = [0]*self.num_qubits
            if gate == 'cx':
                self.barriers[index[0]] += 1
                self.barriers[index[1]] += 1
                self.xcluster.append(self.instructors[0])
                self.instructors.pop(0)
            else:
                if len(self.xcluster) > 0 and sum_barrier >= self.num_qubits:
                    self.xclusters.append(self.xcluster)
                    self.xcluster = []
                if self.barriers[index] == 0:
                    self.cluster.append(self.instructors[0])
                    self.instructors.pop(0)
                else:
                    self.cluster_temp.append(self.instructors.pop(0))
ins = Instructor(3)
ins.append('h', 0)
ins.append('h', 1)
ins.append('h', 2)
ins.append('h', 0)
ins.append('cx', [0,1])
ins.append('h', 2)
ins.append('h', 2)
ins.append('h', 0)
ins.append('cx', [1,2])
ins.append('h', 1)
ins.clustering()