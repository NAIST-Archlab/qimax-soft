

from collections import deque
class Instructor:
    def __init__(self, num_qubits):
        self.operators = []
        self.operator = []
        self.operator_temp = []
        self.xoperator = []
        self.xoperators = []
        self.instructors = deque()  # Better than list for pop(0)
        self.num_qubits = num_qubits
        self.is_cx_first = False
        self.orders = []
        self.LUT = None

    def append(self, gate, index, param=0):
        self.instructors.append((gate, index, param))

    def operatoring(self):
        if self.instructors and self.instructors[0][0] == "cx":
            self.is_cx_first = True
        
        involved_qubits = set()  
        while len(self.instructors) > 0:
            gate, index, _ = self.instructors[0]
            is_break = False
            
            if gate == "cx":
                involved_qubits.add(index[0])
                involved_qubits.add(index[1])
                self.xoperator.append(self.instructors.popleft()[1])  
                if (len(involved_qubits) == self.num_qubits and 
                    len(self.instructors) > 0 and 
                    self.instructors[0][0] != "cx"):
                    is_break = True
            else:
                if index not in involved_qubits:  
                    self.operator.append(self.instructors.popleft())
                else:
                    self.operator_temp.append(self.instructors.popleft())
            
            if is_break:
                if len(self.operator) > 0:
                    self.operators.append(self.operator)
                if len(self.xoperator) > 0:
                    print("xoperator", self.xoperator)
                    self.xoperators.append(self.xoperator)
                self.instructors.extendleft(self.operator_temp[::-1])
                self.operator = []
                self.operator_temp = []
                self.xoperator = []
                involved_qubits = set() 
        
        # Take care of the last operators
        if len(self.operator) > 0:
            self.operators.append(self.operator)
        if len(self.operator_temp) > 0:
            self.operators.append(self.operator_temp)
        if len(self.xoperator) > 0:
            self.xoperators.append(self.xoperator)
        def create_zip_chain(num_operators, num_xoperators, is_cx_first):
            """Create list 0,1,0,1,...
            If is_cx_first is True, then 1 is first, else 0 is first
            Args:
                n (_type_): _description_
                m (_type_): _description_
                is_cx_first (bool): _description_

            Returns:
                _type_: _description_
            """
            result = []
            while num_operators > 0 or num_xoperators > 0:
                if is_cx_first:
                    if num_xoperators > 0:
                        result.append(1)
                        num_xoperators -= 1
                    if num_operators > 0:
                        result.append(0)
                        num_operators -= 1
                else:   
                    if num_operators > 0:
                        result.append(0)
                        num_operators -= 1
                    if num_xoperators > 0:
                        result.append(1)
                        num_xoperators -= 1
            return result


        self.orders = create_zip_chain(len(self.operators), len(self.xoperators), self.is_cx_first)
        return
    

def instructor_to_lut(ins: Instructor):
    """First, diving instructors into k non-cx operators and k+1/k-1/k cx-operator,
    the, utilizing the lut (size k x n x 4 x 4)"""
    grouped_instructorss = group_instructorss_by_qubits(ins.operators, ins.num_qubits)
    LUT = construct_lut_noncx(grouped_instructorss, ins.num_qubits)
    return LUT

def group_instructorss_by_qubits(instructors: list, num_qubits: int) -> list:
    """Group instructors by qubits
    Example: [['h', 0, 0], ['rx', 1, 0], ['h', 1, 0], ['ry', 0, 0]]
    -> [[['h', 0, 0], ['ry', 0, 0]], [['h', 1, 0], ['rx', 1, 0]]]

    Args:
        instructors (list): list of instructors
        num_qubits (int)

    Returns:
        list of list of n instructors: _description_
    """
    grouped_instructors = []
    for sublist in instructors:
        groups = {i: [] for i in range(num_qubits)}
        for instructor in sublist:
            index = instructor[1]
            groups[index].append(instructor)
        grouped_instructors.append([groups[i] for i in range(num_qubits)])
    return grouped_instructors

