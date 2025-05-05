from .pstabilizer import index_to_word, word_to_index


def map_cx(index, control, target, num_qubits):  
    word = list(index_to_word(index, num_qubits))
    original = word.copy()
    char_control = word[control]
    char_target = word[target]
    lambdas = 1
    if char_control == 'i':
        if char_target == 'y':
            word[control] = 'z'
            word[target] = 'y'
        elif char_target == 'z':
            word[control] = 'z'
            word[target] = 'z'
    elif char_control == 'x':
        if char_target == 'i':
            word[control] = 'x'
            word[target] = 'x'
        elif char_target == 'x':
            word[control] = 'x'
            word[target] = 'i'
        elif char_target == 'y':
            word[control] = 'y'
            word[target] = 'z'
        elif char_target == 'z':
            lambdas = -1
            word[control] = 'y'
            word[target] = 'y'
    elif char_control == 'y':
        if char_target == 'i':
            word[control] = 'y'
            word[target] = 'x'
        elif char_target == 'x':
            word[control] = 'y'
            word[target] = 'y'
        elif char_target == 'y':
            word[control] = 'x'
            word[target] = 'z'
            lambdas = -1
        elif char_target == 'z':
            word[control] = 'x'
            word[target] = 'y'
    elif char_control == 'z':
        if char_target == 'y':
            word[control] = 'i'
            word[target] = 'y'
        elif char_target == 'z':
            word[control] = 'i'
            word[target] = 'z'
    return word_to_index(word)*lambdas


# import pandas as pd
# num_qubits = 5

# for control in range(num_qubits):
#     for target in range(num_qubits):
#         if control == target:
#             continue
#         else:
#             ins, outs = [], []
#             for index in range(4**num_qubits):
                
#                 out_index = map_cx(index, control, target, num_qubits)
#                 ins.append(index)
#                 outs.append(out_index)
#             df = pd.DataFrame({'in': ins, 'out': outs})
#             df.to_csv(f'./qimax/db/{num_qubits}_{control}_{target}_cx.csv', index=False)