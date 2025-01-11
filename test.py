from sojo.pc import PauliComposer
from sojo.stabilizer import PauliWord
words = [PauliWord(0.6123724356957946, 'xi'),
         PauliWord(0.7209158734973699, 'zi'),
         PauliWord(-0.32446926409064364, 'yi'),
]
matrix = words[0].to_pc().to_sparse()
print(matrix.toarray())

print(words[0].to_matrix())
# for word in words[1:]:
#     matrix = matrix + word.to_pc().to_sparse()
# print(matrix.toarray())