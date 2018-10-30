import numpy as np

A = np.array([1, 1, 1])
B = np.array([2, 2, 2])
print(A.shape)  # A仅仅是一个拥有3项元素的数组（数列）
# 可以想到按行、按列等多种方式进行合并
C = np.vstack((A, B))
print(C)
print(C.shape)  # 合并后得到的C是一个2行3列的矩阵

D = np.hstack((A, B))  # D本身来源于A，B两个数列的左右合并
print(D)
print(A.shape, D.shape)  # 而且新生成的D本身也是一个含有6项元素的序列。

##########
A = np.array([1, 1, 1])[:, np.newaxis]
B = np.array([2, 2, 2])[:, np.newaxis]

C = np.vstack((A, B))  # vertical stack
D = np.hstack((A, B))  # horizontal stack

print(C)
print(D)
"""
[[1 2]
[1 2]
[1 2]]
"""

print(A.shape, D.shape)

###所以  vstack 是上下数组合并，而hstack 是左右数据的合并


C = np.concatenate((A, B, B, A), axis=0)
print(C)

D = np.concatenate((A, B, B, A), axis=1)
print(D)
