import numpy as np

# 一维的矩阵
A = np.arange(3, 15)
print(A[3])

# 将矩阵转换为二维的
A = np.arange(3, 15).reshape((3, 4))
print(A)
# 时的A[2]对应的就是矩阵A中第三行(从0开始算第一行)的所有元素
print(A[2])
print(A[1][1])
print(A[1, 1:3])

# 用for函数进行打
for row in A:
    print(row[1:3])

# 示方法即对A进行转置，再将得到的矩阵逐行输出即可得到原矩阵的逐列输出
for column in np.transpose(A):
    print(column)

# 这一脚本中的flatten是一个展开性质的函数，将多维的矩阵进行展开成1行的数列。
A = np.arange(3, 15).reshape((3, 4))
print(A.flatten())

# 而flat是一个迭代器，本身是一个object属性。
for item in A.flat:
    print(item)
