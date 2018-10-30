import numpy as np

# 创建数据 建立3行4列的Array
A = np.arange(12).reshape((3, 4))
print(A)

##纵向分割  中间的参数是切出2个数组
a = np.split(A, 2, axis=1)
print(a)

d = np.split(A, 1, axis=1)
print(d)

e = np.split(A, 4, axis=1)
print(e)

# 横向分割 中间的参数是切出3个数组
b = np.split(A, 3, axis=0)
print(b)

c = np.split(A, 1, axis=0)  # 中间的参数是切出1个数组
print(c)

###不等量的分割
k = np.array_split(A, 3, axis=1)
print(k)

###深拷贝和浅拷贝
a = np.arange(4)

b = a
c = a
d = b

print(a)
print(b)
print(c)
print(d)

print(a is b)
print(c is a)
print(d is b)
print(a is d)

# 浅拷贝
a[1] = 22
print(d)

# 深拷贝
p = a.copy()
a[2] = 44

print(p)
print(b)
