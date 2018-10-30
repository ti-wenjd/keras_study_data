import numpy as np

"""
ndim：维度
shape：行数和列数
size：元素个数
"""

# 列表转矩阵
array = np.array([[1, 2, 3], [4, 5, 6]])
print(array)

# 维度
print(array.ndim)
# 行数和列数
print(array.shape)
# 元素格式
print(array.size)

####创建数组###
a = np.array([2.23, 4], dtype=np.float32)
print(a.dtype)

###创建特定数组
a = np.array([[2, 23, 4], [2, 23, 4]])
print(a)

###创建全零数组
a = np.zeros((3, 4))
print(a)

###创建全1数组
a = np.ones((3, 4))
print(a)

###创建全空数组, 其实每个值都是接近于零的数
b = np.empty((3, 4), dtype=float)
print(b)

###用 arange 创建连续数组:
a = np.arange(10, 20, 2)  # 10-19 的数据，2步长
print(a)

###使用 reshape 改变数据的形状
a = np.arange(15).reshape((3, 5))
print(a)

###用 linspace 创建线段型数据:
a = np.linspace(1, 10, 20)  # 开始端1，结束端10，且分割成20个数据，生成线段
print(a)

###同样也能进行 reshape 工作:
a = np.linspace(1, 10, 20).reshape((5, 4))
print(a)

####基础运算#####

a = np.array([10, 20, 30, 40])  # array([10, 20, 30, 40])
b = np.arange(4)  # array([0, 1, 2, 3])

c = a - b
print(c)  # array([10, 19, 28, 37])

# 在Numpy中，想要求出矩阵中各个元素的乘方需要依赖双星符号 **，以二次方举例
c = b ** 2
print(c)

##Numpy中具有很多的数学函数工具，比如三角函数等，当我们需要对矩阵中每一项元素进行函数运算时，可以很简便的调用它们（以sin函数为例）
c = np.sin(b)
print(c)

##在脚本中对print函数进行一些修改可以进行逻辑判断
print(b)
print(b == 3)

print("############################")
###对二维数组的操作
# 构建数据
a = np.array([[1, 1], [0, 1]])
b = np.arange(4).reshape((2, 2))
# print(a)
# print(b)

# 矩阵乘法运算，即对应行乘对应列得到相应元素
c_dot = np.dot(a, b)
print(c_dot)

# other method  另外的一种关于dot的表示方法
c_dot_2 = a.dot(b)
print(c_dot_2)

print("############################")

## 当axis的值为0的时候，将会以列作为查找单元， 当axis的值为1的时候，将会以行作为查找单元
a = np.random.random((2, 4))
print(a)
print(np.sum(a, axis=1))
print(np.min(a, axis=0))
print(np.max(a, axis=1))

print(np.argmin(a))
print(np.argmax(a))

##对应元素的索引也是非常重要的
A = np.arange(2, 14).reshape((3, 4))
print(np.argmin(A))
print(np.argmax(A))
##计算统计中的均值
print(np.mean(A))
print(np.average(A))

##另一种写法
print(A.mean())

##累加 matlab中的cumsum()累加函数类似
"""
生成的每一项矩阵元素均是从原矩阵首项累加到对应项的元素之和。比如元素9，
在cumsum()生成的矩阵中序号为3，即原矩阵中2，3，4三个元素的和。
"""
print(A)
print(np.cumsum(A))

##有累差运算函数：
"""
该函数计算的便是每一行中后一项与前一项之差。
故一个3行4列矩阵通过函数计算得到的矩阵便是3行3列的矩阵。
"""
print(np.diff(A))


###这个函数将所有非零元素的行与列坐标分割开，重构成两个分别关于行和列的矩阵。
###实际上就是取的 行数 和 列数 与具体的值无关
print(np.nonzero(A))
print(np.nonzero(np.ones((4,5))))

#这里的排序函数仍然仅针对每一行进行从小到大排序操作
A = np.arange(14,2,-1).reshape((3,4))
print(np.sort(A))

##矩阵的转置  (行变列  列变行)
print(A)
print(np.transpose(A))


##具有clip()函数
"""
这个函数的格式是clip(Array,Array_min,Array_max)，
顾名思义，Array指的是将要被执行用的矩阵，
而后面的最小值最大值则用于让函数判断矩阵中元素是否有比最小值小的或者比最大值大的元素，
并将这些指定的元素转换为最小值或者最大值。
"""
print(np.clip(A,5,9))
