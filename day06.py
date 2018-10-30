import numpy as np
import pandas as pd

"""
如果用 python 的列表和字典来作比较, 
那么可以说 Numpy 是列表形式的，没有数值标签，
而 Pandas 就是字典形式。Pandas是基于Numpy构建的，让Numpy为中心的应用变得更加简单。
"""

##Series
s = pd.Series([1, 3, 5, np.nan, 44, 1])
print(s)

##DataFrame
dates = pd.date_range("20181030", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
print(df)

print(df['b'])

###创建一组没有给定行标签和列标签的数据 df1
df1 = pd.DataFrame(np.arange(12).reshape((3, 4)))
print(df1)
print(df1[0:2])