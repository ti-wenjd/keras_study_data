import pandas as pd
import numpy as np

dates = pd.date_range('20181030', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])

print(df.loc['2018-10-31'])

# 根据标签 loc   其中第一个是索引值（一个）  第二个参数是列值(多个)
# 本例子主要通过标签名字选择某一行数据， 或者通过选择某行或者所有行（:代表所有行）然后选其中某一列或几列数据
print(df.loc[:, ['A', 'B']])

print(df.loc['2018-11-01', ['A', 'B']])

### iloc
##采用位置进行选择 iloc, 在这里我们可以通过位置选择在不同情况下所需要的数据例如选某一个，连续选或者跨行选等操作


print(df.iloc[4, 2])

print(df.iloc[3:5, 1:3])

print(df.iloc[[1, 3, 5], [1, 2]])
# 或者
print(df.iloc[[1, 3, 5], 1:3])  # 在这里我们可以通过位置选择在不同情况下所需要的数据, 例如选某一个，连续选或者跨行选等操作。

print(df)
###混合的这两种ix
print(df.ix[:3, ['A', 'C']])  # 当然我们可以采用混合选择 ix, 其中选择’A’和’C’的两列，并选择前三行的数据。

##通过判断的筛选
print(df[df.A > 8])  # 我们可以采用判断指令 (Boolean indexing) 进行选择. 我们可以约束某项条件然后选择出当前所有数据.
