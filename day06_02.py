import numpy as np
import pandas as pd

# 首先建立了一个 6X4 的矩阵数据
datas = pd.date_range("20181030", periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=datas, columns=['A', 'B', 'C', 'D'])
print(df)

# 用索引或者标签确定需要修改值的位置。
df.iloc[2, 2] = 1111
df.loc["2018-11-01", 'B'] = 2222
print(df)

###根据条件设置
df.B[df.A > 4] = 0

df.iloc[0, 1] = np.nan
df.iloc[1, 2] = np.nan
print(df)

#如果想直接去掉有 NaN 的行或列, 可以使用 dropna
df1= df.dropna(axis=0, how='any')
print(df1)

#如果是将 NaN 的值用其他值代替, 比如代替成 0:
df2 = df.fillna(value=0)
print(df2)

#判断是否有缺失数据 NaN, 为 True 表示缺失数据:
df3 =df.isna()
print(df3)


print(np.any(df.isnull())== True)
