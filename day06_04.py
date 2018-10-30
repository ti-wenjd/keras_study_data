import numpy as np
import pandas as pd

# 合并
# axis (合并方向)
df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
df3 = pd.DataFrame(np.ones((3, 4)) * 2, columns=['a', 'b', 'c', 'd'])

res = pd.concat([df1, df2, df3], axis=0)
print(res)

# ignore_index(重置 index)
res1 = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
print(res1)

df4 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
df5 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['b', 'c', 'd', 'e'], index=[2, 3, 4])

# join='outer'为预设值，
# 因此未设定任何参数时，函数默认join='outer'。
# 此方式是依照column来做纵向合并，有相同的column上下合并在一起，其他独自的column个自成列，原本没有值的位置皆以NaN填充
res2 = pd.concat([df4, df5], axis=0, join='outer', ignore_index=True)
print(res2)

# 原理同上个例子的说明，但只有相同的column合并在一起，其他的会被抛弃。
res3 = pd.concat([df4, df5], axis=0, join='inner')
print(res3)

##join_axes (依照 axes 合并)
res4 = pd.concat([df4, df5], axis=1)
print(res4)

res5 = pd.concat([df4, df5], axis=1, join_axes=[df4.index])
print(res5)



