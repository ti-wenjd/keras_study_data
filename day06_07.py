import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 随机生成1000个数据
data = pd.Series(np.random.randn(1000), index=np.arange(1000))

# 为了方便观看效果, 我们累加这个数据
data.cumsum()

# pandas 数据可以直接观看其可视化形式
#data.plot()

#plt.show()


###可视化
data1 = pd.DataFrame(
    np.random.randn(1000,4),
    index=np.arange(1000),
    columns=list("ABCD")
    )
data1.cumsum()
#data1.plot()
#plt.show()


ax = data1.plot.scatter(x='A',y='B',color='DarkBlue',label='Class1')

data1.plot.scatter(x='D',y='C',color='LightGreen',label='Class2',ax=ax)
plt.show()