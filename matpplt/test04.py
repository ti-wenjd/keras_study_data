import matplotlib.pyplot as plt
import numpy as np

a = np.arange(36).reshape((4,9))
print(a)
#白色代表值最大的地方，颜色越深值越小。
# cmap的参数时用的是：cmap=plt.cmap.bone，
# 而现在，我们可以直接用单引号传入参数。
# origin='lower'代表的就是选择的原点的位置。
plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower')

plt.show()