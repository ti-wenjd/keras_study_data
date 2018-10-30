import numpy as np
import pandas as pd

"""
pandas中的merge和concat类似,但主要是用于两组有key column的数据,统一索引的数据. 通常也被用在Database的处理当中.
"""

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['AD', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

print(left)
print(right)

res = pd.merge(left, right, on='key')
print(res)



