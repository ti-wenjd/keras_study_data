import numpy as np
import pandas as pd


#读取csv
data = pd.read_csv("./alarm_info.csv")
print(data)

##将资料存取成pickle
data.to_pickle("alarm_info.pickle")
