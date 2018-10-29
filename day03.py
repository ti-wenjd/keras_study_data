
import keras
from keras.layers import Input, LSTM, Dense, Conv2D, Convolution2D, Flatten, Reshape
from keras.models import Model
from keras.models import Sequential


# as first layer in a Sequential model
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# now: model.output_shape == (None, 3, 4)
# note: `None` is the batch dimension
print(model.output_shape)
# as intermediate layer in a Sequential model
model.add(Reshape((6, 2)))
# now: model.output_shape == (None, 6, 2)
print(model.output_shape)

# also supports shape inference using `-1` as dimension
model.add(Reshape((-1, 2, 2)))
# now: model.output_shape == (None, 3, 2, 2)
print(model.output_shape)