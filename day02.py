from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)


print(model.get_config())
print(model.get_layer("input_1"))
print(model.layers)
print(model.inputs)
print(model.outputs)

model.compile(optimizer='RMSprop',)