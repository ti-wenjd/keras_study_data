import keras
from keras.layers import Input, LSTM, Dense, Conv2D
from keras.models import Model
from keras.models import Sequential


def testT():
    a = Input(shape=(32, 32, 3))
    b = Input(shape=(64, 64, 3))

    conv = Conv2D(16, (3, 3), padding='same')
    conved_a = conv(a)

    print(conv.get_input_at(0))
    # Only one input so far, the following will work:
    assert conv.input_shape == (None, 32, 32, 3)

    conved_b = conv(b)
    # now the `.input_shape` property wouldn't work, but this does:
    assert conv.get_input_shape_at(0) == (None, 32, 32, 3)
    assert conv.get_input_shape_at(1) == (None, 64, 64, 3)

    print(conv.get_input_at(0))
    print(conv.get_input_at(1))


##fit_generator
def generate_arrays_from_file(path):
    while 1:
        f = open(path)
        for line in f:
            # create Numpy arrays of input data
            # and labels, from each line in the file
            x, y = process_line(line)
            yield (x, y)
    f.close()


if __name__ == '__main__':
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=784))
    model.add(Dense(32, activation='relu'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(generate_arrays_from_file('./my_file.txt'),
                        samples_per_epoch=10, epochs=5)
