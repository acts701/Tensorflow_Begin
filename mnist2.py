# origin : https://www.youtube.com/watch?v=QqmugTjVbz4

from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(y_train.shape)

for i in range(5):
    print(y_train[i])

# neural network은 실수를 받아들일 수 없다. category 혹은 one hot encoding으로 불리는 것으로 바꿔줘야 한다
# ex 5 → [ 0,0,0,0,0,1,0,0,0,0 ] 실수이기 때문에 10개의 class가 있고 그 중에 5를 1로 바꿈
# data가 기차, 자동차, 배를 구분하는 거면 class는 3개가 된다
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

for i in range(5):
    print(y_train[i])

print(y_train.shape)

print(x_train.shape, x_test.shape)
# 결과는 (60000, 28, 28) (10000, 28, 28) 이다. 6만개의 28 * 28 size data가 있는 것
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

print(x_train.shape, x_test.shape)
# 결과가 (60000, 784) (10000, 784) 로 바뀜

model = keras.Sequential()
model.add(keras.layers.Dense(32, activation="sigmoid", input_shape=(784,)))
model.add(keras.layers.Dense(32, activation="sigmoid"))
model.add(keras.layers.Dense(10, activation="sigmoid"))

optimizer = keras.optimizers.SGD(lr=0.1)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

model.evaluate(x_test, y_test)
