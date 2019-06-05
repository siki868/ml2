import keras
from keras.datasets import fashion_mnist
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from keras.utils.np_utils import to_categorical
# Ucitavanje FashionMNIST skupa podataka
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# Koristimo samo deo trening skupa (prvi od 10 fold-ova) radi efikasnosti treninga
skf = StratifiedKFold(n_splits=6, random_state=0, shuffle=False)
for train_index, test_index in skf.split(x_train, y_train):
    x_train, y_train = x_train[test_index], y_train[test_index]
    break
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

#################################################################################
# U ovoj sekciji implementirati Keras neuralnu mrezu koja postize tacnost barem
# 85% na test skupu. Ne menjati fajl van ove sekcije.

x_train = x_train / 255
x_test = x_test / 255

# Za obican model ovo ne treba
x_train = x_train.reshape(10000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# Obican model sa obicnim slojevima

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(128, activation=tf.nn.sigmoid),
#     keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

# Model sa jednim konvolucionim slojem
# Duze traje, bolja tacnost

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.2),

    # keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Model za 3 konvoluciona sloja, meh

# model = keras.Sequential([
#     keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     keras.layers.Dropout(0.25),
#     keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.25),
#     keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.25),

#     keras.layers.Flatten(),
#     keras.layers.Dense(512, activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.5),

#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.5),

#     keras.layers.Dense(10, activation='softmax')
# ])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Preciznost {test_acc}')
#################################################################################

# Cuvanje istreniranog modela u fajl
#model.save('fashion.h5')