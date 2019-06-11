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

# Delimo sve sa 255 jer hocemo inputi da budu od 0 do 1
x_train = x_train / 255
x_test = x_test / 255

# Za obican model ovo ne treba
# Za convolucionu treba nam dodatna dimenzija za dubino
# S ozirom da su podaci greyscale treba nam samo 1
x_train = x_train.reshape(10000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# Obican model sa obicnim slojevima

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(128, activation='sigmoid'),
#     keras.layers.Dense(10, activation='softmax')
# ])

# Model sa jednim konvolucionim slojem, mislim da je ovaj top za sad
# Duze traje, bolja tacnost
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),        # Konvolucioni sloj sa 32 filtera i kernelom (3, 3)
    keras.layers.MaxPooling2D(pool_size=(2, 2)),                                                    # Smanjujemo dimenzije
    # keras.layers.Dropout(0.2),                                                                      # Ignorisemo neke neurone da ne overfitujemo

    # keras.layers.Conv2D(32, kernel_size=3, activation='relu'),    
    keras.layers.Flatten(),                                                                         # Flattenujemo da bi mogli tensore dalje normalno da koristimo
    keras.layers.Dense(128, activation='relu'),                                                     # 1 obican i 1 output sloj
    keras.layers.Dense(10, activation='softmax')
])

# Model sa 3 konvoluciona sloja, meh

# model = keras.Sequential([
#     keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     keras.layers.Dropout(0.25),
#     keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     keras.layers.Dropout(0.25),
#     keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPooling2D(pool_size=(2, 2)),
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
 
# Model sa 2 konv, oko 88.5% 

# model = keras.Sequential([
#     keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
#     keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),

#     keras.layers.Flatten(),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Preciznost {test_acc}')
#################################################################################

# Cuvanje istreniranog modela u fajl
model.save('fashion.h5')