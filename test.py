import cv2
import sys
import numpy as np
import random as rng
import keras

if len(sys.argv) != 2:
    print("Neispravno pozvan fajl, koristiti komandu \"python3 main.py X\" za pokretanje na test primeru sa brojem X")
    exit(0)

model = keras.models.load_model('fashion.h5')
class_names = ['Majica', 'Pantalone', 'Duks', 'Haljina', 'Kaput', 'Sandale', 'Kosulja', 'Patike', 'Torba', 'Cizme']

tp_idx = sys.argv[1]
img = cv2.imread(f'tests/{tp_idx}.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Denoise na 35 radi lepo 
dst = cv2.fastNlMeansDenoising(img,None, h=30)


ret, thresh = cv2.threshold(dst, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
dobri = []
inp = []
font = cv2.FONT_HERSHEY_SIMPLEX

for contour in contours:
    if contour.shape[0] >= 20:
        dobri.append(contour)

for contour in dobri:
    (x,y,w,h) = cv2.boundingRect(contour)
    cv2.rectangle(dst, (x,y), (x+w,y+h), (0,255,0), 2)
    obl = dst[x:x+w, y:y+h]
    resized = cv2.resize(obl, (28, 28)) 
    image = np.reshape(obl, (1, 28, 28, 1))
    a = class_names[np.where(model.predict(image)[0]==1)[0][0]]
    dst = cv2.putText(dst, a, (x, y), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


cv2.imshow('img', dst)
cv2.waitKey(0)