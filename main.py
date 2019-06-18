import cv2
import keras
import numpy as np
import sys

# Prvi i jedini argument komandne linije je indeks test primera
if len(sys.argv) != 2:
    print("Neispravno pozvan fajl, koristiti komandu \"python3 main.py X\" za pokretanje na test primeru sa brojem X")
    exit(0)

tp_idx = sys.argv[1]
img = cv2.imread('tests/{}.png'.format(tp_idx))

#################################################################################
# U ovoj sekciji implementirati obradu slike, ucitati prethodno trenirani Keras
# model, i dodati bounding box-ove i imena klasa na sliku.
# Ne menjati fajl van ove sekcije.

# Ucitavamo model
model = keras.models.load_model('fashion.h5')
class_names = ['Majica', 'Pantalone', 'Duks', 'Haljina', 'Kaput', 'Sandale', 'Kosulja', 'Patike', 'Torba', 'Cizme']

# TODO
solution = img.copy()

# Donja i gornja granica za cv da prepozna shape
donja = np.array([0,0,0])
gornja = np.array([220,220,220])

# Maska za granice
mask = cv2.inRange(solution, donja, gornja)
#ret, thresh = cv2.threshold(solution, 127, 255, cv2.THRESH_BINARY)

# Primena maske na findContours, vraca shapove 
contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# dobri - dobri shapovi, za_predikcije - slike koje cemo da trpamo u model, xy - x i y za pisanje teksta 
dobri = []
za_predikcije = []
xy = []
font = cv2.FONT_HERSHEY_SIMPLEX

kernel_sharpen = np.asarray([
  [0, -1, 0],
  [-1, 5, -1],
  [0, -1, 0]
])

# print(contours)

# Odvajamo dobre shapove
for contour in contours:
    if contour.shape[0] >= 25:
        dobri.append(contour)

# Nalazimo pravougaonik u kome se nalazi shape, njega inverzujemo jer model radi sa belim shapovima i crnom pozadinom 
# Resize na 28,28 gde joj je onda shape (28,28,3), nama treba (28, 28, 1) i zato ga prebacujemo u greyscale i dodajemo jednu dimenziju
for contour in dobri:
    (x,y,w,h) = cv2.boundingRect(contour)
    xy.append((x, y))
    obl = solution[y:y+h, x:x+w]
    obl = (255-obl)
    # obl = cv2.filter2D(obl, -1, kernel_sharpen)
    obl = cv2.fastNlMeansDenoising(obl, None, h=30)
    resized = cv2.resize(obl, (28, 28))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    resized = np.expand_dims(resized, axis=2)
    resized = resized / 255
    # resized = np.reshape(resized, (28, 28, 1))
    za_predikcije.append(resized)
    cv2.rectangle(solution, (x,y), (x+w,y+h), (255,0,0), 2)
    
    # solution = cv2.putText(solution, a, (x, y), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


# Nalazimo predikcije
za_predikcije = np.array(za_predikcije)
predictions = model.predict_classes(za_predikcije, verbose=1)
print(predictions)

# Stavljamo imena klasa na njihova mesta
for oba, idx in zip(xy, predictions):
    name = class_names[idx]
    solution = cv2.putText(solution, name, (oba[0]-15, oba[1]-5), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('img', solution)
cv2.waitKey(0)


#################################################################################

# Cuvamo resenje u izlazni fajl
# cv2.imwrite("tests/{}_out.png".format(tp_idx), solution)