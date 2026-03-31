import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
 
#lecture de l'image en niveau de gris
img = cv.imread('1.jpeg', cv.IMREAD_GRAYSCALE)

#convolution pour supprimer le bruit de l'image avant de la mettre en noir et blanc
img = cv.medianBlur(img,5)

#Otsu (ret : seuil optimal, th1 : img binaire généré)
ret,th1 = cv.threshold(img,0,255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Création d'une zone de 7x7 pixels
kernel = np.ones((7,7), np.uint8)

#enleve les petits bruits en fonction du kernel crée à la ligne précédente
#on conserve tout ce qui est plus grand que le kernel, soit les pièces
th1 = cv.morphologyEx(th1, cv.MORPH_OPEN, kernel, iterations=3)

#lisse les pièces pour qu'elles soient un maximum blanches
th1 = cv.morphologyEx(th1, cv.MORPH_CLOSE, kernel, iterations=3)

#on compte les groupes de pixels connectés
#labels : image de la même taille que th1, où chaque pixel d'une pièce a un numéro correspondant à sa composante
num_labels, labels = cv.connectedComponents(th1)

#premier label 0 est toujours le fond noir
#le nombre de pièces est donc num_labels - 1
nb_pieces = num_labels - 1

print("Nombre de pièces détectées : " + str(nb_pieces))

#affichage
titles = ["Image de base en niveau de gris", "Otsu seuil optimal : "+str(ret)]
images = [img, th1]
 
for i in range(2):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
