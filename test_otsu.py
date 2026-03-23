from PIL import Image
from otsu import trouver_seuil_otsu 
import cv2 as cv
import numpy as np

img = Image.open('1.jpg').convert('L')

largeur,hauteur = img.size

img_np = np.array(img)

# seuil = int(input("Seuil choisi ? (entre 0 et 255) "))

seuil = trouver_seuil_otsu(img)

print(seuil)

threshold_img = img.point(lambda p: p > seuil and 255)

threshold_img.save('image_final.tif')