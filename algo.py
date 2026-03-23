from PIL import Image

img = Image.open('1.jpeg').convert('LA')

largeur,hauteur = img.size

img_bin = Image.new('1',(largeur,hauteur))

seuil = int(input("Seuil choisi ? (entre 0 et 255) "))

threshold_img = img.point(lambda p: p > seuil and 255)

threshold_img.save('image_final.tif')