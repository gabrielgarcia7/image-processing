# -----------------------------------------------------------
# Trabalho 01 : Geração de Imagens
# SCC0251 — Image Processing
# 
# Prof. Lizeth Andrea Castellanos Beltran
# Aluna PAE: Leo Sampaio Ferraz Ribeiro
#
# 2021.1
# 
# Trabalho feito por:
# Caio Augusto Duarte Basso - NUSP 10801173
# Gabriel Garcia Lorencetti - NUSP 10691891
# -----------------------------------------------------------

import numpy as np
import math
import imageio

####### functions definitions ######

def Equalize_histogram(mat):
    

def Single_histogram(hist, A):
    # computes for all levels in the range
    for i in range(256):
        # sum all positions in which A == i is true
        pixels_value_i = np.sum(A == i)
        # store it in the histogram array
        hist[i] = pixels_value_i
            
    return (hist)

# Histograma Cumulativo Individual 
def Fone(imgs):
    for i in range (4):
        histogram = np.zeros(256, dtype=uint8)
        histogram = Single_histogram (histogram, imgs[i])
        histogram = Equalize_histogram(histogram)

    return (imgs)

# Histograma Cumulativo Conjunto
def Ftwo(imgs):
    histogram = np.zeros(256, dtype=uint8)

    #creating histogram from all images
    for i in range (4):
        histogram = Single_histogram (histogram, imgs[i])

    histogram = Equalize_histogram(histogram)

    return (imgs)

# Funcao de Correcao Gamma
def Fthree(imgs, gamma):
    for img in range(4):
        x, y = (imgs[img]).shape
        for i in range (x):
            for j in range(y):
                (imgs[img])[i][j] = int( 255 * (( (imgs[img])[i][j] / 255.0) ** (1/gamma)) )

    return (imgs)

def RSME(img, ref):
    x, y = img.shape
    sum = 0.0
    for i in range(x):
        for j in range(y):
            sum += (ref[i][j] - img[i][j])**2

    return ( math.sqrt( sum / x*y ) )

####################################

####### input reading #######
_low_res_img_name = input().rstrip() # read low res images' name
_high_res_img_name = input().rstrip() # read high res image's name
_func_used = int(input()) # read chosen function's id 0 <= x <= 3
_gamma = int(input()) # read parameter γ
##############################

####### reading images #######
img1 = imageio.imread(_low_res_img_name + "1.png")
img2 = imageio.imread(_low_res_img_name + "2.png")
img3 = imageio.imread(_low_res_img_name + "3.png")
img4 = imageio.imread(_low_res_img_name + "4.png")
hd_img = imageio.imread(_high_res_img_name + ".png")
##############################
imgs = [img1, img2, img3, img4]

####### enhancing image #######
if (_func_used == 1):
    imgs = Fone(imgs)

if (_func_used == 2):
    imgs = Ftwo(imgs)

if (_func_used == 3):
    imgs = Fthree(imgs, _gamma)

################################

####### Combining images ########
x, y = hd_img.shape
new_img = np.zeros(hd_img.shape)

for i in range (x):
    for j in range (0, y, 2):
        if (i % 2 == 0): # if on even line
            new_img[i][j] = (imgs[0])[i/2][j/2]
            new_img[i][j+1] = (imgs[1])[i/2][j/2]
        else: # if on odd line
            new_img[i][j] = (imgs[2])[i/2][j/2]
            new_img[i][j+1] = (imgs[3])[i/2][j/2]

################################

#### Comparing with reference image ####
print("%.4f" % RSME(new_img, hd_img)) # printing the rsme value
#################################