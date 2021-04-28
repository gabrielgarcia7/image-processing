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
import random

####### functions definitions ######

# image generation - function 1
# f(x,y) = (xy + 2y)
def Fone (x, y):
    return (x * y + 2 * y)


# image generation - function 2
# f(x,y) = |cos(x/Q) + 2sin(y/Q)|
def Ftwo (x, y):
    return ( math.fabs( math.cos(x/_Q) + 2 * math.sin(y/_Q) ) )


# image generation - function 3
# f(x,y) = |3(x/Q)- cbrt(y/Q)|
def Fthree (x, y):
    return ( math.fabs( 3*x/_Q - np.cbrt(y/_Q) ) )


# image generation - function 4
# f(x,y) = rand(0,1,S)
def Ffour (x, y):
    return ( random.random() ) 


# image generation - function 5
# f(x,y) = randomwalk(S)
def Ffive ():
    x = 0
    y = 0
    img[x, y] = 1
    for i in range (1 + _img_size*_img_size):
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        x = (x + dx) % _img_size
        y = (y + dy) % _img_size
        img[x, y] = 1
    return img


# function that normalizes the image, given a max_value
def NormalizeIMG (matrix, max_value):

    imax = np.max(matrix)
    imin = np.min(matrix)

    matrix_norm = (matrix-imin)/(imax-imin)
    matrix_norm = (matrix_norm*max_value)

    return matrix_norm


# function that performs a bitwise right and a bitwise left, to leave only the most significant B bits
def BitwiseShift (matrix, bit_shift):
    
    matrix = np.right_shift(matrix, bit_shift)

    matrix = np.left_shift(matrix, bit_shift)

    return matrix


# function that calculates the relative standard error
def RSE(g, r):
    return math.sqrt(np.sum(np.square(g - r)))


#############################


####### input reading ######
_img_name = input().rstrip() # read starting image's name
_img_size = int(input()) # read starting image's size
_func_used = int(input()) # read chosen function's id 1 <= x <= 5
_Q = int(input()) # read parameter Q
_reduced_img_size = int(input()) # read shown image's size x <= _img_size
_bits_per_pixel = int(input()) # read number of bits per pixel 1 <= x <= 8
_seed = int(input()) # read random func's seed
#############################


####### synthesizing  image #######

img = np.zeros( (_img_size, _img_size) )

if (_func_used == 1):
    for x in range (_img_size):
        for y in range (_img_size):
            img[x,y] = Fone(x,y)

if (_func_used == 2):
    for x in range (_img_size):
        for y in range (_img_size):
            img[x,y] = Ftwo(x,y)

if (_func_used == 3):
    for x in range (_img_size):
        for y in range (_img_size):
            img[x,y] = Fthree(x,y)

if (_func_used == 4):
    random.seed(_seed)
    for x in range (_img_size):
        for y in range (_img_size):
            img[y, x] = Ffour(y, x)

if (_func_used == 5):
    random.seed(_seed)
    img = Ffive()


img = NormalizeIMG (img, 65535) # Normalizing image

####################


###### Sampling image #######

step = int (_img_size / _reduced_img_size)

red = int(_reduced_img_size)
new_img = np.zeros( (_reduced_img_size, _reduced_img_size) )
for i in range (red): #downsampling img into new_img
    for j in range (red):
        new_img[i, j] = img[i * step, j * step]

############################


###### Quantizing ########

new_img = NormalizeIMG(new_img, 255)
new_img = new_img.astype(np.uint8)
new_img = BitwiseShift(new_img, 8-_bits_per_pixel)

##########################


#### Comparing with reference image ####

ref_img = np.load(_img_name) # loading reference image
print("%.4f" % RSE(new_img, ref_img)) # printing the rse value

#################################
