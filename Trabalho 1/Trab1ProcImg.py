import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def Fone (x, y):
    return (x * y + 2 * y)

def Ftwo (x, y):
    return ( math.fabs( math.cos(x/_Q) + 2 * math.sin(y/_Q) ) )

def Fthree (x, y):
    return ( math.fabs( 3*x/_Q - np.cbrt(y/_Q) ) )

def Ffour (x, y):
    return ( random.random() ) 

#def Ffive (x, y):

def NormalizeIMG (matrix, max_value):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix [0,1]
    matrix = matrix * max_value # normalizes range [0, max_value]
    return matrix
  

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
    x = 0
    y = 0
    img[x, y] = 1
    for i in range (1 + _img_size*_img_size):
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        x = (x + dx) % _img_size
        y = (y + dy) % _img_size
        img[x, y] = 1


img = NormalizeIMG (img, 65535) # Normalizing image

####################

###### Sampling image #######
step = int (_img_size / _reduced_img_size)

new_img = np.zeros( (_reduced_img_size, _reduced_img_size) )
for i in range (_reduced_img_size): #downsampling img into new_img
    for j in range (_reduced_img_size):
        new_img[i, j] = img[i * step, j * step]

############################

###### Quantizing ########
new_img = NormalizeIMG(new_img, 255)
new_img = new_img.astype(np.uint8)
#need bitwise shift

##########################

#### Comparing with reference image ####

RSE = 0.0
ref_img = np.load(filename) # loading reference image

for i in range (_reduced_img_size): # RSE sum loop
    for j in range (_reduced_img_size):
        RSE = RSE + ( (img[i,j] - ref_img[i,j]) ** 2 )

RSE = math.sqrt(RSE) # RSE square root
print("%f" % RSE)

#################################
