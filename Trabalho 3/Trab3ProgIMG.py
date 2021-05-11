# -----------------------------------------------------------
# Trabalho 02 : Realce e Superrresolução
# SCC0251 — Image Processing
# 
# Prof. Lizeth Andrea Castellanos Beltran
# Aluna PAE: Leo Sampaio Ferraz Ribeiro
# Monitor: Flavio Salles
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
def Fone (img, weights):

    pad_size = (len(weights) - 1) // 2 # calculating padding size for each side of image
    N,M = img.shape

    flattened_img = img.flatten() # flattening image into 1d array    

    new_img = np.zeros(flattened_img.shape)

    flattened_img = np.pad(flattened_img, (pad_size, pad_size), 'wrap') # adds padding to image


    for i in range (pad_size, len(flattened_img) - pad_size):
        for j in range (len(weights)):
            new_img[i - pad_size] += flattened_img[i-pad_size+j] * weights[j]

    new_img = np.reshape(new_img, (N,M)) # reshapes array into image
    return (new_img)

def Ftwo (img, weights):

    N,M = weights.shape
    pad_size = (N - 1) // 2 # calculating padding size for each side of image

    new_img = np.zeros(img.shape)

    img = np.pad(img, (pad_size, pad_size), 'reflect') # adds padding to image

    imgSize,imgSize2 = img.shape

    for i in range (pad_size, imgSize - pad_size):
        for j in range (pad_size, imgSize - pad_size):
            for pad_i in range (N):
                for pad_j in range (N):
                    new_img[i - pad_size][j - pad_size] += img[i - pad_size + pad_i][j - pad_size + pad_i] * weights[pad_i][pad_j]
    
    return (new_img)

def Fthree (img, filter_size):

    pad_size = (filter_size - 1) // 2 # calculating padding size for each side of image

    new_img = np.zeros(img.shape)

    img = np.pad(img, (pad_size, pad_size)) # adds padding to image

    imgSize,imgSize2 = img.shape

    filter = np.zeros((filter_size * filter_size))

    for i in range (pad_size, imgSize - pad_size):
        for j in range (pad_size, imgSize - pad_size):
            for pad_i in range (0, filter_size):
                for pad_j in range (0, filter_size):
                    filter [pad_i * filter_size + pad_j] = img[i - pad_size + pad_i][j - pad_size + pad_j]

            filter = np.sort(filter)
            new_img[i - pad_size][j - pad_size] = np.median(filter)

    return (new_img)

# Function that calculates the Root-mean-square deviation (RSME)
def RSME(g, r):
	x, y = g.shape
	return math.sqrt((np.sum(np.square(g - r)))/(x*y))

# function that normalizes the image, given a max_value
def NormalizeIMG (matrix, max_value):

    imax = np.max(matrix)
    imin = np.min(matrix)

    matrix_norm = (matrix-imin)/(imax-imin)
    matrix_norm = (matrix_norm*max_value)

    return matrix_norm
  
####################################

####### input reading #######
_img_ref_name = input().rstrip() # read image's name
_func_used = int(input()) # read chosen function's id 1 <= x <= 3

if _func_used == 1: # if using method 1 (Filtering 1D)
    _filter_size = int (input())
    _weight_sequence = np.zeros(_filter_size)
    _weight_sequence =  list(map(int, input().split()))

elif _func_used == 2: # if using method 2 (Filtering 2D)
    _filter_size = int (input())
    _weight_sequence = np.zeros ( (_filter_size, _filter_size))
    for i in range (_filter_size):
        _weight_sequence[i] = list(map(int, input().split()))


elif _func_used == 3: # if using method 3 (2D Median Filter)
    _filter_size = int ( input())

##############################

####### reading image #######
img = imageio.imread(_img_ref_name)
##############################

####### Filtering image ########
if (_func_used == 1):
	new_img = Fone(img, _weight_sequence)

elif (_func_used == 2):
	new_img = Ftwo(img, _weight_sequence)

elif (_func_used == 3):
	new_img = Fthree(img, _filter_size)
################################

#### Comparing with reference image ####

new_img = NormalizeIMG(new_img, 255)
print("%.4f" % RSME(new_img, img)) # printing the rsme value
#################################