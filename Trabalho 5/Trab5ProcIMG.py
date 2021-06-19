# -----------------------------------------------------------
# Assignment 5:  image descriptors
# SCC0251 — Image Processing
# 
# SCC5830/0251 — Prof. Moacir Ponti
# Teaching Assistant: Leo Sampaio Ferraz Ribeiro
# Teaching Assistant: Flavio Salles
#
# 2021.1
# 
# Made by:
# Caio Augusto Duarte Basso - NUSP 10801173
# Gabriel Garcia Lorencetti - NUSP 10691891
# -----------------------------------------------------------


import numpy as np
import imageio
import math
import scipy


####### functions definitions ######

# Function for transforming RGB to gray scale
def To_grayscale(img):
    N,M = img.shape
    for i in range(N):
        for j in range(M):
            img[i, j, 0] = np.floor(img[i, j, 0] * 0.299 + img[i, j, 1] * 0.587 + img[i, j, 2] * 0.114)

    img = img[:, :, 0:1].squeeze()
    return img

# Quantisation to b bits
def Quantisation (img, b):
    img = np.right_shift(img, b)
    return img

# Calculates and concatenates the descriptors
def Create_descriptors(img, b):
    dc = Normalized_histogram(img, (2**b)-1)
    dt = Haralick_descriptor(img)
    dg = Gradients_histogram(img)
    return np.concatenate(dc, dt, dg)

# Calculates normalized histogram, returns vector
def Normalized_histogram(img, b):
    hist = np.zeros(b, dtype=np.int32)
    for i in range(b):
		# sum all positions in which A == i is true
        pixels_value_i = np.sum(A == i)
		# store it in the histogram array
        hist[i] += pixels_value_i
    
    N,M = img.shape
    #simply divide h(k) by the total number of pixels in the image.
    p = hist / (N*M)
    p = np.linalg.norm(p, b)
    return p

# Calculates haralick texture descriptor, returns vector
def Haralick_descriptor(img):
    return something

# Calculates histogram of oriented gradients, returns vector
def Gradients_histogram(img):
    return something

# Function that calculates the Root-mean-square deviation (RSME)
def RSME(g, r):
	x, y = g.shape
	return math.sqrt((np.sum(np.square(g - r)))/(x*y))

####################################

####### input reading #######
_img_obj = input().rstrip() # read object image's name
_img_ref = input().rstrip() # read reference image's name
_quant_par = int(input()) # read quantisation parameter b <= 8
##############################

####### reading image #######
obj_img = imageio.imread(_img_obj)
ref_img = imageio.imread(_img_ref)
#############################

###### Preprocessing and Quantisation ######
grayscale_img = To_grayscale(obj_img)
grayscale_img = Quantisation(grayscale_img, _quant_par)
############################################

###### Creating Image Descriptors ##########
descriptor = Create_descriptors(grayscale_img, _quant_par)
############################################

###### Finding the object #########

###################################

###### Printing the results #####
print(close_x + " " + close_y)
#################################