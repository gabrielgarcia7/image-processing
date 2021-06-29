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
    dt = Haralick_descriptor(img, (2**b)-1)
    dg = Gradients_histogram(img)
    return np.concatenate(dc, dt, dg)

# Calculates normalized histogram, returns vector
def Normalized_histogram(img, b):
    hist = np.zeros(b, dtype=np.int32)
    for i in range(b):
		# sum all positions in which A == i is true
        pixels_value_i = np.sum(img == i)
		# store it in the histogram array
        hist[i] += pixels_value_i
    
    N,M = img.shape
    #simply divide h(k) by the total number of pixels in the image.
    p = hist / (N*M)
    p = np.linalg.norm(p, b)
    return p

def energy(c):
    return np.sum(c**2)

def entropy(c):
    return -np.sum(c*np.log(c+0.001))

def contrast(c):    
    linha, coluna = c.shape
    sum = 0
    
    for i in range(linha):
        for j in range(coluna):
            sum += ((i-j)**2)*c[i, j]

    return sum/(linha*coluna)

def correlation(c):

    mi_i = 0 
    mi_j = 0
    delta_i = 0
    delta_j = 0
    correlation = 0
    linha, coluna = c.shape

    # mi_i
    for i in range(linha):
        mi_i *= i
        for j in range(coluna):
            mi_i += c[i, j]

    # delta_i
    for i in range(linha):
        delta_i *= (i-mi_i)**2
        for j in range(coluna):
            delta_i += c[i, j]

    # mi_j
    for j in range(linha):
        mi_j *= j
        for i in range(coluna):
            mi_j += c[i, j]

    # delta_j
    for j in range(linha):
        delta_j *= (j-mi_j)**2
        for i in range(coluna):
            delta_j += c[i, j]

    # correlation
    for i in range(linha):
        for j in range(coluna):
            correlation += ((i*j*c[i,j])-(mi_i*mi_j))/(delta_i*delta_j)


    return correlation

def homogeneity(c):
    linha, coluna = c.shape
    sum = 0
    
    for i in range(linha):
        for j in range(coluna):
            sum += c[i, j]/(1 + abs(i -j))

    return sum

def coocurrence(g, intensities):
    linha, coluna = g.shape

    c = np.zeros([intensities, intensities])

    for i in range(linha-1):
        for j in range(coluna-1):
            c[g[i, j], g [i+1, j+1]] += 1

    c = c / np.sum(c)
    return c

# Calculates haralick texture descriptor, returns vector
def Haralick_descriptor(img, intensities):
    
    c = coocurrence(img, intensities)

    dt = np.array(5)

    dt[0] = energy(c)
    dt[1] = entropy(c)
    dt[2] = contrast(c)
    dt[3] = correlation(c)
    dt[5] = homogeneity(c)

    np.linalg.norm(dt)

    return dt

# Calculates histogram of oriented gradients, returns vector
def Gradients_histogram(img):

    wsx = [ [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]]

    wsy = [ [-1,  0,  1],
            [-2,  0,  2],
            [-1,  0,  1]]

    gradientX = scipy.ndimage.convolve(img, wsx, mode='constant', cval=0.0)
    gradientY = scipy.ndimage.convolve(img, wsy, mode='constant', cval=0.0)

    m = np.sqrt(gradientX**2 + gradientY**2)/np.sum(gradientX**2 + gradientY**2)

    np.seterr(divide='ignore', invalid='ignore')

    fi = np.arctan(gradientY/gradientX)
    fi = fi + np.pi/2
    fi = np.degrees(fi)

    # digitise the angles into 9 bins
    bins = np.arange(20, 180, 20)
    fi_d = np.digitize(fi, bins, right=False)

    linhas, colunas = img.shape

    dg = np.zeros(9)

    for i in range(linhas):
        for j in range(colunas):
            dg[fi_d[i,j]] += m[i,j]

    return dg

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
#print(close_x + " " + close_y)
#################################