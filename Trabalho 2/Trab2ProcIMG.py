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

def Equalize_histogram(hist, A, nm):
	
	histC = np.zeros(256).astype(int)

	histC[0] = hist[0]

	for i in range(1, 256):
		histC[i] = hist[i] + histC[i-1]

	hist_transform = np.zeros(256).astype(np.int32)
	
	# size of the input image
	N, M = A.shape
	
	# creating a matrix to store the equalised image version
	A_eq = np.zeros([N,M]).astype(np.int32)
	
	# for each intensity value, transforms it into a new intensity
	# using the np.where() function 
	for z in range(256):
		# computes what would be the output level 's' for an input value 'z'
		s = ((255.0)/float(nm))*histC[z] 

		A_eq[ np.where(A == z) ] = s
		
		# storing the tranformation function
		hist_transform[z] = s
	
	return (A_eq)
	

def Single_histogram(hist, A):
	# computes for all levels in the range
	for i in range(256):
		# sum all positions in which A == i is true
		pixels_value_i = np.sum(A == i)
		# store it in the histogram array
		hist[i] += pixels_value_i
			
	return (hist)

# Histograma Cumulativo Individual 
def Fone(imgs):
	for i in range (4):
		histogram = np.zeros(256, dtype=np.int32)
		histogram = Single_histogram (histogram, imgs[i])
		N, M = imgs[i].shape
		imgs[i] = Equalize_histogram(histogram, imgs[i], N*M)

	return (imgs)

# Histograma Cumulativo Conjunto
def Ftwo(imgs):
	histogram = np.zeros(256, dtype=np.int32)
	nm = 0
	#creating histogram from all images
	for i in range (4):
		histogram = Single_histogram (histogram, imgs[i])	
		N, M = imgs[i].shape
		nm += (N * M)

	for i in range (4):
		imgs[i] = Equalize_histogram(histogram, imgs[i], nm)

	return (imgs)

# Funcao de Correcao Gamma
def Fthree(imgs, gamma):
	for img in (imgs):
		#x, y = img.shape
		#for i in range (x):
		#	for j in range(y):
		#		img[i][j] = int( 255 * (( img[i][j] / 255.0) ** (1/gamma)) )
		img =  (255 * ( np.floor( (img / 255.0) ** (1.0/gamma) ) ) )

	return (imgs)

def RSME(img, ref):
	x, y = img.shape
	sum = 0.0
	for i in range(x):
		for j in range(y):
			sum += (ref[i][j] - img[i][j])**2

	return ( math.sqrt( sum / (x*y) ) )

def RSMEf(g, r):
	x, y = g.shape
	return math.sqrt((np.sum(np.square(g - r)))/(x*y))

####################################

####### input reading #######
_low_res_img_name = input().rstrip() # read low res images' name
_high_res_img_name = input().rstrip() # read high res image's name
_func_used = int(input()) # read chosen function's id 0 <= x <= 3
_gamma = float(input()) # read parameter γ
##############################

####### reading images #######
img1 = imageio.imread(_low_res_img_name + "0.png")
img2 = imageio.imread(_low_res_img_name + "1.png")
img3 = imageio.imread(_low_res_img_name + "2.png")
img4 = imageio.imread(_low_res_img_name + "3.png")
hd_img = imageio.imread(_high_res_img_name)
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

for i in range (0, x, 2):
	for j in range (0, y, 2):
			new_img[i][j] = (imgs[0])[int(i/2)][int(j/2)]
			new_img[i][j+1] = (imgs[3])[int(i/2)][int(j/2)]
			new_img[i+1][j] = (imgs[1])[int(i/2)][int(j/2)]
			new_img[i+1][j+1] = (imgs[2])[int(i/2)][int(j/2)]

################################

#### Comparing with reference image ####
print("%.4f" % RSMEf(new_img, hd_img)) # printing the rsme value
#################################