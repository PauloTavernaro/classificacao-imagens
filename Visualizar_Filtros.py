import numpy as np
import cv2
from skimage.filters import roberts, sobel_h, sobel_v, prewitt_h, prewitt_v
from scipy import ndimage as nd
from skimage.segmentation import chan_vese

#Lendo a imagem, garantindo o tamanho de 960x540 px, e convertendo para escala de cinza
img = cv2.imread("/home/user/Área de Trabalho/Original_2/120.png")

img = cv2.resize(img, (960, 540))

img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)



#Definido o filtro Canny
edges = cv2.Canny(img, 100, 200)

#Definido o filtro Mediana
imagem_mediana = nd.median_filter(img, size=3)

#Definido o filtro Gaussiano com sigma = 3
imagem_gaussiana3 = nd.gaussian_filter(img, sigma=3)

#Definido o filtro Gaussiano com sigma = 7
imagem_gaussiana7 = nd.gaussian_filter(img, sigma=7)

#Definido o filtro Roberts
edges_roberts = roberts(img)

#Calculando o gradiente Sobel para cada direção (horizontal e vertical)
sobel_x = sobel_h(img)
sobel_y = sobel_v(img)
#Calculando a magnitude
magnitude_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
edges_sobel = magnitude_sobel

#Calculando o gradiente Prewitt para cada direção (horizontal e vertical)
prewitt_x = prewitt_h(img)
prewitt_y = prewitt_v(img)
#Calculando a magnitude
magnitude_prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)
edges_prewitt = magnitude_prewitt


#Definindo o filtro Chan-Vese
img=np.uint8(img)

cv = chan_vese(img, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
               max_num_iter=200, dt=0.5, init_level_set="checkerboard",
               extended_output=True)

#Definindo os  filtros Gabor
nucleos = []
for theta in range(1,4):
    theta = theta / 4 * np.pi
    for sigma in (1,3):
        for lamda in np.arange(np.pi/4, np.pi, np.pi/4):
            for gama in (0.05, 0.5):
                tamanho_nucleo = 2
                nucleo = cv2.getGaborKernel((tamanho_nucleo,tamanho_nucleo), sigma, theta, lamda, gama, 0, ktype=cv2.CV_32F)
                nucleos.append(nucleo)
                imagem_filtrada = cv2.filter2D(img, cv2.CV_8UC3, nucleo)
                #Mostrando cada uma das imagens com cada um dos filtros Gabor
                cv2.imshow(f'Filtradas (Gabor) - Theta:{theta} - Sigma:{sigma} - Lambda:{lamda} - Gama:{gama}', imagem_filtrada)
                
#Mostrando as imagens com seus respectivos filtros
cv2.imshow('Original', img)
cv2.imshow('Roberts', edges_roberts)
cv2.imshow('Sobel', edges_sobel)
cv2.imshow('Prewitt', edges_prewitt)
cv2.imshow('Gaussiana s3', imagem_gaussiana3)
cv2.imshow('Gaussiana s7', imagem_gaussiana7)
cv2.imshow('Canny Edge', edges)
cv2.imshow('Mediana', imagem_mediana)
cv2.imshow('Chan Vese', cv[1])


cv2.waitKey()
cv2.destroyAllWindows()