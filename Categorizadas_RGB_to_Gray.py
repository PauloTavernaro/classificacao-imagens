import numpy as np
import cv2
import os

#Convertendo as imagens de referência de RGB para escala de cinza
for i, imagem_referencia in enumerate(sorted(os.listdir("/home/user/Área de Trabalho/Categorizadas_RGB/")), 1):
    imagem = cv2.imread(f'/home/user/Área de Trabalho/Categorizadas_RGB/{imagem_referencia}')

    imagem = cv2.resize(imagem, (960, 540))

    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    imagem = imagem.ravel()

    nova_imagem = np.empty(imagem.shape)

    for j, valor in enumerate(imagem):

        if valor in range(53, 114):
            nova_imagem[j] = 76
        elif valor in range(114, 256):
            nova_imagem[j] = 150
        elif valor in range(0, 53):
            nova_imagem[j] = 29

    nova_imagem = nova_imagem.reshape(540, 960)

    cv2.imwrite(f'/home/user/Área de Trabalho/Imagens_Ref_Classificadas/00{i}_Gr.png', nova_imagem)
