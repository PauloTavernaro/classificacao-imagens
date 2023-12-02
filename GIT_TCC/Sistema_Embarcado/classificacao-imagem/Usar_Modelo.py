import numpy as np
import cv2
import pandas as pd
import pickle
import os
from skimage.filters import roberts, sobel_h, sobel_v, prewitt_h, prewitt_v
from scipy import ndimage as nd
from skimage.segmentation import chan_vese


#Definindo as bibliotecas do Qt
from PySide2.QtCore import Qt
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import (
    QApplication, QLabel, QWidget, QHBoxLayout, QMainWindow
)


#Filtro Gabor, que resulta em 48 versões de filtro com diferentes perfis de intensidade e foco
def filtro_Gabor(imagem):

	df = pd.DataFrame()
	index = 1
	nucleos = []
	gama = 0.5
	# sigma = 1
	for theta in range(1,4):
		theta = theta / 4 * np.pi
		for sigma in (1,3):
			for lamda in np.arange(0, np.pi, np.pi/4):
				for gama in (0.05, 0.5):
					gabor_index = 'Gabor' + str(index)
					tamanho_nucleo = 2
					nucleo = cv2.getGaborKernel((tamanho_nucleo,tamanho_nucleo), sigma, theta, lamda, gama, 0, ktype=cv2.CV_32F)
					nucleos.append(nucleo)
					imagem_filtrada = cv2.filter2D(imagem, cv2.CV_8UC3, nucleo)
					imagem_redimensionada = imagem_filtrada.reshape(-1)
					df_red = pd.DataFrame(imagem_redimensionada, columns = [gabor_index])
					df = pd.concat([df, df_red], axis = 1)
					index += 1

	return df

#Função que retorna como atributos de cada pixel o próprio valor de cada um deles    
def Original(imagem):
    original_redimensionado = imagem.reshape(-1)
    df_red = pd.DataFrame(original_redimensionado, columns = ['Pixels Originais'])

    return df_red

#Filtro Canny, detecta bordas
def Canny_Edge(imagem):
    magnitude = cv2.Canny(imagem, 100, 200)
    #Padronizando os dados
    magnitude = magnitude * 255/np.max(magnitude)
    #Arredondando os dados
    magnitude = np.round(magnitude)
    #Convertendo para inteiro
    magnitude = magnitude.astype(int)
    canny_redimensionado = magnitude.reshape(-1)
    df_red = pd.DataFrame(canny_redimensionado, columns = ['Canny Edge'])

    return df_red

#Filtro Roberts, detecta bordas
def Roberts(imagem):
    magnitude = roberts(imagem)
    #Padronizando os dados
    magnitude *= 255/np.max(magnitude)
    #Arredondando os dados
    magnitude = np.round(magnitude)
    #Convertendo para inteiro
    magnitude = magnitude.astype(int)
    roberts_redimensionado = magnitude.reshape(-1)
    df_red = pd.DataFrame(roberts_redimensionado, columns = ['Roberts'])

    return df_red

#Filtro Sobel, detecta bordas

def Sobel(imagem):
    #Calculando o gradiente Sobel para cada direção (horizontal e vertical)
    sobel_x = sobel_h(imagem)
    sobel_y = sobel_v(imagem)
    #Padronizando os dados
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude *= 255/np.max(magnitude)
    #Arredondando os dados
    magnitude = np.round(magnitude)
    #Convertendo para inteiro
    magnitude = magnitude.astype(int)
    sobel_redimensionado = magnitude.reshape(-1)
    df_red = pd.DataFrame(sobel_redimensionado, columns = ['Sobel'])

    return df_red

#Filtro Prewitt, detecta bordas
def Prewitt(imagem):
    #Calculando o gradiente Prewitt para cada direção (horizontal e vertical)
    prewitt_x = prewitt_h(imagem)
    prewitt_y = prewitt_v(imagem)
    #Padronizando os dados
    magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
    magnitude *= 255/np.max(magnitude)
    #Arredondando os dados
    magnitude = np.round(magnitude)
    #Convertendo para inteiro
    magnitude = magnitude.astype(int)
    prewitt_redimensionado = magnitude.reshape(-1)
    df_red = pd.DataFrame(prewitt_redimensionado, columns = ['Prewitt'])

    return df_red

#Filtro Gaussiano, aplica desfoque nas imagens
def Gaussiana(imagem, sigma):
    imagem_gaussiana = nd.gaussian_filter(imagem, sigma=sigma)
    gaussiana_redimensionada = imagem_gaussiana.reshape(-1)
    df_red = pd.DataFrame(gaussiana_redimensionada, columns = [f'Gaussiana_s{sigma}'])

    return df_red

#Filtro Mediana, reduz ruído
def Mediana(imagem):
    imagem_mediana = nd.median_filter(imagem, size=3)
    mediana_redimensionada = imagem_mediana.reshape(-1)
    df_red = pd.DataFrame(mediana_redimensionada, columns = ['Mediana'])

    return df_red

def Chan_Vese(imagem):
    imagem_chan_vese = chan_vese(imagem, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
               max_num_iter=200, dt=0.5, init_level_set="checkerboard",
               extended_output=True)
    array_chan_vese = np.array(imagem_chan_vese[1])
    #Padronizando os dados
    array_chan_vese *= 255/np.max(array_chan_vese)
    #Arredondando os dados
    array_chan_vese = np.round(array_chan_vese)
    #Convertendo para inteiro
    array_chan_vese = array_chan_vese.astype(int)
    chan_vese_redimensionada = array_chan_vese.reshape(-1)
    df_red = pd.DataFrame(chan_vese_redimensionada, columns = ['Chan_Vese'])

    return df_red


#Função que aplica os filtros a cada imagem, e organiza seus pixeis em um dataframe de dimensão
#518.400 (arranjos de pixeis) por 56 (quantidade de atributos de cada arranjo)

def definir_filtros(imagem):
	
    #Definindo o dataframe final de cada imagem
	dados_todos_filtros = pd.DataFrame()
	
    #Garantindo que os dataframes parciais de cada filtro sejam reiniciados a cada loop
	df_gabors = pd.DataFrame()
	df_original = pd.DataFrame()
	df_canny = pd.DataFrame()
	df_roberts = pd.DataFrame()
	df_sobel = pd.DataFrame()
	df_scharr = pd.DataFrame()
	df_prewitt = pd.DataFrame()
	df_gaussiana_s3 = pd.DataFrame()
	df_gaussiana_s7 = pd.DataFrame()
	df_mediana = pd.DataFrame()
	df_chan_vese = pd.DataFrame()

	#Como a biblioteca OpenCV lê as imagens na escala BGR, elas são primeiramente convertidas
    #em RGB, e depois em escala de cinza
	imagem_RGB = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
	imagem_tons_cinza = cv2.cvtColor(imagem_RGB, cv2.COLOR_RGB2GRAY)

	#Aplicando os filtros para cada imagem
	df_gabors = filtro_Gabor(imagem_tons_cinza)
	df_original = Original(imagem_tons_cinza)
	df_canny = Canny_Edge(imagem_tons_cinza)
	df_roberts = Roberts(imagem_tons_cinza)
	df_sobel = Sobel(imagem_tons_cinza)
	df_prewitt = Prewitt(imagem_tons_cinza)
	df_gaussiana_s3 = Gaussiana(imagem_tons_cinza, 3)
	df_gaussiana_s7 = Gaussiana(imagem_tons_cinza, 7)
	df_mediana = Mediana(imagem_tons_cinza)
	df_chan_vese = Chan_Vese(imagem_tons_cinza)

	#Reunindo todos os dataframes de cada filtro um outro dataframe com todos os atributos
    #de cada filtro alinhados horizontalmente
	dados_todos_filtros = pd.concat([df_gabors, df_original, df_canny, df_roberts, df_sobel, df_prewitt, df_gaussiana_s3, df_gaussiana_s7, df_mediana, df_chan_vese], axis=1)

	return dados_todos_filtros

#Configurando a classe principal, que fornece a interface gráfica com Qt
class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        #Configurando um Widget para servir de base para o layout da tela
        self.base = QWidget()
        #Configurando o layout para mostrar as labels na horizontal
        self.layout = QHBoxLayout()
        
		#Definindo a imagem original: tamanho e posição
        self.foto_original = QLabel(self)
        self.foto_original.resize(480, 270)
        self.foto_original.setPixmap(QPixmap("/usr/local/bin/fotos/Foto_Original.jpg"))
        self.foto_original.setScaledContents(True)
        self.foto_original.move(100, 200)
		
		#Definindo a legenda da foto original
        self.titulo_original = QLabel(self)
        self.titulo_original.resize(200, 20)
        self.titulo_original.setText("Imagem Original")
        self.titulo_original.move(280, 500)
        
		#Definindo a imagem classificada: tamanho e posição
        self.foto_classificada = QLabel(self)
        self.foto_classificada.resize(480, 270)
        self.foto_classificada.setPixmap(QPixmap("/home/imagens_classificadas/classificada.jpg"))
        self.foto_classificada.setScaledContents(True)
        self.foto_classificada.move(700, 200)
		
		#Definindo a legenda da foto classificada
        self.titulo_classificada = QLabel(self)
        self.titulo_classificada.resize(200, 20)
        self.titulo_classificada.setText("Imagem Classificada")
        self.titulo_classificada.move(880, 500)
		
        #Configurando o layout no Widget
        self.base.setLayout(self.layout)
        #Centralizando a tela
        self.setCentralWidget(self.base)
        #Maximizando o tamanho da tela
        self.setWindowState(Qt.WindowMaximized)
        

#Função principal    
if __name__ == "__main__":
    #Definindo o caminho do modelo convertido para arquitetura arm64 e carregando-o
	caminho_modelo = "/usr/local/bin/Modelo_ML_comp"
	carregar_modelo = pickle.load(open(caminho_modelo, 'rb'))
      
	#Definindo o caminho da foto original
	caminho_fotos_originais = "/usr/local/bin/fotos/"

	#Classificando cada uma das fotos originais
	for foto in sorted(os.listdir(caminho_fotos_originais)):
		imagem = cv2.imread(caminho_fotos_originais + foto)
		imagem = cv2.resize(imagem, (960, 540))
		imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
		imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

		X = definir_filtros(imagem)
		X = X.to_numpy()
		resultado = carregar_modelo.predict(X)
		segmentado = resultado.reshape(540, 960)
		cv2.imwrite(f'/home/imagens_classificadas/classificada.jpg', segmentado)

	app = QApplication()

	window = Window()
	window.show()

	app.exec_()
