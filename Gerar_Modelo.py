import numpy as np
import cv2
import pandas as pd
import pickle
import os
from skimage.filters import roberts, sobel_h, sobel_v, prewitt_h, prewitt_v
from scipy import ndimage as nd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import compute_class_weight
from skimage.segmentation import chan_vese
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

#Filtro Gabor, que resulta em 48 versões de filtro com diferentes perfis de intensidade e foco

def filtro_Gabor(imagem):
    df = pd.DataFrame()
    index = 1
    nucleos = []
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
    #Padronizando os dados entre 0 e 255
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
    #Padronizando os dados entre 0 e 255
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
    #Padronizando os dados entre 0 e 255
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
    #Padronizando os dados entre 0 e 255
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

#Filtro Chan-Vese, detecta regiões da imagem

def Chan_Vese(imagem):
    imagem_chan_vese = chan_vese(imagem, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
               max_num_iter=200, dt=0.5, init_level_set="checkerboard",
               extended_output=True)
    array_chan_vese = np.array(imagem_chan_vese[1])
    #Padronizando os dados entre 0 e 255
    array_chan_vese *= 255/np.max(array_chan_vese)
    #Arredondando os dados
    array_chan_vese = np.round(array_chan_vese)
    #Convertendo para inteiro
    array_chan_vese = array_chan_vese.astype(int)
    chan_vese_redimensionada = array_chan_vese.reshape(-1)
    df_red = pd.DataFrame(chan_vese_redimensionada, columns = ['Chan_Vese'])

    return df_red

#Função que aplica os filtros a cada imagem, e organiza seus pixeis em um dataframe de dimensão
#518.400 (arranjos de pixeis) por 56 (quantidade de atributos de cada arranjo), que será anexado
#em um dicionário cuja chave é sua imagem

def definir_filtros(caminho_imagens):
    #Definindo o dicionário que terá todos os dataframes de cada imagem 
    dict_features = {}

    i = 1

    for imagem_treino in sorted(os.listdir(caminho_imagens)):

        #Definindo o dataframe final de cada imagem
        dados_todos_filtros = pd.DataFrame()
        #Garantindo que os dataframes parciais de cada filtro sejam reiniciados a cada loop
        df_gabors = pd.DataFrame()
        df_original = pd.DataFrame()
        df_canny = pd.DataFrame()
        df_roberts = pd.DataFrame()
        df_sobel = pd.DataFrame()
        df_prewitt = pd.DataFrame()
        df_gaussiana_s3 = pd.DataFrame()
        df_gaussiana_s7 = pd.DataFrame()
        df_mediana = pd.DataFrame()
        df_chan_vese = pd.DataFrame()

        #Lendo a imagem 
        imagem = cv2.imread(caminho_imagens + imagem_treino)

        #Garantindo que tenha o tamanho de 960x540 px
        imagem = cv2.resize(imagem, (960, 540))

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

        #Preenchendo o dicionário com o dataframe anterior e sua respectiva imagem
        dict_features.update({f'Imagem_{i}' : dados_todos_filtros})

        i += 1

    return dict_features

#Função que recebe cada imagem de referência e organiza seus pixeis em um dataframe de dimensão
#518.400 (pixeis) por 1 (atributo de classificação), que será anexado em um dicionário cuja 
#chave é sua imagem

def definir_referencia(caminho_imagens):
    #Definindo o dicionário que terá todos os dataframes de cada imagem
    dict_ref = {}

    j = 1

    for imagem_referencia in sorted(os.listdir(caminho_imagens)):

        #Garantindo que o dataframe final de cada imagem seja reiniciado a cada loop
        dados_categorizadas = pd.DataFrame()

        #Lendo a imagem 
        imagem = cv2.imread(caminho_imagens + imagem_referencia)

        #Garantindo que tenha o tamanho de 960x540 px
        imagem = cv2.resize(imagem, (960, 540))

        #Como a biblioteca OpenCV lê as imagens na escala BGR, elas são primeiramente convertidas
        #em RGB, e depois em escala de cinza, ainda que as imagens de referência já estejam na 
        #escala de cinza
        imagem_RGB = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        imagem_tons_cinza = cv2.cvtColor(imagem_RGB, cv2.COLOR_RGB2GRAY)

        #Redimensionando cada imagem para ter apenas uma dimensão
        dados_imagem_ref = imagem_tons_cinza.reshape(-1)

        #Organizando os dados de cada imagem em um dataframe
        dados_categorizadas = pd.DataFrame(dados_imagem_ref, columns = ['Imagens de Referência'])

        #Preenchendo o dicionário com o dataframe anterior e sua respectiva imagem
        dict_ref.update({f'Imagem_Ref_{j}':dados_categorizadas})

        j += 1
    
    return dict_ref

#Função que divide cada dataframe de X e y em grupos de treino e teste para que então o modelo seja
#treinado em etapas, ou seja, a partir dos dados de cada imagem para economizar armazenamento de RAM

def divisao_treino_teste(X, y, model, i, todos_X_teste, todos_y_teste):

    #Divide os valores de X e y em grupos de treino e teste. A parcela de teste é de 20%
    #Além disso, "random_state" garante a reprodutibilidade 
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=1)

    #Convertendo o dataframe em arranjos NumPy
    X_treino = X_treino.to_numpy()
    y_treino = y_treino.to_numpy().ravel()

    #Treinando o modelo em partes (dados de uma imagem por vez)
    #As classes correspondem às classificações:
    #29: Vegetação Rasteira e Arbustos
    #76: Sem Vegetação
    #150: Vegetação de Floresta
    model.partial_fit(X_treino, y_treino, classes=[29, 76, 150])

    #Convertendo o dataframe em arranjos NumPy
    X_teste = X_teste.to_numpy()
    y_teste = y_teste.to_numpy().ravel()


    #Concatenando todos os dados de teste de todas as fotos até o momento
    #para que o modelo resultante de todas as iterações até então avalie todas
    #as informações de teste até o momento
    if i == 1:
        todos_y_teste = y_teste
        todos_X_teste = X_teste
    else:
        todos_y_teste = np.append(todos_y_teste, y_teste)
        todos_X_teste = np.append(todos_X_teste, X_teste, axis=0)

    #Obtendo a acurácia do modelo
    print(f"Acurácia depois da Imagem {i} = {model.score(todos_X_teste, todos_y_teste)}")

    #Realizando a previsão de resposta do modelo frente a todos os dados de teste de X até então
    previsao = model.predict(todos_X_teste)

    #Obtendo o relatório de classificação, o qual retorna valores como precisão e recall
    print(f"Relatório de Classificação depois da Imagem {i}: \n{metrics.classification_report(todos_y_teste, previsao)}\n")

    return model, todos_X_teste, todos_y_teste


###################################################

#Função principal

if __name__ == "__main__":

    #Definindo dataframes para captar os dados das imagens
    todos_original = pd.DataFrame()
    todos_categorizadas = pd.DataFrame()
    todos_X_teste = pd.DataFrame()
    todos_y_teste = []
    todos_y_teste = np.array(todos_y_teste)
    
    #Definindo os caminhos para encontrar as fotos
    caminho_imagens_original = '/home/user/Área de Trabalho/Originais/'
    caminho_imagens_categorizadas = '/home/user/Área de Trabalho/Referencia/'


    #Transformando os dados em listas para facilitar a iteração
    dados_original = list(definir_filtros(caminho_imagens_original).values())
    dados_categorizadas = list(definir_referencia(caminho_imagens_categorizadas).values())


    #Iteração de cada foto pelo treino e teste do modelo
    for i, (X, y) in enumerate(zip(dados_original, dados_categorizadas), 1):

        if i == 1:

            #Peso das classes sendo definidos
            peso = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y.to_numpy().ravel())

            modelo_inicial = lambda peso, y : SGDClassifier(loss='hinge', shuffle=True, random_state=1, warm_start=True, learning_rate='adaptive', eta0 = 0.01, class_weight=dict(zip(np.unique(y), peso)), average=True)

            modelo_pronto, todos_X_teste, todos_y_teste = divisao_treino_teste(X, y, modelo_inicial(peso, y), i, todos_X_teste, todos_y_teste)

        else:

            #Peso das classes sendo definidos
            peso = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y.to_numpy().ravel())
            
            modelo_pronto.set_params(**{'class_weight' : dict(zip(np.unique(y), peso))})

            modelo_pronto, todos_X_teste, todos_y_teste = divisao_treino_teste(X, y, modelo_pronto, i, todos_X_teste, todos_y_teste)

    
    #Calculando a matriz de confusão final
    previsao = modelo_pronto.predict(todos_X_teste)

    print(f'\nMatriz de Confusão: \n{metrics.confusion_matrix(todos_y_teste, previsao, labels=[29, 76, 150])}')


    #Salvando o modelo treinado
    caminho_modelo = "/home/user/Área de Trabalho/Modelo_ML"
    pickle.dump(modelo_pronto, open(caminho_modelo, 'wb'))