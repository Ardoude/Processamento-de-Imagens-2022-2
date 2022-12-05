# Eduardo Pereira Costa - 650503
# Rafael Maia - 635921

from pickle import FALSE, TRUE
from re import X
import os
import utils
import pandas as pd
import tkinter as tk
import tensorflow as tf
from tkinter import ACTIVE, DISABLED
from tkinter.filedialog import askdirectory
from PIL import ImageTk, Image
import cv2
import numpy as np
from skimage.feature import match_template
from matplotlib import pyplot as plt
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as preProcImg
import random

#Bibliotecas para realizar o processo de espelhamento horizontal
# e equalização do histograma
import numpy as np
from skimage import data
import math
#fim

# Constantes
tamanhoDaImagem = 224
btnHeight = 1  # Altura padrão dos botões
btnWidth = 20  # Largura padrão dos botões
imgSelecionada = None # Imagem a ser classificada
copiaImgSelecionada = None # Cópia da imagem selecionada (Para fins de exibição)
caminhoDaImagemAberta = "" # Caminho da imagem selecionada
dirPath = "" # Diretorio com a base a ser treinada
imagemARecortar = None # Imagem a ser recortada (Cópia)
cropping = False # Imagem sendo recortada
cropped = False # Imagem foi recortada
DATADIR  = ""
CATEGORIAS = ["0", "1", "2", "3", "4"]
trainingData = []
x_start, y_start, x_end, y_end = 0, 0, 0, 0 # Coordenadas de recorte da imagem

X_train = []
y_train = []
X_val = []
y_val = []

model = None

# Configuração da tela
janela = tk.Tk(className= ' Trabalho Prático - Processamento de Imagens')
janela.geometry("350x500")
    
# Métodos utilitários

# Atualiza a imagem a ser exibida
def atualizaImagem(path):
    global imgSelecionada, copiaImgSelecionada, caminhoDaImagemAberta
    imgSelecionada = ImageTk.PhotoImage( Image.open(path))#.resize( (255, 255), resample=3) )
    copiaImgSelecionada = ImageTk.PhotoImage( Image.open(path).resize( (255, 255), resample=3) )
    labelImagem.config(image = copiaImgSelecionada, height=255, width=255)
    labelImagem.img = copiaImgSelecionada
    caminhoDaImagemAberta = path

# Carrega uma imagem por um endereço e a exibe na tela
def carregarImagem():
    global caminhoDaImagemAberta, imgSelecionada
    caminhoDaImagemAberta = utils.browseFiles()
    if caminhoDaImagemAberta!="":
        atualizaImagem(caminhoDaImagemAberta)
        btnRecortarImagem.config(state=ACTIVE)
        btnCorrelacionarImagem.config(state=ACTIVE)
        print(imgSelecionada)

# Obtem as coordenadas de corte com base nos cliques do mouse
def recorte(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping, cropped, i
    # Clique do mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    # Mouse movendo
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    # Se soltar o botao do mouse
    elif event == cv2.EVENT_LBUTTONUP:
        # Gravar coordenadas
        x_end, y_end = x, y
        cropping = False # Finalizou o recorte
        pontoDeReferencia = [(x_start+2, y_start+2), (x_end-1, y_end-2)]
        if len(pontoDeReferencia) == 2: # Dois pontos encontrados
            regiaoDeInteresse = imagemARecortar[pontoDeReferencia[0][1]:pontoDeReferencia[1][1], pontoDeReferencia[0][0]:pontoDeReferencia[1][0]]
            cv2.imwrite("recorte.png", regiaoDeInteresse)
            cropped = True # Recorte já foi finalizado
            atualizaImagem("recorte.png")

# Abre uma janela para recortar uma região da imagem
def recortarImagem():
    global imagemARecortar, cropped
    janelaAberta = True
    cropped = False
    if imgSelecionada == None:
        print("Sem imagem selecionada")
        return
    cv2.namedWindow("recortar", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("recortar", recorte)
    while not cropped and janelaAberta:
        imagemARecortar = cv2.imread(caminhoDaImagemAberta).copy()

        # Finalizar se a janela for fechada sem que um recorte tenha sido feito
        if cv2.getWindowProperty("recortar", 0) < 0:
            janelaAberta = False

        if not cropping:
            cv2.imshow("recortar", imagemARecortar)
        elif cropping:
            cv2.rectangle(imagemARecortar, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("recortar", imagemARecortar)
        cv2.waitKey(1)
    cv2.destroyWindow("recortar")

#Correlacionando a Imagem
def correlacionarImagem():
    imgRecorte = cv2.imread("recorte.png")
    imgReferencia = cv2.imread(caminhoDaImagemAberta)
    
    result = cv2.matchTemplate(imgReferencia, imgRecorte, cv2.TM_CCOEFF)
    (tH, tW) = imgRecorte.shape[:2] # Tamanho da imagem de recorte
    _, maxVal, _, maxLoc = cv2.minMaxLoc(result)
    (startX, startY) = (int(maxLoc[0]), int(maxLoc[1])) 
    (endX, endY) = (int(maxLoc[0]+tW), int(maxLoc[1]+tH)) 
    cv2.rectangle(imgReferencia, (startX, startY), (endX, endY), (255,0,0), 2) # Desenhar retângulo onde a imagem se encaixa

    # Exibir resultados
    cv2.namedWindow("correlacao", cv2.WINDOW_NORMAL)
    cv2.imshow("correlacao", imgReferencia)
    cv2.waitKey(0)





#Iniciando o histograma, que será um objeto com 256 valores, 
# sendo cada um deles um dos níveis de intensidade que uma 
# imagem pode assumir, pois o domínio de intensidade de um 
# pixel varia entre 0 (para cor preta) até 255 (para cor branca).
def instantiate_histogram():    
    hist_array= []
    
    for i in range(0,256):
        hist_array.append(str(i))
        hist_array.append(0)
    
    hist_dct = {hist_array[i]: hist_array[i + 1] for i in range(0, len(hist_array), 2)} 
    
    return hist_dct

#Uma vez que o dicionário(histograma) foi criado, basta apenas contar quantas
# vezes cada valor de intensidade aparece na imagem, percorrendo
# cada um dos pixels e contabilizando o número de aparições.
def count_intensity_values(hist, img):
    for row in img:
        for column in row:
            hist[str(int(column))] = hist[str(int(column))] + 1
     
    return hist

#A função plot_hist abaixo foi utilizada para exibir o histograma da imagem,
# seja individualmente ou uma comparação lado a lado entre dois histogramas.
def plot_hist(hist, hist2=''):
    if hist2 != '':
        figure, axarr = plt.subplots(1,2, figsize=(20, 10))
        axarr[0].bar(hist.keys(), hist.values())
        axarr[1].bar(hist2.keys(), hist2.values())
    else:
        plt.bar(hist.keys(), hist.values())
        plt.xlabel("Níveis intensidade")
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        plt.grid(True)
        plt.show()
        
        
# Uma vez com o histograma da imagem em mãos, podemos
# dar continuidade ao desenvolvimento, sendo necessário
# calcular agora um outro dicionário porém ao invés do número
# de vezes que determinado valor de intensidade aparece estamos
# interessados na probabilidade desse valor aparecer.
# Um cálculo probabilístico é nada mais que a divisão de quantas vezes 
# o valor apareceu pelo número total de pixels na imagem.
def get_hist_proba(hist, n_pixels):
    hist_proba = {}
    for i in range(0, 256):
        hist_proba[str(i)] = hist[str(i)] / n_pixels
    
    return hist_proba

#O próximo passo, ainda utilizando a estrutura de dicionário,
# é calcularmos a probabilidade acumulada, onde para cada iteração
# o valor do histograma é somado à probabilidade acumulada das iterações
# anteriores. A implementação e detalhes matemáticos foram retirados do
# livro de Gonzalez e Woods.
def get_accumulated_proba(hist_proba): 
    acc_proba = {}
    sum_proba = 0
    
    for i in range(0, 256):
        if i == 0:
            pass
        else: 
            sum_proba += hist_proba[str(i - 1)]
            
        acc_proba[str(i)] = hist_proba[str(i)] + sum_proba
        
    return acc_proba

#Com todas essas probabilidades podemos fazer o cálculo dos novos
# valores de cinza da imagem, ou seja, dado um pixel na posição (x,y)
# com nível de intensidade z, qual será seu novo valor de intensidade
# para que o histograma resultante seja equalizado.
#Primeiro, calculamos um novo objeto que irá mapear os respectivos valores de cinza em novos valores equalizados
def get_new_gray_value(acc_proba):
    new_gray_value = {}
    
    for i in range(0, 256):
        new_gray_value[str(i)] = np.ceil(acc_proba[str(i)] * 255)
        
    return new_gray_value

#Por fim, basta aplicar os novos valores na imagem original.
def equalize_hist(img, new_gray_value):
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            img[row][column] = new_gray_value[str(int(img[row] [column]))]
            
    return img

# Aumenta a quantidade de imagens do dataset
def aumentarDados(image):
    histogram = instantiate_histogram()
    histogram = count_intensity_values(histogram, image)
    n_pixels = image.shape[0] * image.shape[1]
    hist_proba = get_hist_proba(histogram, n_pixels)
    accumulated_proba = get_accumulated_proba(hist_proba)
    new_gray_value = get_new_gray_value(accumulated_proba)
    eq_img = equalize_hist(image.copy(), new_gray_value)



def classificarRN():
    img_path = caminhoDaImagemAberta
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    classificacao = model.predict(img)

    # print('Resultado:', decode_predictions(classificacao, top=3)[0])
    labelClassificacao.config(text = classificacao)

def treinarRede():
    global model
    model = EfficientNetV2B0(weights = None, classes = 1, input_shape = (224, 224, 1))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy() ,optimizer= "adam", metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=5, epochs = 3, validation_split=0.1)#validation_data = (X_val, y_val))

    btnClassificarRN.config(state=ACTIVE) # Habilitar classificação

def carregarDataset():
    global trainingData, DATADIR, X_train, y_train, X_val, y_val

    # Diretorio de Treino
    DATADIR = askdirectory(title='Diretório de Treino')

    for categoria in CATEGORIAS:
        path = os.path.join(DATADIR, categoria)
        classNum = CATEGORIAS.index(categoria)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            trainingData.append([img_array, classNum])

    random.shuffle(trainingData)

    for features, label in trainingData:
        X_train.append(features)
        y_train.append(label)


    # Diretório de Validação
    validationData = []
    DATADIR = askdirectory(title='Diretório de Validação')
    for categoria in CATEGORIAS:
        path = os.path.join(DATADIR, categoria)
        classNum = CATEGORIAS.index(categoria)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            validationData.append([img_array, classNum])

    for features, label in validationData:
        X_val.append(features)
        y_val.append(label)


    X_train = np.array(X_train).reshape(-1, tamanhoDaImagem, tamanhoDaImagem, 1)
    y_train = np.array(y_train)

    X_val = np.array(X_val).reshape(-1, tamanhoDaImagem, tamanhoDaImagem, 1)
    y_val = np.array(y_val)

    btnTreinarRede.config(state=ACTIVE) # Habilitar treino




#Etapa 3 Parte 2

# modelo = MLPClassifier()
# modelo.fit(X_treino, y_treino)

# previsoesDefault = modelo.predict(X_teste)
# cm = ConfusionMatrix(modelo)
# cm.score(X_teste, y_teste)

# print(classification_report(y_teste, previsoesDefault))

#Etapa 4 Parte 2

#EfficientNetV2, XGBoost

#modelo = MLPClassifier()#
#modelo.fit(X_treino, y_treino)#

#Etapa 5 Parte 2

# Metricas


# Acuracia

# metrics.accuracy_score(y_under, y_predict)

# Matriz de confusao:

# metrics.confusion_matrix(y_under, y_predict)

# Classification Report:( Displays the precision, recall, F1, and support scores for the mode)

# print(metrics.classification_report(y_under, y_predict))


# Criação de componentes
titulo = tk.Label(
    text="Diagnóstico de Osteoartrite Femorotibial",
    height=2,
    width=50
)
btnAbrirImagem = tk.Button(
    text="Abrir Imagem",
    height= btnHeight,
    width= btnWidth,
    command= carregarImagem
)
btnRecortarImagem = tk.Button(
    text="Recortar Imagem",
    height= btnHeight,
    width= btnWidth,
    state= DISABLED,
    command= recortarImagem
)
btnCorrelacionarImagem = tk.Button(
    text="Correlacionar Imagem",
    height= btnHeight,
    width= btnWidth,
    state= DISABLED,
    command= correlacionarImagem
)
btnCarregarDataset = tk.Button(
    text="Carregar Dataset",
    height= btnHeight,
    width= btnWidth,
    command= carregarDataset
)
btnTreinarRede = tk.Button(
    text="Treinar Rede Neural",
    height= btnHeight,
    width= btnWidth,
    state= DISABLED,
    command= treinarRede
)
btnClassificarRN = tk.Button(
    text="Classificar (Rede Neural)",
    height= btnHeight,
    width= btnWidth,
    state= DISABLED,
    command= classificarRN
)
labelImagem = tk.Label(
    image=imgSelecionada,
    height=15,
    width=30
)
labelClassificacao = tk.Label(
    text="",
    height=2,
    width=10
)


# Adicionar componentes à tela
titulo.pack()
btnAbrirImagem.pack()
btnRecortarImagem.pack()
btnCorrelacionarImagem.pack()
btnCarregarDataset.pack()
btnTreinarRede.pack()
btnClassificarRN.pack()
labelImagem.pack()
labelClassificacao.pack()
janela.mainloop()
