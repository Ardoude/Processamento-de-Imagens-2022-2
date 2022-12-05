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
from xgboost import XGBClassifier
from sklearn import svm
import random
import time
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from pandas import crosstab

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
X_test = []
y_test = []

model = None
modelXG = None
modelSVM = None

# Configuração da tela
janela = tk.Tk(className= ' Trabalho Prático - Processamento de Imagens')
janela.geometry("350x600")
    
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




def instantiate_histogram():    
    hist_array= []
    
    for i in range(0,256):
        hist_array.append(str(i))
        hist_array.append(0)
    
    hist_dct = {hist_array[i]: hist_array[i + 1] for i in range(0, len(hist_array), 2)} 
    
    return hist_dct

def count_intensity_values(hist, img):
    for row in img:
        for column in row:
            hist[str(int(column))] = hist[str(int(column))] + 1
     
    return hist

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
        
        
def get_hist_proba(hist, n_pixels):
    hist_proba = {}
    for i in range(0, 256):
        hist_proba[str(i)] = hist[str(i)] / n_pixels
    
    return hist_proba

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

def get_new_gray_value(acc_proba):
    new_gray_value = {}
    
    for i in range(0, 256):
        new_gray_value[str(i)] = np.ceil(acc_proba[str(i)] * 255)
        
    return new_gray_value

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



def classificarSVM():
    img_path = caminhoDaImagemAberta
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = tf.expand_dims(tf.expand_dims(img,2), 0)

    classificacao = modelSVM.predict(img)

    # print('Resultado:', decode_predictions(classificacao, top=3)[0])
    labelClassificacao.config(text = classificacao)

def classificarRN():
    img_path = caminhoDaImagemAberta
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = tf.expand_dims(tf.expand_dims(img,2), 0)
    classificacao = model.predict(img)

    # print('Resultado:', decode_predictions(classificacao, top=3)[0])
    labelClassificacao.config(text = classificacao)

def classificarXGBoost():
    img_path = caminhoDaImagemAberta
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = tf.expand_dims(tf.expand_dims(img,2), 0)

    classificacao = modelXG.predict(img)

    # print('Resultado:', decode_predictions(classificacao, top=3)[0])
    labelClassificacao.config(text = classificacao)

def obterMetricas(y_pred, t2, tClassificacao):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    confMatrix = confusion_matrix(y_test, y_pred)

    FP = confMatrix.sum(axis=0) - np.diag(confMatrix)
    FN = confMatrix.sum(axis=1) - np.diag(confMatrix)
    TP = np.diag(confMatrix)
    TN = confMatrix.sum() - (FP + FN + TP)
    TPR = TP / (TP + FN)  # Sensibilidade ou hit rate
    TNR = TN / (TN + FP)  # Specificidade
    
    sensibilidade = round(TPR.mean(), 2)
    especificidade = round(TNR.mean(), 2)

    # Nova janela com dados do treino
    novaJanela = tk.Toplevel(janela)
    novaJanela.title("Informações de Treino")
    novaJanela.geometry("400x500")
    tk.Label( novaJanela, text="Accuracy: " + str(accuracy)).pack()
    tk.Label( novaJanela, text="Sensibilidade: " + str(sensibilidade)).pack()
    tk.Label( novaJanela, text="Especificidade: " + str(especificidade)).pack()
    tk.Label( novaJanela, text="Tempo de treino: " + str(round(t2, 4)) + "s").pack()
    tk.Label( novaJanela, text="Tempo gasto classificando: " + str(round(tClassificacao, 4)) + "s").pack()
    tk.Label( novaJanela, text="Matriz de Confusão\n" + str(crosstab(y_train, y_val, rownames=['Real'], colnames=['Predito'], margins=True))).pack()

    novaJanela.mainloop()

def treinarSVM():
    global modelSVM
    modelSVM = svm.SVC(kernel='linear', C=1, gamma=1) # Classificador
    
    # Treinar
    t1 = time.time()
    modelSVM.fit(X_train, y_train)
    t2 = time.time() - tInicial

    # Testar resto da base
    tInicial = time.time()
    y_pred = modelSVM.predict(X_test)
    tClassificacao = time.time() - tInicial

    btnClassificarSVM.config(state=ACTIVE) # Habilitar classificação

    # Obter dados
    obterMetricas(y_pred, t2, tClassificacao)


def treinarXGBoost():
    global modelXG
    modelXG = XGBClassifier(max_depth=4, booster= "dart", learning_rate= 0.25)
    
    t1 = time.time()
    modelXG.fit(X_train, y_train)
    t2 = time.time() - t1

    # Testar resto da base
    tInicial = time.time()
    y_pred = modelXG.predict(X_test)
    tClassificacao = time.time() - tInicial

    btnClassificarXG.config(state=ACTIVE) # Habilitar classificação

    # Obter dados
    obterMetricas(y_pred, t2, tClassificacao)



def treinarRede():
    global model
    model = EfficientNetV2B0(weights = None, classes = 1, input_shape = (224, 224, 1))
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer= "adam", metrics=['accuracy'])

    # Treinar
    t1 = time.time()
    model.fit(X_train, y_train, batch_size=5, epochs = 3, validation_split=0.1)#validation_data = (X_val, y_val))
    t2 = time.time() - t1

    # Testar resto da base
    tInicial = time.time()
    y_pred = model.predict(X_test)
    tClassificacao = time.time() - tInicial

    btnClassificarRN.config(state=ACTIVE) # Habilitar classificação

    # Obter dados
    obterMetricas(y_pred, t2, tClassificacao)
    

    

def carregarDataset():
    global trainingData, DATADIR, X_train, y_train, X_val, y_val, X_test, y_test

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

    
    # Diretório de Teste
    testData = []
    DATADIR = askdirectory(title='Diretório de Teste')
    for categoria in CATEGORIAS:
        path = os.path.join(DATADIR, categoria)
        classNum = CATEGORIAS.index(categoria)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            testData.append([img_array, classNum])

    for features, label in testData:
        X_test.append(features)
        y_test.append(label)


    X_test = np.array(X_test).reshape(-1, tamanhoDaImagem, tamanhoDaImagem, 1)
    y_test = np.array(y_test)

    X_train = np.array(X_train).reshape(-1, tamanhoDaImagem, tamanhoDaImagem, 1)
    y_train = np.array(y_train)

    X_val = np.array(X_val).reshape(-1, tamanhoDaImagem, tamanhoDaImagem, 1)
    y_val = np.array(y_val)

    btnTreinarRede.config(state=ACTIVE) # Habilitar treino
    btnTreinarXG.config(state=ACTIVE) # Habilitar treino
    btnTreinarSVM.config(state=ACTIVE) # Habilitar treino



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
btnTreinarXG = tk.Button(
    text="Treinar XGBoost",
    height= btnHeight,
    width= btnWidth,
    state= DISABLED,
    command= treinarXGBoost
)
btnTreinarSVM = tk.Button(
    text="Treinar SVM",
    height= btnHeight,
    width= btnWidth,
    state= DISABLED,
    command= treinarSVM
)
btnClassificarRN = tk.Button(
    text="Classificar (Rede Neural)",
    height= btnHeight,
    width= btnWidth,
    state= DISABLED,
    command= classificarRN
)
btnClassificarXG = tk.Button(
    text="Classificar (XGBoost)",
    height= btnHeight,
    width= btnWidth,
    state= DISABLED,
    command= classificarXGBoost
)
btnClassificarSVM = tk.Button(
    text="Classificar (SVM)",
    height= btnHeight,
    width= btnWidth,
    state= DISABLED,
    command= classificarSVM
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
btnTreinarXG.pack()
btnTreinarSVM.pack()
btnClassificarRN.pack()
btnClassificarXG.pack()
btnClassificarSVM.pack()
labelImagem.pack()
labelClassificacao.pack()
janela.mainloop()
