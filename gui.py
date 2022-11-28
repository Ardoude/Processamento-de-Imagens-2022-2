# Eduardo Pereira Costa - 650503
# Rafael Maia - 635921

from pickle import FALSE, TRUE
from re import X
import utils
import pandas as pd
import tkinter as tk
from tkinter import ACTIVE, DISABLED
from tkinter.filedialog import askdirectory
from PIL import ImageTk, Image
import cv2
import numpy as np
from skimage.feature import match_template
from matplotlib import pyplot as plt

# Constantes
btnHeight = 1  # Altura padrão dos botões
btnWidth = 20  # Largura padrão dos botões
imgSelecionada = None # Imagem a ser classificada
filename = "" # Caminho da imagem selecionada
dirPath = "" # Diretorio com a base a ser treinada
imagemARecortar = None # Imagem a ser recortada (Cópia)
cropping = False # Imagem sendo recortada
cropped = False # Imagem foi recortada
x_start, y_start, x_end, y_end = 0, 0, 0, 0 # Coordenadas de recorte da imagem

# Configuração da tela
janela = tk.Tk(className= ' Trabalho Prático - Processamento de Imagens')
janela.geometry("350x500")
    
# Métodos utilitários

# Atualiza a imagem a ser exibida
def atualizaImagem(path):
    global imgSelecionada, filename
    imgSelecionada = ImageTk.PhotoImage( Image.open(path).resize( (255, 255), resample=3) )
    labelImagem.config(image = imgSelecionada, height=255, width=255)
    labelImagem.img = imgSelecionada
    filename = path

# Carrega uma imagem por um endereço e a exibe na tela
def carregarImagem():
    global filename, imgSelecionada
    filename = utils.browseFiles()
    if filename!="":
        atualizaImagem(filename)
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
        pontoDeReferencia = [(x_start, y_start), (x_end, y_end)]
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
        imagemARecortar = cv2.imread(filename).copy()

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
    imgReferencia = cv2.imread(filename)
    
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
labelImagem = tk.Label(
    image=imgSelecionada,
    height=15,
    width=30
)

#treino = ImageTk.PhotoImage( Image.open(path).resize( (255, 255), resample=3) )#
#valiadacao = ImageTk.PhotoImage( Image.open(path).resize( (255, 255), resample=3) )#
#teste = ImageTk.PhotoImage( Image.open(path).resize( (255, 255), resample=3) )#


# Adicionar componentes à tela
titulo.pack()
btnAbrirImagem.pack()
btnRecortarImagem.pack()
btnCorrelacionarImagem.pack()
labelImagem.pack()

janela.mainloop()
