from pickle import FALSE, TRUE
import utils
import tkinter as tk
from tkinter import ACTIVE, DISABLED
from tkinter.filedialog import askdirectory
from PIL import ImageTk, Image
import cv2

# Constantes
btnHeight = 1  # Altura padrão dos botões
btnWidth = 20  # Largura padrão dos botões
imgSelecionada = None # Imagem a ser classificada
filename = "" # Caminho da imagem selecionada
dirPath = "" # Diretorio com a base a ser treinada
oriImage = None
cropping = False
cropped = False
pathImagemRecortada = ""
x_start, y_start, x_end, y_end = 0, 0, 0, 0

# Configuração da tela
janela = tk.Tk(className= ' Trabalho Prático - Processamento de Imagens')
janela.geometry("350x500")
    

# Métodos utilitários

def atualizaImagem(path):
    global imgSelecionada
    imgSelecionada = ImageTk.PhotoImage( Image.open(path).resize( (255, 255), resample=3) )
    labelImagem.config(image = imgSelecionada, height=255, width=255)
    labelImagem.img = imgSelecionada

# Carrega imagem e a exibe na tela
def carregarImagem():
    global filename, imgSelecionada
    filename = utils.browseFiles()
    if filename!="":
        atualizaImagem(filename)

def recorte(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping, pathImagemRecortada, cropped, i
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
        refPoint = [(x_start, y_start), (x_end, y_end)]
        if len(refPoint) == 2: # Dois pontos encontrados
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imwrite("cropped.png", roi)
            cropped = True # Recorte já foi finalizado
            # cv2.destroyAllWindows() # Fecha a janela de recorte
            # cv2.destroyWindow("image")
            atualizaImagem("cropped.png")

def recortarImagem():
    global oriImage, cropped
    cropped = False
    if imgSelecionada == None:
        print("Erro")
        return
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", recorte)
    while not cropped:
        oriImage = cv2.imread(filename).copy()
        if not cropping:
            cv2.imshow("image", oriImage)
        elif cropping:
            cv2.rectangle(oriImage, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", oriImage)
        cv2.waitKey(1)
    cv2.destroyWindow("image")
# Criação de componentes

titulo = tk.Label(
    text="Trabalho de PI",
    height=2,
    width=10
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
    state= ACTIVE,
    command= recortarImagem
)
btnCorrelacionarImagem = tk.Button(
    text="Correlacionar Imagem",
    height= btnHeight,
    width= btnWidth,
    state= ACTIVE
)
labelImagem = tk.Label(
    image=imgSelecionada,
    height=15,
    width=30
)


# Adicionar componentes à tela
titulo.pack()
btnAbrirImagem.pack()
btnRecortarImagem.pack()
btnCorrelacionarImagem.pack()
labelImagem.pack()

janela.mainloop()