# Eduardo Pereira Costa - 650503
# Rafael Maia - 635921

from tkinter import filedialog

# Abre o explorador de arquivos em busca da imagem
def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "/", 
                                            title = "Select a File", 
                                            filetypes = (
                                                ( "all files", "*.*" ),
                                                ( "Text files", "*.txt*") 
                                            )
                                        )
    return filename
