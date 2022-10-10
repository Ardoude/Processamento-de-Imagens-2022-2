from tkinter import filedialog

grayImg = None

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