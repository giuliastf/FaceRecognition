from tkinter import ttk

def apply_styles(root):
    style = ttk.Style()
    style.configure('TFrame', background='#34e5eb')  # Match the main window background
    style.configure('TLabel', background='#34e5eb', foreground='#2c3e50', font=('Helvetica', 12))

