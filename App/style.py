from tkinter import ttk

def apply_styles(root):
    style = ttk.Style()
    style.configure('TFrame', background="#5a95e3")  # Match the main window background
    style.configure('TLabel', background='#5a95e3', foreground='#2c3e50', font=('Helvetica', 12))

