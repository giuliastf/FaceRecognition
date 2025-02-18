from tkinter import ttk

def apply_styles(root):
    style = ttk.Style()
    style.configure('TFrame', background='#C0C0C0')  # Change background color for the main frame
    style.configure('TLabel', background='#C0C0C0', foreground='white')  # Update label background color

