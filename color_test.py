import tkinter as tk

def test_colors():
    root = tk.Tk()
    root.title("Color Test")
    root.geometry("400x300")
    root.configure(bg='#34e5eb')
    
    # Test different text colors
    colors = ['#ffff00', '#ffffff', '#000000', '#ff0000', '#00ff00', '#0000ff']
    color_names = ['Yellow', 'White', 'Black', 'Red', 'Green', 'Blue']
    
    for i, (color, name) in enumerate(zip(colors, color_names)):
        btn = tk.Button(root, 
                       text=f"{name} Text", 
                       bg='#6a1fcc', 
                       fg=color,
                       font=('Helvetica', 12, 'bold'))
        btn.pack(pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    test_colors()