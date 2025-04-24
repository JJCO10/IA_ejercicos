import tkinter as tk

def mostrar_ventana():
    ventana = tk.Toplevel()
    ventana.title("Ejercicio 6:")
    ventana.geometry("400x300")
    ventana.configure(bg="#1e1e1e")

    tk.Label(ventana, text="Ejercicio 6", font=("Arial", 14, "bold"), fg="white", bg="#1e1e1e").pack(pady=50)
    tk.Label(ventana, text="Aquí irá el contenido del ejercicio.", font=("Arial", 12), fg="white", bg="#1e1e1e").pack()
