import tkinter as tk
from ejercicio1 import mostrar_ventana as ejercicio1
from ejercicio2 import mostrar_ventana as ejercicio2
from ejercicio3 import mostrar_ventana as ejercicio3
from ejercicio4 import mostrar_ventana as ejercicio4
from ejercicio5 import mostrar_ventana as ejercicio5
from ejercicio6 import mostrar_ventana as ejercicio6

ventana = tk.Tk()
ventana.title("Ejercicios de Redes Neuronales")
ventana.geometry("400x450")
ventana.configure(bg="#1e1e1e")

estilo = {
    "width": 30,
    "height": 2,
    "bg": "#007acc",
    "fg": "white",
    "font": ("Arial", 12),
    "activebackground": "#005f99"
}

tk.Label(ventana, text="Menú de Ejercicios IA", font=("Arial", 16, "bold"), fg="white", bg="#1e1e1e").pack(pady=20)

tk.Button(ventana, text="1. Red", command=ejercicio1, **estilo).pack(pady=5)
tk.Button(ventana, text="2. Red", command=ejercicio2, **estilo).pack(pady=5)
tk.Button(ventana, text="3. Red", command=ejercicio3, **estilo).pack(pady=5)
tk.Button(ventana, text="4. Red", command=ejercicio4, **estilo).pack(pady=5)
tk.Button(ventana, text="5. Red", command=ejercicio5, **estilo).pack(pady=5)
tk.Button(ventana, text="6. Red", command=ejercicio6, **estilo).pack(pady=5)

ventana.mainloop()
