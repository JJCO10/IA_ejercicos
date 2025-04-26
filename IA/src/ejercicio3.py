import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import json
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def mostrar_ventana():
    window = tk.Toplevel()
    app = NeuralNetworkGUI(window)
    return window

class NeuralNetworkGUI:
    def __init__(self, master):
        self.master = master
        master.title("Red Neuronal - Ejercicio 3")
        master.configure(bg="#1e1e1e")
        
        # Variables de configuración
        self.layer_config = []
        self.activation_var = tk.StringVar(value='relu')
        self.solver_var = tk.StringVar(value='adam')
        self.learning_rate_var = tk.StringVar(value='constant')
        self.learning_rate_init_var = tk.DoubleVar(value=0.001)
        self.max_iter_var = tk.IntVar(value=200)
        self.random_state_var = tk.IntVar(value=1)
        
        # Variables para entradas/salidas
        self.input_vars = []
        self.output_vars = []
        self.X_cols = []
        self.y_cols = []
        
        # Dataset
        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.is_classification = True
        
        # Estilo
        self.style = {
            "bg": "#252526",
            "fg": "white",
            "font": ("Arial", 10),
            "insertbackground": "white"
        }
        
        self.button_style = {
            "bg": "#007acc",
            "fg": "white",
            "activebackground": "#005f99",
            "font": ("Arial", 10)
        }
        
        # Crear interfaz
        self.create_widgets()
        
    def create_widgets(self):
        # Panel de configuración
        config_frame = tk.LabelFrame(self.master, text="Configuración de la Red", 
                                   bg="#1e1e1e", fg="white", font=("Arial", 12, "bold"))
        config_frame.pack(padx=10, pady=10, fill="x")
        
        # Configuración de capas
        tk.Label(config_frame, text="Configuración de Capas (ej. 10,5):", 
                bg="#1e1e1e", fg="white").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.layer_entry = tk.Entry(config_frame, **self.style)
        self.layer_entry.grid(row=0, column=1, padx=5, pady=2)
        self.layer_entry.insert(0, "10, 5")
        
        tk.Button(config_frame, text="Aplicar", command=self.apply_layer_config, 
                **self.button_style).grid(row=0, column=2, padx=5)
        
        # Función de activación
        tk.Label(config_frame, text="Función de Activación:", 
                bg="#1e1e1e", fg="white").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        activation_menu = ttk.OptionMenu(config_frame, self.activation_var, 
                                       'relu', 'identity', 'logistic', 'tanh', 'relu')
        activation_menu.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        
        # Algoritmo de entrenamiento
        tk.Label(config_frame, text="Algoritmo:", 
                bg="#1e1e1e", fg="white").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        solver_menu = ttk.OptionMenu(config_frame, self.solver_var, 
                                    'adam', 'lbfgs', 'sgd', 'adam')
        solver_menu.grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        
        # Tasa de aprendizaje
        tk.Label(config_frame, text="Tasa de Aprendizaje:", 
                bg="#1e1e1e", fg="white").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        lr_menu = ttk.OptionMenu(config_frame, self.learning_rate_var, 
                                'constant', 'constant', 'invscaling', 'adaptive')
        lr_menu.grid(row=3, column=1, padx=5, pady=2, sticky="ew")
        
        tk.Label(config_frame, text="Valor Inicial:", 
                bg="#1e1e1e", fg="white").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(config_frame, textvariable=self.learning_rate_init_var, 
                **self.style).grid(row=4, column=1, padx=5, pady=2)
        
        # Parámetros de entrenamiento
        tk.Label(config_frame, text="Épocas Máximas:", 
                bg="#1e1e1e", fg="white").grid(row=5, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(config_frame, textvariable=self.max_iter_var, 
                **self.style).grid(row=5, column=1, padx=5, pady=2)
        
        tk.Label(config_frame, text="Semilla Aleatoria:", 
                bg="#1e1e1e", fg="white").grid(row=6, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(config_frame, textvariable=self.random_state_var, 
                **self.style).grid(row=6, column=1, padx=5, pady=2)
        
        # Cargar dataset
        tk.Label(config_frame, text="Dataset JSON:", 
                bg="#1e1e1e", fg="white").grid(row=7, column=0, sticky="w", padx=5, pady=2)
        tk.Button(config_frame, text="Cargar Dataset", command=self.load_json_dataset, 
                **self.button_style).grid(row=7, column=1, padx=5, pady=2, sticky="ew")
        
        # Frame para selección de entradas/salidas
        self.io_frame = tk.LabelFrame(self.master, text="Selección de Entradas/Salidas", 
                                    bg="#1e1e1e", fg="white", font=("Arial", 12, "bold"))
        self.io_frame.pack(padx=10, pady=5, fill="x")
        
        # Botones de acción
        action_frame = tk.Frame(self.master, bg="#1e1e1e")
        action_frame.pack(padx=10, pady=10, fill="x")
        
        tk.Button(action_frame, text="Entrenar Red", command=self.train_network, 
                **self.button_style).pack(side="left", padx=5)
        tk.Button(action_frame, text="Evaluar Modelo", command=self.evaluate_model, 
                **self.button_style).pack(side="left", padx=5)
        tk.Button(action_frame, text="Simular", command=self.simulate, 
                **self.button_style).pack(side="left", padx=5)
        tk.Button(action_frame, text="Visualizar", command=self.visualize, 
                **self.button_style).pack(side="left", padx=5)
        
        # Área de resultados
        result_frame = tk.LabelFrame(self.master, text="Resultados", 
                                   bg="#1e1e1e", fg="white", font=("Arial", 12, "bold"))
        result_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.result_text = tk.Text(result_frame, height=10, width=60, 
                                 bg="#252526", fg="white", insertbackground="white")
        self.result_text.pack(fill="both", expand=True, padx=5, pady=5)
        
    def apply_layer_config(self):
        try:
            layer_str = self.layer_entry.get()
            self.layer_config = [int(x.strip()) for x in layer_str.split(",")]
            self.result_text.insert("end", f"Configuración de capas aplicada: {self.layer_config}\n")
        except ValueError:
            messagebox.showerror("Error", "Formato incorrecto. Use números separados por comas (ej. '10, 5')")
    
    def load_json_dataset(self):
        filepath = filedialog.askopenfilename(title="Seleccionar archivo JSON", 
                                            filetypes=[("JSON files", "*.json")])
        if not filepath:
            return
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convertir a DataFrame de pandas
            df = pd.DataFrame(data)
            
            # Limpiar frame de selección anterior
            for widget in self.io_frame.winfo_children():
                widget.destroy()
            
            # Crear checkboxes para cada columna
            self.input_vars = []
            self.output_vars = []
            all_columns = list(df.columns)
            
            tk.Label(self.io_frame, text="Seleccione entradas:", 
                    bg="#1e1e1e", fg="white").grid(row=0, column=0, padx=5, pady=2, sticky="w")
            tk.Label(self.io_frame, text="Seleccione salidas:", 
                    bg="#1e1e1e", fg="white").grid(row=0, column=1, padx=5, pady=2, sticky="w")
            
            for i, col in enumerate(all_columns):
                # Checkbox para entradas
                var_in = tk.BooleanVar(value=True if i < len(all_columns)-1 else False)
                cb_in = tk.Checkbutton(self.io_frame, text=col, variable=var_in,
                                     bg="#1e1e1e", fg="white", selectcolor="#252526")
                cb_in.grid(row=i+1, column=0, padx=5, pady=2, sticky="w")
                self.input_vars.append((col, var_in))
                
                # Checkbox para salidas
                var_out = tk.BooleanVar(value=True if i >= len(all_columns)-1 else False)
                cb_out = tk.Checkbutton(self.io_frame, text=col, variable=var_out,
                                      bg="#1e1e1e", fg="white", selectcolor="#252526")
                cb_out.grid(row=i+1, column=1, padx=5, pady=2, sticky="w")
                self.output_vars.append((col, var_out))
            
            # Botón para confirmar selección
            tk.Button(self.io_frame, text="Confirmar Selección", 
                     command=lambda: self.process_dataset_selection(df, filepath),
                     **self.button_style).grid(row=len(all_columns)+1, column=0, columnspan=2, pady=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el dataset: {str(e)}")
    
    def process_dataset_selection(self, df, filepath):
        try:
            # Obtener columnas seleccionadas
            self.X_cols = [col for col, var in self.input_vars if var.get()]
            self.y_cols = [col for col, var in self.output_vars if var.get()]
            
            if not self.X_cols or not self.y_cols:
                messagebox.showerror("Error", "Seleccione al menos una entrada y una salida")
                return
            
            # Verificar si hay superposición
            overlap = set(self.X_cols) & set(self.y_cols)
            if overlap:
                messagebox.showerror("Error", f"Las siguientes columnas están en entradas y salidas: {overlap}")
                return
            
            # Separar características (X) y etiquetas (y)
            self.X = df[self.X_cols].values
            self.y = df[self.y_cols].values
            
            # Determinar si es problema de clasificación (valores enteros) o regresión
            unique_vals = len(np.unique(self.y))
            self.is_classification = unique_vals < 0.1 * len(self.y) or np.issubdtype(self.y.dtype, np.integer)
            
            # Dividir en train/test
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=self.random_state_var.get())
            
            # Escalar datos
            self.X_train = self.scaler_X.fit_transform(self.X_train)
            self.X_test = self.scaler_X.transform(self.X_test)
            
            if not self.is_classification:
                self.y_train = self.scaler_y.fit_transform(self.y_train)
                self.y_test = self.scaler_y.transform(self.y_test)
            
            self.result_text.insert("end", f"Dataset cargado desde {filepath}\n")
            self.result_text.insert("end", f"Entradas: {self.X_cols} ({self.X.shape[1]} características)\n")
            self.result_text.insert("end", f"Salidas: {self.y_cols} ({self.y.shape[1]} variables)\n")
            self.result_text.insert("end", f"Tipo de problema: {'Clasificación' if self.is_classification else 'Regresión'}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar dataset: {str(e)}")
    
    def train_network(self):
        if not self.layer_config:
            messagebox.showerror("Error", "Configure las capas primero")
            return
            
        if self.X is None:
            messagebox.showerror("Error", "Cargue un dataset primero")
            return
            
        try:
            if self.is_classification:
                self.model = MLPClassifier(
                    hidden_layer_sizes=self.layer_config,
                    activation=self.activation_var.get(),
                    solver=self.solver_var.get(),
                    learning_rate=self.learning_rate_var.get(),
                    learning_rate_init=self.learning_rate_init_var.get(),
                    max_iter=self.max_iter_var.get(),
                    random_state=self.random_state_var.get(),
                    verbose=True)
            else:
                self.model = MLPRegressor(
                    hidden_layer_sizes=self.layer_config,
                    activation=self.activation_var.get(),
                    solver=self.solver_var.get(),
                    learning_rate=self.learning_rate_var.get(),
                    learning_rate_init=self.learning_rate_init_var.get(),
                    max_iter=self.max_iter_var.get(),
                    random_state=self.random_state_var.get(),
                    verbose=True)
            
            self.result_text.insert("end", "Entrenando red neuronal...\n")
            self.master.update()
            
            self.model.fit(self.X_train, self.y_train)
            
            self.result_text.insert("end", "Entrenamiento completado!\n")
            
            # Calcular métricas
            if self.is_classification:
                train_score = self.model.score(self.X_train, self.y_train)
                self.result_text.insert("end", f"Precisión en entrenamiento: {train_score:.2f}\n")
            else:
                from sklearn.metrics import mean_squared_error, r2_score
                y_train_pred = self.model.predict(self.X_train)
                mse = mean_squared_error(self.y_train, y_train_pred)
                r2 = r2_score(self.y_train, y_train_pred)
                self.result_text.insert("end", f"MSE en entrenamiento: {mse:.4f}\n")
                self.result_text.insert("end", f"R² en entrenamiento: {r2:.4f}\n")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en el entrenamiento: {str(e)}")
    
    def evaluate_model(self):
        if self.model is None:
            messagebox.showerror("Error", "Entrene el modelo primero")
            return
            
        if self.X_test is None:
            messagebox.showerror("Error", "No hay datos de prueba disponibles")
            return
            
        try:
            if self.is_classification:
                test_score = self.model.score(self.X_test, self.y_test)
                self.result_text.insert("end", f"Precisión en prueba: {test_score:.2f}\n")
            else:
                from sklearn.metrics import mean_squared_error, r2_score
                y_test_pred = self.model.predict(self.X_test)
                mse = mean_squared_error(self.y_test, y_test_pred)
                r2 = r2_score(self.y_test, y_test_pred)
                self.result_text.insert("end", f"MSE en prueba: {mse:.4f}\n")
                self.result_text.insert("end", f"R² en prueba: {r2:.4f}\n")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en evaluación: {str(e)}")
    
    def simulate(self):
        if self.model is None:
            messagebox.showerror("Error", "Entrene el modelo primero")
            return
            
        if self.X is None:
            messagebox.showerror("Error", "No hay dataset cargado")
            return
            
        # Crear ventana de simulación
        sim_window = tk.Toplevel(self.master)
        sim_window.title("Simulación")
        sim_window.configure(bg="#1e1e1e")
        
        # Crear campos de entrada para cada característica de entrada
        inputs = []
        for i, col in enumerate(self.X_cols):
            tk.Label(sim_window, text=f"{col}:", 
                    bg="#1e1e1e", fg="white").grid(row=i, column=0, padx=5, pady=2, sticky="e")
            entry = tk.Entry(sim_window, **self.style)
            entry.grid(row=i, column=1, padx=5, pady=2)
            inputs.append(entry)
        
        # Botón de predicción
        tk.Button(sim_window, text="Predecir", 
                 command=lambda: self.make_prediction(inputs, sim_window),
                 **self.button_style).grid(
            row=len(self.X_cols), column=0, columnspan=2, pady=10)
        
        # Área de resultado
        self.prediction_result = tk.Label(sim_window, text="", 
                                        bg="#1e1e1e", fg="white", font=("Arial", 10))
        self.prediction_result.grid(row=len(self.X_cols)+1, column=0, columnspan=2)
    
    def make_prediction(self, inputs, window):
        try:
            input_values = []
            for entry in inputs:
                input_values.append(float(entry.get()))
                    
            # Escalar los valores de entrada
            scaled_input = self.scaler_X.transform([input_values])
            
            # Hacer predicción
            prediction = self.model.predict(scaled_input)
            
            # Formatear resultado
            result_text = "Resultados:\n"
            
            if self.is_classification:
                # Caso CLASIFICACIÓN (muestra clases y probabilidades)
                if len(self.y_cols) == 1:
                    result_text += f"{self.y_cols[0]}: {prediction[0]}\n"
                else:
                    for i, col in enumerate(self.y_cols):
                        result_text += f"{col}: {prediction[0][i]}\n"
                
                # Mostrar probabilidades si es clasificación
                if hasattr(self.model, "predict_proba"):
                    proba = self.model.predict_proba(scaled_input)
                    result_text += "\nProbabilidades por clase:\n"
                    for i, class_probs in enumerate(proba[0]):  # Accedemos al primer elemento [0]
                        result_text += f"Clase {i}: {class_probs:.4f}\n"
            else:
                # Caso REGRESIÓN (muestra valor predicho + intervalo de confianza)
                if len(self.y_cols) == 1:
                    # Calcular error estándar (aproximado) usando el MSE del entrenamiento
                    y_train_pred = self.model.predict(self.X_train)
                    mse = np.mean((y_train_pred - self.y_train) ** 2)
                    std_error = np.sqrt(mse)
                    
                    result_text += f"{self.y_cols[0]}: {prediction[0]:.4f}\n"
                    result_text += f"Intervalo aproximado (±2σ): [{prediction[0] - 2*std_error:.4f}, {prediction[0] + 2*std_error:.4f}]\n"
                else:
                    for i, col in enumerate(self.y_cols):
                        result_text += f"{col}: {prediction[0][i]:.4f}\n"
            
            self.prediction_result.config(text=result_text)
        except Exception as e:
            messagebox.showerror("Error", f"Error en predicción: {str(e)}")
    
    def visualize(self):
        if self.model is None or self.X.shape[1] < 2:
            messagebox.showerror("Error", "El modelo no está entrenado o no hay suficientes características para visualizar")
            return
            
        try:
            # Reducir dimensionalidad para visualización
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(self.X_train)
            
            # Crear figura con fondo oscuro
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(8, 6))
            fig.patch.set_facecolor('#1e1e1e')
            
            if self.is_classification and len(self.y_cols) == 1:
                # Visualización para clasificación con una sola salida
                ax = fig.add_subplot(111)
                ax.set_facecolor('#1e1e1e')
                
                # Graficar puntos de datos
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y_train, alpha=0.6)
                
                # Crear leyenda
                legend = ax.legend(*scatter.legend_elements(), title="Clases")
                ax.add_artist(legend)
                
                # Crear gráfico de decisión
                xx, yy = np.meshgrid(np.linspace(X_pca[:, 0].min()-1, X_pca[:, 0].max()+1, 100),
                                     np.linspace(X_pca[:, 1].min()-1, X_pca[:, 1].max()+1, 100))
                
                # Predecir para cada punto en la malla
                Z = self.model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
                Z = Z.reshape(xx.shape)
                
                # Graficar contornos
                ax.contourf(xx, yy, Z, alpha=0.2, levels=len(np.unique(self.y_train))-1)
                
                ax.set_title(f"Visualización de Clasificación - {self.y_cols[0]}", color="white")
            else:
                # Visualización para regresión o múltiples salidas
                n_outputs = self.y.shape[1] if len(self.y.shape) > 1 else 1
                n_plots = min(3, n_outputs)  # Mostrar máximo 3 gráficos
                
                for i in range(n_plots):
                    ax = fig.add_subplot(1, n_plots, i+1)
                    ax.set_facecolor('#1e1e1e')
                    
                    if n_outputs == 1:
                        y_vals = self.y_train
                        col_name = self.y_cols[0]
                    else:
                        y_vals = self.y_train[:, i]
                        col_name = self.y_cols[i]
                    
                    # Graficar puntos de datos
                    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_vals, alpha=0.6, cmap='viridis')
                    fig.colorbar(sc, ax=ax)
                    
                    ax.set_title(f"Visualización - {col_name}", color="white")
            
            # Configuración común
            for ax in fig.axes:
                ax.set_xlabel("Componente Principal 1", color="white")
                ax.set_ylabel("Componente Principal 2", color="white")
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white') 
                ax.spines['right'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
            
            fig.tight_layout()
            
            # Mostrar en interfaz
            vis_window = tk.Toplevel(self.master)
            vis_window.title("Visualización")
            vis_window.configure(bg="#1e1e1e")
            
            canvas = FigureCanvasTkAgg(fig, master=vis_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en visualización: {str(e)}")