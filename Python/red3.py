import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU  # Se añade LeakyReLU
from tensorflow.keras.utils import plot_model
import os

# --- CONFIGURACIÓN ---
k_folds = 5
normalizar = True
topologias = [
    [8, 4]
]
activaciones = ['relu', 'tanh', 'sigmoid', 'leaky_relu']  # Se añade leaky_relu
epocas = 50
batch_size = 8

# --- INTERFAZ PARA CARGAR ARCHIVO ---
def seleccionar_archivo():
    ruta = filedialog.askopenfilename(title="Selecciona el archivo de entrenamiento")
    if ruta:
        ejecutar_experimento(ruta)

def ejecutar_experimento(ruta_archivo):
    try:
        # Cargar datos
        datos = pd.read_csv(ruta_archivo, header=None)
        X = datos.iloc[:, :-1].values
        y = datos.iloc[:, -1].values

        if normalizar:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        for topologia in topologias:
            for activacion in activaciones:
                print(f"\nTopología: {topologia}, Activación: {activacion}")
                fold_scores = []

                for i, (train_index, test_index) in enumerate(kf.split(X)):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    model = Sequential()

                    # Capa de entrada
                    if activacion == 'leaky_relu':
                        model.add(Dense(topologia[0], input_dim=X.shape[1]))
                        model.add(LeakyReLU(alpha=0.01))
                    else:
                        model.add(Dense(topologia[0], input_dim=X.shape[1], activation=activacion, name="entrada"))

                    # Capas ocultas
                    for i, units in enumerate(topologia[1:], start=1):
                        if activacion == 'leaky_relu':
                            model.add(Dense(units))
                            model.add(LeakyReLU(alpha=0.01))
                        else:
                            model.add(Dense(units, activation=activacion, name=f"oculta_{i}"))

                    # Capa de salida
                    model.add(Dense(1, activation='sigmoid', name="salida"))

                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    history = model.fit(X_train, y_train, epochs=epocas, batch_size=batch_size,
                                        verbose=0, validation_data=(X_test, y_test))

                    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                    print(f"Fold {i+1}: Exactitud = {accuracy:.4f}")
                    fold_scores.append(accuracy)

                promedio = np.mean(fold_scores)
                print(f"Promedio de exactitud: {promedio:.4f}")

                # Visualizar topología
                nombre_png = f"modelo_topologia_{'_'.join(map(str, topologia))}_{activacion}.png"
                plot_model(model, to_file=nombre_png, show_shapes=True, show_layer_names=True)
                print(f"Topología guardada en: {nombre_png}")

                # Graficar resultados de los folds
                plt.figure()
                plt.bar(range(1, k_folds + 1), fold_scores)
                plt.ylim(0, 1)
                plt.xlabel('Fold')
                plt.ylabel('Exactitud')
                plt.title(f'Topología {topologia} con activación {activacion}')
                plt.savefig(f"grafica_fold_{'_'.join(map(str, topologia))}_{activacion}.png")
                plt.close()

        messagebox.showinfo("Finalizado", "El análisis ha terminado. Revisa la consola y las imágenes generadas.")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# --- INTERFAZ GRÁFICA BÁSICA ---
ventana = tk.Tk()
ventana.title("Análisis de Red Neuronal Feedforward")
ventana.geometry("400x150")

etiqueta = tk.Label(ventana, text="Selecciona el archivo de datos para comenzar")
etiqueta.pack(pady=20)

boton_cargar = tk.Button(ventana, text="Seleccionar archivo", command=seleccionar_archivo)
boton_cargar.pack()

ventana.mainloop()
