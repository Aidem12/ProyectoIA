import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import os

# --- CONFIGURACIÓN ---
k_folds = 5
normalizar = True
topologias = [
    [32, 16],
    [64, 32],
    [128, 64, 16]
]
activaciones = ['relu', 'tanh', 'sigmoid']
epocas = 100
batch_size = 10

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

                for j, (train_index, test_index) in enumerate(kf.split(X)):
                    print(f"Fold {j+1} - Entrenando modelo...")

                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    model = Sequential()
                    model.add(Input(shape=(X.shape[1],)))
                    model.add(Dense(topologia[0], activation=activacion, name="entrada"))

                    for k, units in enumerate(topologia[1:], start=1):
                        model.add(Dense(units, activation=activacion, name=f"oculta_{k}"))

                    model.add(Dense(1, activation='sigmoid', name="salida"))

                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    model.fit(X_train, y_train, epochs=epocas, batch_size=batch_size,
                              verbose=0, validation_data=(X_test, y_test))

                    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                    print(f"Fold {j+1}: Exactitud = {accuracy:.4f}")
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

                # Limpiar sesión de Keras después de usar el modelo
                K.clear_session()

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
