#importacion de librerias
import tkinter as tk #Creación de interfaz gráfica
from tkinter import filedialog, messagebox #PAara abrir archivos y mostrar mensajes
import numpy as np # Para trabajar con números
import pandas as pd  # Para manejar datos en formato tabla
import matplotlib.pyplot as plt # Para hacer gráficos
from sklearn.model_selection import KFold # Para dividir los datos en grupos (validación cruzada)
from sklearn.preprocessing import MinMaxScaler # Para normalizar los datos
from tensorflow.keras.models import Sequential # Para crear el modelo de red neuronal
from tensorflow.keras.layers import Dense, Input # Capas del modelo
from tensorflow.keras.utils import plot_model  # Para guardar un diagrama del modelo en imagen
from tensorflow.keras import backend as K # Para liberar memoria entre entrenamientos
import os
import tensorflow as tf
import random

# --- CONFIGURACIÓN ---
k_folds = 5 #numero de divisiones para validación cruzada

normalizar = True 

topologias = [ #diferentes estructuras de red a probar 
    [32, 16]
]
activaciones = ['relu', 'tanh', 'sigmoid'] #tipos de funciones que activan las neuronas
epocas = 100 # Cuantas veces se entrena el modelo sobre el archivo
batch_size = 10 # Cantidad de datos que se usan en cada paso del entrenamiento

# Semillas para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Funcion para sleccionar archivo 
def seleccionar_archivo():
    ruta = filedialog.askopenfilename(title="Selecciona el archivo de entrenamiento")
    if ruta:
        ejecutar_experimento(ruta)

#funcion principal para raealizar analiss
def ejecutar_experimento(ruta_archivo):
    try:
        #Leer archivo cvs 
        datos = pd.read_csv(ruta_archivo, header=None)
        #se separa en variables x y y se salida
        X = datos.iloc[:, :-1].values
        y = datos.iloc[:, -1].values


#si está activado, se normalizan los datos entre 0 y 1
        if normalizar:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

#sE dividen los datos en k grupos para validación cruzada
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Se prueba cada combinación de arquitectura y función de activación
        for topologia in topologias:
            for activacion in activaciones:
                print(f"\nTopología: {topologia}, Activación: {activacion}")
                fold_scores = []

    # Se entrena y evalua con cada grupo (fold)
                for j, (train_index, test_index) in enumerate(kf.split(X)):
                    print(f"Fold {j+1} - Entrenando modelo...")

                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

        # Se crea la red neuronal con la estructura y función de activación seleccionadas 
                    model = Sequential()
                    model.add(Input(shape=(X.shape[1],))) # Capa de entrada
                    model.add(Dense(topologia[0], activation=activacion, name="entrada"))

        # Ccapas ocultas según la topología
                    for k, units in enumerate(topologia[1:], start=1):
                        model.add(Dense(units, activation=activacion, name=f"oculta_{k}"))

        #capa de salida con activación sigmoid para clasificación binaria
                    model.add(Dense(1, activation='sigmoid', name="salida"))

        #Compila el modelo, prepara para entrenamiento 
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
         #Se entrena el modelo con los datos
                    model.fit(X_train, y_train, epochs=epocas, batch_size=batch_size,
                              verbose=0, validation_data=(X_test, y_test))

        # Evalua la precisión del modelo en los datos de prueba
                    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                    print(f"Fold {j+1}: Exactitud = {accuracy:.4f}")
                    fold_scores.append(accuracy)
            
            #limpi la memoria del modelo antes de crear otro
                    K.clear_session()

            #Imprime el promedio de precisión de los folds
                promedio = np.mean(fold_scores)
                print(f"Promedio de exactitud: {promedio:.4f}")

                # guarda una imagen con la estructura del modelo
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

        #muestra en pantalla un mensaje cuando el análisis termina
        messagebox.showinfo("Finalizado", "El análisis ha terminado. Revisa la consola y las imágenes generadas.")

    except Exception as e:
        # Si ocurre un error muestra el mensaje
        messagebox.showerror("Error", str(e))

#Diseño de intrefaz para ingresar archivo
ventana = tk.Tk()
ventana.title("Análisis de Red Neuronal Feedforward")
ventana.geometry("400x150")

etiqueta = tk.Label(ventana, text="Selecciona el archivo de datos para comenzar")
etiqueta.pack(pady=20)

boton_cargar = tk.Button(ventana, text="Seleccionar archivo", command=seleccionar_archivo)
boton_cargar.pack()

ventana.mainloop()
