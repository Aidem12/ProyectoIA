#importacion de librerias
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
    [5,6,4]
]
activaciones = ['relu', 'tanh', 'sigmoid'] #tipos de funciones que activan las neuronas
epocas = 40 # Cuantas veces se entrena el modelo sobre el archivo
batch_size = 10 # Cantidad de datos que se usan en cada paso del entrenamiento

# Semillas para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

#funcion principal para realizar análisis
def ejecutar_experimento(ruta_archivo):
    try:
        print(f"Leyendo archivo: {ruta_archivo}")
        #Leer archivo txt con valores separados por comas
        datos = pd.read_csv(ruta_archivo, header=None)
        #se separa en variables x y y se salida
        X = datos.iloc[:, :-1].values
        y = datos.iloc[:, -1].values

        print(f"Datos cargados: {X.shape[0]} muestras con {X.shape[1]} características")

        #si está activado, se normalizan los datos entre 0 y 1
        if normalizar:
            print("Normalizando datos...")
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

        #Se dividen los datos en k grupos para validación cruzada
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

                    # Capas ocultas según la topología
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
            
                    #limpia la memoria del modelo antes de crear otro
                    K.clear_session()

                #Imprime el promedio de precisión de los folds
                promedio = np.mean(fold_scores)
                print(f"Promedio de exactitud: {promedio:.4f}")

                # guarda una imagen con la estructura del modelo
                nombre_png = f"modelo_topologia_{'_'.join(map(str, topologia))}_{activacion}.png"
                
                # Crear un modelo temporal para guardar la estructura
                temp_model = Sequential()
                temp_model.add(Input(shape=(X.shape[1],)))
                temp_model.add(Dense(topologia[0], activation=activacion, name="entrada"))
                for k, units in enumerate(topologia[1:], start=1):
                    temp_model.add(Dense(units, activation=activacion, name=f"oculta_{k}"))
                temp_model.add(Dense(1, activation='sigmoid', name="salida"))
                
                plot_model(temp_model, to_file=nombre_png, show_shapes=True, show_layer_names=True)
                print(f"Topología guardada en: {nombre_png}")
                K.clear_session()

                # Graficar resultados de los folds
                plt.figure()
                plt.bar(range(1, k_folds + 1), fold_scores)
                plt.ylim(0, 1)
                plt.xlabel('Fold')
                plt.ylabel('Exactitud')
                plt.title(f'Topología {topologia} con activación {activacion}')
                plt.savefig(f"grafica_fold_{'_'.join(map(str, topologia))}_{activacion}.png")
                plt.close()

        print("El análisis ha terminado. Se han generado las imágenes de modelos y gráficas.")

    except Exception as e:
        # Si ocurre un error muestra el mensaje
        print(f"Error: {str(e)}")

def main():
    # Nombre del archivo a analizar (en la misma carpeta)
    nombre_archivo = "datos.txt"
    
    # Verificar si el archivo existe
    if not os.path.exists(nombre_archivo):
        print(f"Error: No se encontró el archivo '{nombre_archivo}' en el directorio actual.")
        nombre_archivo = input("Por favor, ingrese el nombre del archivo de datos: ")
        
        if not os.path.exists(nombre_archivo):
            print(f"Error: El archivo '{nombre_archivo}' no existe.")
            return
    
    print(f"Analizando archivo: {nombre_archivo}")
    ejecutar_experimento(nombre_archivo)

if __name__ == "__main__":
    main()
