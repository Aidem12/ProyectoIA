import random #Generar valores aleatorios
import statistics # Para calcular media y desviación estándar
from tkinter import Tk, Label, Button, filedialog, Entry, StringVar, messagebox #Interfaz gráfica


#clase para manejar archivos de entrada y salida
class ArchivoGenetico:
    def __init__(self, path_base, path_objetivo):
        self.basex = self.cargar(path_base)  # carga archivo base
        self.filex = self.cargar(path_objetivo) # cargar archovo file


    def cargar(self, path):
         #lee el rachivo y lo convierte en una lista de listas de enteros
        with open(path, "r") as f:
            return [list(map(int, line.strip().split(","))) for line in f.readlines()]

    def guardar_archivo_generado(self, contenido, nombre_archivo="Generado.txt"):
        #guarda el resultado final en un archivo
        with open(nombre_archivo, "w") as f:
            for fila in contenido:
                f.write(",".join(map(str, fila)) + "\n")

    def guardar_estadisticas(self, estadisticas, nombre_archivo="resultado.txt"):
        #guarda las estadísticas del algoritmo genético en un archivo
        with open(nombre_archivo, "w") as f:
            for linea_stats in estadisticas:
                f.write(f"{linea_stats}\n")


#clase que implementa el algoritmo genético ===
class AlgoritmoGenetico:
    def __init__(self, archivo, tam_poblacion, generaciones, tasa_mutacion):
        self.archivo = archivo
        self.tam_poblacion = tam_poblacion
        self.generaciones = generaciones
        self.tasa_mutacion = tasa_mutacion

    def ejecutar(self):
        resultado = [] #se guarda la mejor línea generada
        estadisticas_finales = [] #linea para guardar estadísticas por línea

 #recorre línea por línea de ambos archivos
        for index, (linea_base, linea_objetivo) in enumerate(zip(self.archivo.basex, self.archivo.filex)):
  #crea una población inicial aleatoria          
            poblacion = [self.generar_individuo(len(linea_base)) for _ in range(self.tam_poblacion)]
            mejores = [] #guarda la mejor aptitud por generación

        # Ejecuta generaciones del algoritmo genético
            for g in range(self.generaciones):
                aptitudes = [self.evaluar(ind, linea_objetivo) for ind in poblacion] # evalúa individuos
                mejor_ind = max(poblacion, key=lambda ind: self.evaluar(ind, linea_objetivo)) #mejor individuo
                mejor_apt = self.evaluar(mejor_ind, linea_objetivo)
                media = statistics.mean(aptitudes)
                std = statistics.stdev(aptitudes) if len(aptitudes) > 1 else 0
                mejores.append(mejor_apt)

                print(f"Línea {index + 1}, Generación {g + 1} -> Mejor: {mejor_apt}, Media: {media:.2f}, Std: {std:.2f}")

                nueva_poblacion = [mejor_ind] #se usa elitismo 
            
            #se crean nuevos individuos cruzando y mutando     
                while len(nueva_poblacion) < self.tam_poblacion:
                    padre1 = self.seleccionar_torneo(poblacion, aptitudes)
                    padre2 = self.seleccionar_torneo(poblacion, aptitudes)
                    hijo1, hijo2 = self.cruzar_uniforme(padre1, padre2)
                    nueva_poblacion.extend([self.mutar(hijo1), self.mutar(hijo2)])

                poblacion = nueva_poblacion[:self.tam_poblacion]  #se asegura tamaño fijo

            #guarda el mejor individuo final y sus estadísticas
            mejor_ind = max(poblacion, key=lambda ind: self.evaluar(ind, linea_objetivo))
            resultado.append(mejor_ind)

            media_mejores = statistics.mean(mejores)
            std_mejores = statistics.stdev(mejores) if len(mejores) > 1 else 0
            estadisticas_finales.append(
                f"Línea {index + 1}: Mejor: {self.evaluar(mejor_ind, linea_objetivo)}, "
                f"Media Aptitud: {media:.2f}, Std Aptitud: {std:.2f}, "
                f"Media Mejores: {media_mejores:.2f}, Std Mejores: {std_mejores:.2f}"
            )

        self.archivo.guardar_archivo_generado(resultado)
        self.archivo.guardar_estadisticas(estadisticas_finales)

 #funciones auxiliares del algoritmo genético ===
    def generar_individuo(self, longitud):
        return [random.randint(0, 1) for _ in range(longitud)] #genera bits aleatorios

    def evaluar(self, individuo, objetivo):
        return sum(1 for a, b in zip(individuo, objetivo) if a == b) # compara con el objetivo

    def seleccionar_torneo(self, poblacion, aptitudes, k=3):
        seleccionados = random.sample(list(zip(poblacion, aptitudes)), k) 
        return max(seleccionados, key=lambda x: x[1])[0] #selección por torneo

    def cruzar_uniforme(self, padre1, padre2):
        hijo1 = [padre1[i] if random.random() < 0.5 else padre2[i] for i in range(len(padre1))]
        hijo2 = [padre2[i] if random.random() < 0.5 else padre1[i] for i in range(len(padre2))]
        return hijo1, hijo2

    def mutar(self, individuo):
        return [bit if random.random() > self.tasa_mutacion else 1 - bit for bit in individuo]


#Clase de intrefaz grafica 
class InterfazGrafica:
    def __init__(self):
        self.root = Tk()
        self.root.title("Algoritmo Genético - Comparador de Archivos")

        # Variables para guardar rutas de archivos y parámetros
        self.path_base = ""
        self.path_file = ""

        self.var_poblacion = StringVar(value="20")
        self.var_generaciones = StringVar(value="50")
        self.var_mutacion = StringVar(value="0.01")

    #elementos de la interfaz: etiquetas, entradas y botones
        Label(self.root, text="Archivo Base:").grid(row=0, column=0)
        Button(self.root, text="Cargar", command=self.cargar_base).grid(row=0, column=1)

        Label(self.root, text="Archivo Objetivo:").grid(row=1, column=0)
        Button(self.root, text="Cargar", command=self.cargar_objetivo).grid(row=1, column=1)

        Label(self.root, text="Tamaño Población:").grid(row=2, column=0)
        Entry(self.root, textvariable=self.var_poblacion).grid(row=2, column=1)

        Label(self.root, text="Generaciones:").grid(row=3, column=0)
        Entry(self.root, textvariable=self.var_generaciones).grid(row=3, column=1)

        Label(self.root, text="Tasa de Mutación:").grid(row=4, column=0)
        Entry(self.root, textvariable=self.var_mutacion).grid(row=4, column=1)

        Button(self.root, text="Ejecutar", command=self.ejecutar).grid(row=5, column=0, columnspan=2)

        self.root.mainloop()  #ejecuta ventana

    def cargar_base(self):
        self.path_base = filedialog.askopenfilename(title="Seleccionar archivo Basex")

    def cargar_objetivo(self):
        self.path_file = filedialog.askopenfilename(title="Seleccionar archivo Filex")

    def ejecutar(self):
        if not self.path_base or not self.path_file:
            messagebox.showerror("Error", "Debes seleccionar ambos archivos.")
            return

    #crea el objeto archivo y ejecuta el algoritmo genético
        archivo = ArchivoGenetico(self.path_base, self.path_file)
        ag = AlgoritmoGenetico(
            archivo,
            tam_poblacion=int(self.var_poblacion.get()),
            generaciones=int(self.var_generaciones.get()),
            tasa_mutacion=float(self.var_mutacion.get())
        )
        ag.ejecutar()
        messagebox.showinfo("Éxito", "Proceso completado. Revisa los archivos 'Generado.txt' y 'resultado.txt'.")


#ejecutar interfaz
InterfazGrafica()
