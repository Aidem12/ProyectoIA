import random  # Generar valores aleatorios
import statistics  # Para calcular media y desviación estándar


# Clase para manejar archivos de entrada y salida
class ArchivoGenetico:
    def __init__(self, path_base="Base.txt", path_objetivo="File.txt"):
        self.basex = self.cargar(path_base)  # carga archivo base
        self.filex = self.cargar(path_objetivo)  # cargar archivo file

    def cargar(self, path):
        # Lee el archivo y lo convierte en una lista de listas de enteros
        with open(path, "r") as f:
            return [list(map(int, line.strip().split(","))) for line in f.readlines()]

    def guardar_archivo_generado(self, contenido, nombre_archivo="Generado.txt"):
        # Guarda el resultado final en un archivo
        with open(nombre_archivo, "w") as f:
            for fila in contenido:
                f.write(",".join(map(str, fila)) + "\n")

    def guardar_estadisticas(self, estadisticas, nombre_archivo="resultado.txt"):
        # Guarda las estadísticas del algoritmo genético en un archivo
        with open(nombre_archivo, "w") as f:
            for linea_stats in estadisticas:
                f.write(f"{linea_stats}\n")


# Clase que implementa el algoritmo genético
class AlgoritmoGenetico:
    def __init__(self, archivo, tam_poblacion, generaciones, tasa_mutacion):
        self.archivo = archivo
        self.tam_poblacion = tam_poblacion
        self.generaciones = generaciones
        self.tasa_mutacion = tasa_mutacion

    def ejecutar(self):
        resultado = []  # se guarda la mejor línea generada
        estadisticas_finales = []  # linea para guardar estadísticas por línea

        # Recorre línea por línea de ambos archivos
        for index, (linea_base, linea_objetivo) in enumerate(zip(self.archivo.basex, self.archivo.filex)):
            # Crea una población inicial aleatoria
            poblacion = [self.generar_individuo(len(linea_base)) for _ in range(self.tam_poblacion)]
            mejores = []  # guarda la mejor aptitud por generación

            # Ejecuta generaciones del algoritmo genético
            for g in range(self.generaciones):
                aptitudes = [self.evaluar(ind, linea_objetivo) for ind in poblacion]  # evalúa individuos
                mejor_ind = max(poblacion, key=lambda ind: self.evaluar(ind, linea_objetivo))  # mejor individuo
                mejor_apt = self.evaluar(mejor_ind, linea_objetivo)
                media = statistics.mean(aptitudes)
                std = statistics.stdev(aptitudes) if len(aptitudes) > 1 else 0
                mejores.append(mejor_apt)

                print(f"Línea {index + 1}, Generación {g + 1} -> Mejor: {mejor_apt}, Media: {media:.2f}, Std: {std:.2f}")

                nueva_poblacion = [mejor_ind]  # se usa elitismo

                # Se crean nuevos individuos cruzando y mutando
                while len(nueva_poblacion) < self.tam_poblacion:
                    padre1 = self.seleccionar_torneo(poblacion, aptitudes)
                    padre2 = self.seleccionar_torneo(poblacion, aptitudes)
                    hijo1, hijo2 = self.cruzar_uniforme(padre1, padre2)
                    nueva_poblacion.extend([self.mutar(hijo1), self.mutar(hijo2)])

                poblacion = nueva_poblacion[:self.tam_poblacion]  # se asegura tamaño fijo

            # Guarda el mejor individuo final y sus estadísticas
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
        print("Proceso completado. Se han generado los archivos 'Generado.txt' y 'resultado.txt'.")

    # Funciones auxiliares del algoritmo genético
    def generar_individuo(self, longitud):
        return [random.randint(0, 1) for _ in range(longitud)]  # genera bits aleatorios

    def evaluar(self, individuo, objetivo):
        return sum(1 for a, b in zip(individuo, objetivo) if a == b)  # compara con el objetivo

    def seleccionar_torneo(self, poblacion, aptitudes, k=3):
        seleccionados = random.sample(list(zip(poblacion, aptitudes)), k)
        return max(seleccionados, key=lambda x: x[1])[0]  # selección por torneo

    def cruzar_uniforme(self, padre1, padre2):
        hijo1 = [padre1[i] if random.random() < 0.5 else padre2[i] for i in range(len(padre1))]
        hijo2 = [padre2[i] if random.random() < 0.5 else padre1[i] for i in range(len(padre2))]
        return hijo1, hijo2

    def mutar(self, individuo):
        return [bit if random.random() > self.tasa_mutacion else 1 - bit for bit in individuo]


def main():
    print("=== Algoritmo Genético - Comparador de Archivos ===")
    
    # Solicitar parámetros por consola
    try:
        tam_poblacion = int(input("Ingrese el tamaño de la población: "))
        generaciones = int(input("Ingrese el número de generaciones: "))
        tasa_mutacion = float(input("Ingrese la tasa de mutación (0-1): "))
        
        # Validaciones básicas
        if tam_poblacion <= 0 or generaciones <= 0 or tasa_mutacion < 0 or tasa_mutacion > 1:
            raise ValueError("Los valores ingresados no son válidos.")
            
    except ValueError as e:
        print(f"Error: {e}")
        print("Se usarán valores predeterminados: población=20, generaciones=50, mutación=0.01")
        tam_poblacion = 20
        generaciones = 50
        tasa_mutacion = 0.01

    # Crear instancia de archivo y algoritmo genético
    try:
        print("Cargando archivos Base.txt y File.txt...")
        archivo = ArchivoGenetico()
        
        print(f"Ejecutando algoritmo con: población={tam_poblacion}, "
              f"generaciones={generaciones}, mutación={tasa_mutacion}")
        
        ag = AlgoritmoGenetico(
            archivo,
            tam_poblacion=tam_poblacion,
            generaciones=generaciones,
            tasa_mutacion=tasa_mutacion
        )
        ag.ejecutar()
        
    except FileNotFoundError:
        print("Error: No se encontraron los archivos Base.txt y/o File.txt en el directorio actual.")
    except Exception as e:
        print(f"Error inesperado: {e}")


if __name__ == "__main__":
    main()
