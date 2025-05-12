import random
import statistics
from tkinter import Tk, Label, Button, filedialog, Entry, StringVar, messagebox


class ArchivoGenetico:
    def __init__(self, path_base, path_objetivo):
        self.basex = self.cargar(path_base)
        self.filex = self.cargar(path_objetivo)
        self.validar_tamanos()

    def cargar(self, path):
        with open(path, "r") as f:
            return [list(map(int, line.strip().split(","))) for line in f.readlines()]

    def validar_tamanos(self):
        if len(self.basex) != len(self.filex):
            raise ValueError("Los archivos no tienen el mismo número de líneas.")
        for i, (l1, l2) in enumerate(zip(self.basex, self.filex)):
            if len(l1) != len(l2):
                raise ValueError(f"Las líneas {i+1} no tienen la misma longitud entre Basex y Filex.")

    def guardar_archivo_generado(self, contenido, nombre_archivo="Generado.txt"):
        with open(nombre_archivo, "w") as f:
            for fila in contenido:
                f.write(",".join(map(str, fila)) + "\n")

    def guardar_estadisticas(self, estadisticas, nombre_archivo="resultado.txt"):
        with open(nombre_archivo, "w") as f:
            for linea_stats in estadisticas:
                f.write(f"{linea_stats}\n")


class AlgoritmoGenetico:
    def __init__(self, archivo, tam_poblacion, generaciones, tasa_mutacion):
        self.archivo = archivo
        self.tam_poblacion = tam_poblacion
        self.generaciones = generaciones
        self.tasa_mutacion = tasa_mutacion

    def ejecutar(self):
        resultado = []
        estadisticas_finales = []

        for index, (linea_base, linea_objetivo) in enumerate(zip(self.archivo.basex, self.archivo.filex)):
            poblacion = [self.generar_individuo(len(linea_base)) for _ in range(self.tam_poblacion)]
            mejores = []

            for g in range(self.generaciones):
                aptitudes = [self.evaluar(ind, linea_objetivo) for ind in poblacion]
                mejor_ind = max(poblacion, key=lambda ind: self.evaluar(ind, linea_objetivo))
                mejor = self.evaluar(mejor_ind, linea_objetivo)
                media = statistics.mean(aptitudes)
                std = statistics.stdev(aptitudes) if len(aptitudes) > 1 else 0
                mejores.append(mejor)

                print(f"L\u00ednea {index + 1}, Generaci\u00f3n {g + 1} -> Mejor: {mejor}, Media: {media:.2f}, Std: {std:.2f}")

                nueva_poblacion = [mejor_ind]  # Elitismo: conservar mejor individuo
                while len(nueva_poblacion) < self.tam_poblacion:
                    padre1 = self.seleccionar(poblacion, aptitudes)
                    padre2 = self.seleccionar(poblacion, aptitudes)
                    hijo1, hijo2 = self.cruzar(padre1, padre2)
                    nueva_poblacion.extend([self.mutar(hijo1), self.mutar(hijo2)])

                poblacion = nueva_poblacion[:self.tam_poblacion]

            mejor_final = max(poblacion, key=lambda ind: self.evaluar(ind, linea_objetivo))
            resultado.append(mejor_final)

            media_mejores = statistics.mean(mejores)
            std_mejores = statistics.stdev(mejores) if len(mejores) > 1 else 0
            estadisticas_finales.append(
                f"L\u00ednea {index + 1}: Mejor: {self.evaluar(mejor_final, linea_objetivo)}, "
                f"Media Aptitud: {media:.2f}, Std Aptitud: {std:.2f}, "
                f"Media Mejores: {media_mejores:.2f}, Std Mejores: {std_mejores:.2f}"
            )

        self.archivo.guardar_archivo_generado(resultado)
        self.archivo.guardar_estadisticas(estadisticas_finales)

    def generar_individuo(self, longitud):
        return [random.randint(0, 1) for _ in range(longitud)]

    def evaluar(self, individuo, objetivo):
        return sum(1 for a, b in zip(individuo, objetivo) if a == b)

    def seleccionar(self, poblacion, aptitudes):
        total = sum(aptitudes)
        if total == 0:
            return random.choice(poblacion)
        r = random.uniform(0, total)
        acc = 0
        for ind, apt in zip(poblacion, aptitudes):
            acc += apt
            if acc >= r:
                return ind

    def cruzar(self, padre1, padre2):
        punto = random.randint(1, len(padre1) - 2)
        return padre1[:punto] + padre2[punto:], padre2[:punto] + padre1[punto:]

    def mutar(self, individuo):
        return [bit if random.random() > self.tasa_mutacion else 1 - bit for bit in individuo]


class InterfazGrafica:
    def __init__(self):
        self.root = Tk()
        self.root.title("Algoritmo Gen\u00e9tico - Comparador de Archivos")

        self.path_base = ""
        self.path_file = ""

        self.var_poblacion = StringVar(value="20")
        self.var_generaciones = StringVar(value="50")
        self.var_mutacion = StringVar(value="0.01")

        self.configurar_gui()
        self.root.mainloop()

    def configurar_gui(self):
        Label(self.root, text="Archivo Base:").grid(row=0, column=0)
        Button(self.root, text="Cargar", command=self.cargar_base).grid(row=0, column=1)

        Label(self.root, text="Archivo Objetivo:").grid(row=1, column=0)
        Button(self.root, text="Cargar", command=self.cargar_objetivo).grid(row=1, column=1)

        Label(self.root, text="Tama\u00f1o Poblaci\u00f3n:").grid(row=2, column=0)
        Entry(self.root, textvariable=self.var_poblacion).grid(row=2, column=1)

        Label(self.root, text="Generaciones:").grid(row=3, column=0)
        Entry(self.root, textvariable=self.var_generaciones).grid(row=3, column=1)

        Label(self.root, text="Tasa de Mutaci\u00f3n:").grid(row=4, column=0)
        Entry(self.root, textvariable=self.var_mutacion).grid(row=4, column=1)

        Button(self.root, text="Ejecutar", command=self.ejecutar).grid(row=5, column=0, columnspan=2)

    def cargar_base(self):
        self.path_base = filedialog.askopenfilename(title="Seleccionar archivo Basex")

    def cargar_objetivo(self):
        self.path_file = filedialog.askopenfilename(title="Seleccionar archivo Filex")

    def ejecutar(self):
        if not self.path_base or not self.path_file:
            messagebox.showerror("Error", "Debes seleccionar ambos archivos.")
            return

        try:
            archivo = ArchivoGenetico(self.path_base, self.path_file)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        ag = AlgoritmoGenetico(
            archivo,
            tam_poblacion=int(self.var_poblacion.get()),
            generaciones=int(self.var_generaciones.get()),
            tasa_mutacion=float(self.var_mutacion.get())
        )
        ag.ejecutar()
        messagebox.showinfo("\u00c9xito", "Proceso completado. Revisa los archivos 'Generado.txt' y 'resultado.txt'.")


# Ejecutar interfaz
InterfazGrafica()
