import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.layers import Dense, LSTM
from keras.utils import to_categorical
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib


def graficar_onda(df, etiqueta):
    # Graficar la onda de aceleración en el eje X.. cada fila son 0.5ms
    df['Tiempo (ms)'] = df.index * 0.5
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    
    # Onda completa
    ax[0].plot(df['Tiempo (ms)'], df['x'], label='x')
    ax[0].set_ylabel('Aceleración (g)')
    ax[0].set_xlabel('Tiempo (ms)')
    ax[0].legend()
    ax[0].set_title(f'Onda de aceleración completa ({etiqueta})')
    
    # Onda de los primeros 1000ms
    ax[1].plot(df['Tiempo (ms)'], df['x'], label='x')
    ax[1].set_ylabel('Aceleración (g)')
    ax[1].set_xlabel('Tiempo (ms)')
    ax[1].legend()
    ax[1].set_title(f'Onda de aceleración primeros 1000ms ({etiqueta})')
    ax[1].set_xlim(0, 1000)
    
    fig.suptitle(f'Onda de aceleración en el eje X ({etiqueta})')
    fig.tight_layout()
    
    # Guardar la imagen
    fig.savefig(f'charts/onda_aceleracion_{etiqueta}.png')


ruta_balanceado = os.path.join(os.path.dirname(__file__), 'data/old/balanceado')
ruta_desbalanceado = os.path.join(os.path.dirname(__file__),'data/old/desbalanceado')
etiquetas = {
	'bal': 0,
   }

def listar_archivo(ruta):
	 return [f for f in os.listdir(ruta) if os.path.isfile(os.path.join(ruta, f))]
def main_graficos_por_archivo():
	cp_selected= None
	while True:
		print("")
		print("=" * 45)
		print(" " * 10 + "👽 Menú de opciones: 👽")
		print("=" * 45)
		print("1. Seleccionar Archivo Balanceado ➡️ ")
		print("")
		print("2. Seleccionar Archivo Desbalanceado ➡️ ")
		print("")
		print('3. Graficar todos los archivos')
		print("")
		print("4. Apagar el sistema. ")
		print("=" * 45)
		opcion = input("Selecciona una opción: ")
		if opcion == '1':
			cp_selected = ruta_balanceado
		elif opcion == '2':
			cp_selected = ruta_desbalanceado
		elif opcion == '3':
			for archivo in listar_archivo(ruta_balanceado):
				df = pd.read_csv(os.path.join(ruta_balanceado, archivo))
				etiqueta = archivo.replace('datos_', '').replace('.csv', '')
				graficar_onda(df, etiqueta)
			for archivo in listar_archivo(ruta_desbalanceado):
				df = pd.read_csv(os.path.join(ruta_desbalanceado, archivo))
				etiqueta = archivo.replace('datos_', '').replace('.csv', '')
				graficar_onda(df, etiqueta)
		elif opcion == '4':
			print("Saliendo del programa.")
			break
		else:
			print("Opción no válida. Intenta de nuevo.")
			continue
		archivos = listar_archivo(cp_selected)
		if not archivos:
			print("No hay archivos en la carpeta seleccionada.")
			continue
		print("Archivos disponibles:")
		for i, archivo in enumerate(archivos, start=1):
			print(f"{i}. {archivo}")
		seleccion = int(input("Selecciona el número del archivo que deseas usar: ")) - 1
		if 0 <= seleccion < len(archivos):
			archivo_seleccionado = archivos[seleccion]
			ruta_archivo= os.path.join(cp_selected,archivo_seleccionado)
			# Aquí puedes cargar y procesar el archivo como desees
			df = pd.read_csv(ruta_archivo)
			etiqueta = archivo_seleccionado.replace('datos_', '').replace('.csv', '')
			graficar_onda(df, etiqueta)
		else:
			print("Selección no válida. Intenta de nuevo.")
			continue