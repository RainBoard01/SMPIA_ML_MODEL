import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from keras.models import load_model

import joblib

# Cargar el modelo y el escalador guardados
modelo_path = 'modelos/balanced_m3_100.h5'
model = load_model(os.path.join(os.path.dirname(__file__), modelo_path))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'modelos/scaler.pkl'))

# Configurar rutas de los datasets
ruta_balanceado = os.path.join(os.path.dirname(__file__), 'data/balanceado')
ruta_desbalanceado = os.path.join(os.path.dirname(__file__),'data/desbalanceado')

#mapeo de etiquetas
etiquetas = {
    'bal': 0
}
def cargar_etiquetas(ruta):
    for archivo in os.listdir(ruta):
        if archivo.endswith('.csv'):
            balanceado = archivo.startswith('datos_bal')
            if balanceado:
                return
            else:
                etiqueta = archivo.replace('datos_', '').replace('.csv', '')
                if etiqueta not in etiquetas:
                    etiquetas[etiqueta] = len(etiquetas)
                

cargar_etiquetas(ruta_balanceado)
cargar_etiquetas(ruta_desbalanceado)

# Cargar los datos de un nuevo archivo CSV
def cargar_datos_nuevo(archivo):
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), archivo))
    fft_vals = np.fft.fft(df[['x', 'y', 'z']].values, axis=0)
    magnitudes = np.abs(fft_vals)
    df['fft_magnitud'] = magnitudes[:len(magnitudes) // 2].mean()
    return df

# Preprocesar los datos
def preprocesar_datos(df, scaler):
    # Normalizar columnas de características
    df[['x', 'y', 'z', 'fft_magnitud']] = scaler.transform(df[['x', 'y', 'z', 'fft_magnitud']])
    df['magnitud'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2)
    return df

# Crear ventanas de tiempo
def crear_ventanas(data, time_steps):
    X_windows = []
    for i in range(len(data) - time_steps):
        X_windows.append(data[i:i + time_steps])
    return np.array(X_windows)

def predict(archivo_nuevo):
    # Cargar y preprocesar los datos
    datos_nuevos = cargar_datos_nuevo(archivo_nuevo)

    # Normalizador: asegúrate de haber guardado el scaler después del entrenamiento
    # Preprocesar los nuevos datos
    datos_nuevos = preprocesar_datos(datos_nuevos, scaler)

    # # Asegúrate de tener todos los datos como un único conjunto
    # X_nuevo = datos_nuevos[['x', 'y', 'z', 'fft_magnitud']].values

    # # Redimensionar para que sea (1, número de timesteps, número de características)
    # timesteps = X_nuevo.shape[0]
    # X_nuevo = X_nuevo.reshape(1, timesteps, X_nuevo.shape[1])  # (1, timesteps, 4)

    # Definir el tamaño de la ventana
    time_steps = 100  # Por ejemplo, 10 ms

    # Crear ventanas de tiempo
    X_nuevos = crear_ventanas(datos_nuevos[['x', 'y', 'z', 'fft_magnitud', 'magnitud']].values, time_steps)

    # Realizar predicciones
    print("Ejecutando modelo: ", modelo_path)
    predicciones = model.predict(X_nuevos)

    # Convertir las predicciones a clases
    clases_predichas = np.argmax(predicciones, axis=1)
    print("")
    print("=" * 30)
    print('Archivo:', archivo_nuevo)
    clase_predominante = np.bincount(clases_predichas).argmax()
    print("")
    print('Clase predominante: ',list(etiquetas.keys())[list(etiquetas.values()).index(clase_predominante)])
    print("")
    print('Porcentaje de confianza: ',np.max(predicciones) * 100)
    print("")
    print('Clases detectadas:')
    set_clases = set(clases_predichas)
    for clase in set_clases:
        print(list(etiquetas.keys())[list(etiquetas.values()).index(clase)], str(np.count_nonzero(clases_predichas == clase) / len(clases_predichas) * 100)[:5] + '%')
    print("=" * 30)
    print("")

    return {
        'archivo': archivo_nuevo,
        'clase_predominante': list(etiquetas.keys())[list(etiquetas.values()).index(clase_predominante)],
        'porcentaje_confianza': float(np.max(predicciones) * 100),
        'clases_detectadas': [{ 'clase': list(etiquetas.keys())[list(etiquetas.values()).index(clase)], 'porcentaje': float(np.count_nonzero(clases_predichas == clase) / len(clases_predichas) * 100)} for clase in set_clases]
    }


