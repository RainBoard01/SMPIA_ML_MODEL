import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from keras.models import load_model

import joblib

# Cargar el modelo y el escalador guardados
modelo_path = 'models/optimized_m4_200.h5'
model = load_model(os.path.join(os.path.dirname(__file__), modelo_path))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'models/scaler.pkl'))

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
                etiqueta = '_'.join(archivo.replace('datos_', '').replace('.csv', '').split('_')[:2])
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

def predict(ruta_archivo):
    # Cargar y preprocesar los datos
    datos_archivo = cargar_datos_nuevo(ruta_archivo)

    # Preprocesar los nuevos datos
    datos_archivo = preprocesar_datos(datos_archivo, scaler)

    # Definir el tamaño de la ventana
    time_steps = 200  # Por ejemplo, 10 ms

    # Crear ventanas de tiempo
    X_nuevos = crear_ventanas(datos_archivo[['x', 'y', 'z', 'fft_magnitud', 'magnitud']].values, time_steps)

    # Realizar predicciones
    print("Ejecutando modelo: " + modelo_path + " en archivo: " + ruta_archivo + "...")
    predicciones = model.predict(X_nuevos)

    # Convertir las predicciones a clases
    clases_predichas = np.argmax(predicciones, axis=1)
    # print("")
    # print("=" * 30)
    # print('Archivo:', archivo_nuevo)
    clase_predominante = np.bincount(clases_predichas).argmax()
    # print("")
    # print('Clase predominante: ',list(etiquetas.keys())[list(etiquetas.values()).index(clase_predominante)])
    # print("")
    # print('Porcentaje de confianza: ',np.max(predicciones) * 100)
    # print("")
    # print('Clases detectadas:')
    set_clases = set(clases_predichas)
    # for clase in set_clases:
    #     print(list(etiquetas.keys())[list(etiquetas.values()).index(clase)], str(np.count_nonzero(clases_predichas == clase) / len(clases_predichas) * 100)[:5] + '%')
    # print("=" * 30)
    # print("")

    # Definir la norma esperada
    norma_esperada = 1

    # Calcular el porcentaje de desbalanceo
    datos_archivo['porcentaje_desbalanceo'] = (datos_archivo['magnitud'] / norma_esperada) * 100

    # Ajustar el rango de porcentajes
    porcentaje_min_original = 20
    porcentaje_max_original = 400

    # Escalar los porcentajes
    datos_archivo['porcentaje_escalado'] = np.where(
        datos_archivo['porcentaje_desbalanceo'] < porcentaje_min_original, 
        0,
        np.clip((datos_archivo['porcentaje_desbalanceo'] - porcentaje_min_original) / 
                (porcentaje_max_original - porcentaje_min_original) * 100, 0, 100)
    )

    # Calcular mínimo, promedio, máximo y delta en el nuevo rango
    porcentaje_min = datos_archivo['porcentaje_escalado'].min()
    porcentaje_promedio = datos_archivo['porcentaje_escalado'].mean()
    porcentaje_max = datos_archivo['porcentaje_escalado'].max()
    delta = porcentaje_max - porcentaje_min

    return {
        'archivo': ruta_archivo,
        'clase_predominante': list(etiquetas.keys())[list(etiquetas.values()).index(clase_predominante)],
        'porcentaje_confianza': float(np.max(predicciones) * 100),
        'clases_detectadas': [{ 'clase': list(etiquetas.keys())[list(etiquetas.values()).index(clase)], 'porcentaje': float(np.count_nonzero(clases_predichas == clase) / len(clases_predichas) * 100)} for clase in set_clases],
        'modelo': modelo_path,
        'magnitudes': {
            'min': float(porcentaje_min),
            'promedio': float(porcentaje_promedio),
            'max': float(porcentaje_max),
            'delta': float(delta)
        }
    }


