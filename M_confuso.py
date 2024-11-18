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

def main_graficos():
    model = load_model(os.path.join(os.path.dirname(__file__), 'modelo_clasificador_lstm.h5'))
    scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'scaler_lstm.pkl'))
    ruta_balanceado = os.path.join(os.path.dirname(__file__), 'data/balanceado')
    ruta_desbalanceado = os.path.join(os.path.dirname(__file__),'data/desbalanceado')
    etiquetas = {
        'bal': 0,
    }
    def cargar_datos(ruta):
        dataframes = []
        for archivo in os.listdir(ruta):
            if archivo.endswith('.csv'):
                df = pd.read_csv(os.path.join(ruta, archivo))
                balanceado = archivo.startswith('datos_bal')
                etiqueta = archivo.replace('datos_', '').replace('.csv', '')
                if balanceado:
                    etiqueta = 'bal'
                    df['estado'] = etiqueta
                else:
                    df['estado'] = etiqueta
                    if etiqueta not in etiquetas:
                        etiquetas[etiqueta] = len(etiquetas)

                # Calcular FFT y agregar la magnitud promedio
                fft_vals = np.fft.fft(df[['x', 'y', 'z']].values, axis=0)
                magnitudes = np.abs(fft_vals)
                df['fft_magnitud'] = magnitudes[:len(magnitudes) // 2].mean()
                dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True)
    datos_balanceado = cargar_datos(ruta_balanceado)
    datos_desbalanceado = cargar_datos(ruta_desbalanceado)
    datos = pd.concat([datos_balanceado, datos_desbalanceado], ignore_index=True)
    datos[['x', 'y', 'z', 'fft_magnitud']] = scaler.transform(datos[['x', 'y', 'z', 'fft_magnitud']])
    datos['magnitud'] = np.sqrt(datos['x'] ** 2 + datos['y'] ** 2 + datos['z'] ** 2)
    X = datos[['x', 'y', 'z', 'fft_magnitud', 'magnitud']]
    y = datos['estado'].apply(lambda x: etiquetas[x])  # Convertir a numérico según etiquetas
    def crear_ventanas(data, labels, time_steps):
        X_windows = []
        y_windows = []
        for i in range(len(data) - time_steps):
            X_windows.append(data[i:i + time_steps])
            y_windows.append(labels[i + time_steps])  # Etiqueta correspondiente a la ventana
        return np.array(X_windows), np.array(y_windows)
    time_steps = 100  # Número de registros a considerar
    X_windows, y_windows = crear_ventanas(X.values, y.values, time_steps)
    X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Obtener las clases predichas
    def plot_confusion_matrix(y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.show()

    def plot_histograms(data):
        plt.figure(figsize=(12, 8))
        data[['x', 'y', 'z', 'fft_magnitud', 'magnitud']].hist(bins=30, figsize=(12, 8), layout=(3, 2))
        plt.suptitle('Histogramas de Características')
        plt.show()

    def plot_scatter(data, etiquetas):
        plt.figure(figsize=(10, 6))
        plt.scatter(data['x'], data['y'], c=data['estado'].apply(lambda x: etiquetas[x]), cmap='viridis', alpha=0.5)
        plt.title('Gráfico de Dispersión entre X e Y')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar(label='Estado')
        plt.show()

    def plot_correlation_matrix(data):
        plt.figure(figsize=(10, 8))
        correlation_matrix = data[['x', 'y', 'z', 'fft_magnitud', 'magnitud']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Matriz de Correlación de Características')
        plt.show()

    def menu_gf():
        while True:
            print("")
            print("=" * 45)
            print(" " * 10 + "👽 Menú de opciones: 👽")
            print("=" * 45)
            print("1. Mostrar Matriz de Confusión 😕")
            print("")
            print("2. Mostrar Histogramas de Características ")
            print("")
            print("3. Mostrar Gráfico de Dispersión entre X e Y ⬆️ ➡️")
            print("")
            print("4. Mostrar Matriz de Correlación de Características")
            print("")
            print("5. Salir")
            print("")
            
            opcion = input("Selecciona una opción (1-5): ")
            
            if opcion == '1':
                # Asume que y_test, y_pred_classes y etiquetas están definidos
                plot_confusion_matrix(y_test, y_pred_classes, list(etiquetas.keys()))
            
            elif opcion == '2':
                # Asume que el DataFrame 'datos' está definido
                plot_histograms(datos)
            
            elif opcion == '3':
                # Asume que el DataFrame 'datos' y 'etiquetas' están definidos
                plot_scatter(datos, etiquetas)
            
            elif opcion == '4':
                # Asume que el DataFrame 'datos' está definido
                plot_correlation_matrix(datos)
            
            elif opcion == '5':
                print("Saliendo del menú.")
                break
            
            else:
                print("Opción no válida. Por favor selecciona una opción del 1 al 5.")

    # Llama a la función del menú
    menu_gf()
