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
from viz import main_graficos_por_archivo
import joblib

def calcular_fft(df):
    # Calcular la FFT de las columnas 'x', 'y', y 'z'
    fft_x = np.fft.fft(df['x'])
    fft_y = np.fft.fft(df['y'])
    fft_z = np.fft.fft(df['z'])

    # Calcular la magnitud de la FFT
    magnitude_fft_x = np.abs(fft_x)
    magnitude_fft_y = np.abs(fft_y)
    magnitude_fft_z = np.abs(fft_z)

    # Extraer características (promedio y desviación estándar de las magnitudes)
    df['fft_mean_x'] = np.mean(magnitude_fft_x)
    df['fft_std_x'] = np.std(magnitude_fft_x)
    df['fft_mean_y'] = np.mean(magnitude_fft_y)
    df['fft_std_y'] = np.std(magnitude_fft_y)
    df['fft_mean_z'] = np.mean(magnitude_fft_z)
    df['fft_std_z'] = np.std(magnitude_fft_z)
    return df

def main_graficos():
    model = load_model(os.path.join(os.path.dirname(__file__), 'models/optimized_m4_200.h5'))
    scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'models/scaler.pkl'))
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
                    etiqueta = '_'.join(etiqueta.split('_')[:2])
                    df['estado'] = etiqueta
                    if etiqueta not in etiquetas:
                        etiquetas[etiqueta] = len(etiquetas)

                calcular_fft(df)
                dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True)
    datos_balanceado = cargar_datos(ruta_balanceado)
    datos_desbalanceado = cargar_datos(ruta_desbalanceado)
    datos = pd.concat([datos_balanceado, datos_desbalanceado], ignore_index=True)
    datos[['x', 'y', 'z', 'fft_mean_x', 'fft_std_x', 'fft_mean_y', 'fft_std_y', 'fft_mean_z', 'fft_std_z']] = scaler.transform(datos[['x', 'y', 'z', 'fft_mean_x', 'fft_std_x', 'fft_mean_y', 'fft_std_y', 'fft_mean_z', 'fft_std_z']])
    datos['magnitud'] = np.sqrt(datos['x'] ** 2 + datos['y'] ** 2 + datos['z'] ** 2)
    X = datos[['x', 'y', 'z', 'fft_mean_x', 'fft_std_x', 'fft_mean_y', 'fft_std_y', 'fft_mean_z', 'fft_std_z', 'magnitud']]
    y = datos['estado'].apply(lambda x: etiquetas[x])  # Convertir a numérico según etiquetas
    def crear_ventanas(data, labels, time_steps):
        X_windows = []
        y_windows = []
        for i in range(len(data) - time_steps):
            X_windows.append(data[i:i + time_steps])
            y_windows.append(labels[i + time_steps])  # Etiqueta correspondiente a la ventana
        return np.array(X_windows), np.array(y_windows)
    time_steps = 200  # Número de registros a considerar
    X_windows, y_windows = crear_ventanas(X.values, y.values, time_steps)

    print('Preparando datos para graficar...')
    X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Obtener las clases predichas
    
    def plot_confusion_matrix(y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 7))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.xticks(rotation=45)
        plt.ylabel('Real')
        plt.show()

    def plot_histograms(data):
        plt.close('all')
        features = ['x', 'y', 'z', 'fft_mean_x', 'fft_std_x', 'fft_mean_y', 'fft_std_y', 'fft_mean_z', 'fft_std_z', 'magnitud']
        num_features = len(features)
        num_rows = (num_features // 2) + (num_features % 2)
        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(12, 8))
        axes = axes.flatten()
        for i, feature in enumerate(features):
            axes[i].hist(data[feature], bins=30, alpha=0.7)
            axes[i].set_title(feature)
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frecuencia')
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])  # 
        plt.suptitle('Histogramas de Características')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajustar el título
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
        correlation_matrix = data[['x', 'y', 'z', 'fft_mean_x', 'fft_std_x', 'fft_mean_y', 'fft_std_y', 'fft_mean_z', 'fft_std_z', 'magnitud']].corr()
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
            print("5. Graficos por archivo")
            print("")
            print("6. Salir")
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
                main_graficos_por_archivo()
            elif opcion == '6':
                print("Saliendo del menú.")
                break
            
            else:
                print("Opción no válida. Por favor selecciona una opción del 1 al 5.")

    # Llama a la función del menú
    menu_gf()
