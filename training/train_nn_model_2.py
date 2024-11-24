import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import joblib

# Configurar rutas de los datasets
ruta_balanceado = os.path.join(os.path.dirname(__file__), 'data/balanceado')
ruta_desbalanceado = os.path.join(os.path.dirname(__file__),'data/desbalanceado')

# Mapeo de etiquetas
etiquetas = {
    'bal': 0,
}

# Cargar y etiquetar datos
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

# Cargar datos balanceados y desbalanceados
datos_balanceado = cargar_datos(ruta_balanceado)
datos_desbalanceado = cargar_datos(ruta_desbalanceado)
datos = pd.concat([datos_balanceado, datos_desbalanceado], ignore_index=True)


# Normalizar columnas de características y agregar características adicionales
scaler = StandardScaler()
datos[['x', 'y', 'z', 'fft_magnitud']] = scaler.fit_transform(datos[['x', 'y', 'z', 'fft_magnitud']])
datos['magnitud'] = np.sqrt(datos['x'] ** 2 + datos['y'] ** 2 + datos['z'] ** 2)

# Dividir datos en características (X) y etiquetas
X = datos[['x', 'y', 'z', 'fft_magnitud', 'magnitud']]
y = datos['estado'].apply(lambda x: etiquetas[x])  # Convertir a numérico según etiquetas

# Crear ventanas para series de tiempo
def crear_ventanas(data, labels, time_steps):
    X_windows = []
    y_windows = []
    for i in range(len(data) - time_steps):
        X_windows.append(data[i:i + time_steps])
        y_windows.append(labels[i + time_steps])  # Etiqueta correspondiente a la ventana
    return np.array(X_windows), np.array(y_windows)

time_steps = 10  # Número de registros a considerar
X_windows, y_windows = crear_ventanas(X.values, y.values, time_steps)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=0.2, random_state=42)

# One-hot encode the labels for training
y_train = to_categorical(y_train, num_classes=len(etiquetas))
y_test = to_categorical(y_test, num_classes=len(etiquetas))

# Construir y entrenar la red neuronal
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50, activation='relu', return_sequences=True),
    Dropout(0.2),  # Regularización
    LSTM(50, activation='relu'),
    Dropout(0.2),  # Regularización
    Dense(len(etiquetas), activation='softmax')  # Clasificación multiclase
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {str(accuracy * 100)[:5]}%')

# Guardar el modelo y el escalador
model.save('balanced_m2_100.h5')
joblib.dump(scaler, 'scaler_lstm.pkl')

# Check model summary
print(model.summary())

# Check model input shape
print(X_train.shape)
print(y_train.shape)

# Check model output shape
print(X_test.shape)
print(y_test.shape)

# Check model prediction
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
print(y_pred_classes)
print(y_test_classes)
print(accuracy_score(y_test_classes, y_pred_classes))