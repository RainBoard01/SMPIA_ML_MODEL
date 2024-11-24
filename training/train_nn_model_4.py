import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, BatchNormalization, Dropout, Dense, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt

# Configurar rutas de los datasets
ruta_balanceado = os.path.join(os.path.dirname(__file__), '../data/balanceado')
ruta_desbalanceado = os.path.join(os.path.dirname(__file__),'../data/desbalanceado')

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
                etiqueta = '_'.join(etiqueta.split('_')[:2])
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

time_steps = 200  # Número de registros a considerar
X_windows, y_windows = crear_ventanas(X.values, y.values, time_steps)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=0.2, random_state=42)

# One-hot encode the labels for training
y_train = to_categorical(y_train, num_classes=len(etiquetas))
y_test = to_categorical(y_test, num_classes=len(etiquetas))

model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(100, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(100)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(len(etiquetas), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# Entrenamiento
history = model.fit(X_train, y_train, epochs=50, batch_size=25, 
                    validation_split=0.2, callbacks=[early_stopping, reduce_lr])

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Guardar el modelo y el escalador
model.save('optimized_m4_200.h5')
joblib.dump(scaler, 'scaler.pkl')

# Check model summary
print(model.summary())

plt.figure(figsize=(12, 4))

# Pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Precisión
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.show()