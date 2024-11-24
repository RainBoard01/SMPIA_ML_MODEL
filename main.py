import os
import pandas as pd
from predict import predict
from charts_menu import main_graficos
from test_suite import run_test_suite


#prediction = predict('data/desbalanceado/datos_desbal_1+2.csv')

#print(prediction)
def mostrar_menu():
    print("")
    print("=" * 45)
    print(" " * 10 + "👽 Menú de opciones: 👽")
    print("=" * 45)
    print("1. Seleccionar Archivo Balanceado ➡️ ")
    print("")
    print("2. Seleccionar Archivo Desbalanceado ➡️ ")
    print("")
    print("3. Seleccionar Menu de graficos ")
    print("")
    print("4. Correr set de pruebas al modelo")
    print("")
    print("5. Apagar el sistema. ")
    print("=" * 45)


#def mostrar_menu():
#    print("👽 Menú de opciones: 👽")
#    print("1. Seleccionar Archivo Balanceado ➡️ ")
#    print("2. Seleccionar Archivo Desbalanceado ➡️ ")
#    print("3. Apagar el sistema. ")

def listar_archivo(ruta):
    return [f for f in os.listdir(ruta) if os.path.isfile(os.path.join(ruta, f))]

def main(): 
    rutabalanceado = os.path.join(os.path.dirname(__file__), 'data/old/balanceado')
    rutadesbalanceado = os.path.join(os.path.dirname(__file__),'data/old/desbalanceado')
    maingf = main_graficos
    cp_selected= None

    while True:
        mostrar_menu()
        opcion = input("Selecciona una opción: ")
        
        if opcion == '1':
            cp_selected = rutabalanceado
        elif opcion == '2':
            cp_selected = rutadesbalanceado
        elif opcion == '3':
            maingf()
        elif opcion == '4':
            run_test_suite()
        elif opcion == '5':
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
            prediction = predict(ruta_archivo)
            print(f"Resultado de procesamiento del archivo: {archivo_seleccionado}")  # Simulación de resultado
            print(prediction)
        else:
            print("Selección inválida.")

if __name__ == "__main__":
    main()