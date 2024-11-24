import numpy as np
import pandas as pd
import os


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
                etiqueta = '_'.join(etiqueta.split('_')[:2])
                df['estado'] = etiqueta
                if etiqueta not in etiquetas:
                    etiquetas[etiqueta] = len(etiquetas)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# Cargar datos balanceados y desbalanceados
datos_balanceado = cargar_datos(ruta_balanceado)
datos_desbalanceado = cargar_datos(ruta_desbalanceado)

datos = pd.concat([datos_balanceado, datos_desbalanceado], ignore_index=True)

#for every etiqueta create a new dataframe and print 5 first rows
for etiqueta in etiquetas:
	etiqueta_df = datos[datos['estado'] == etiqueta]
	
	# make a folder named test_files if it doesn't exist
	if not os.path.exists('test_files'):
		os.makedirs('test_files')
          
	#inside test_files create a folder with the name of the etiqueta if it doesn't exist
	if not os.path.exists('test_files/'+etiqueta):
		os.makedirs('test_files/'+etiqueta)
   
	#sizes of the files to be created
	sizes = [250, 500, 1000, 2000, 4000, 10000, 20000]
   
	for size in sizes:
		#take a random number to start the cut
		if (size > len(etiqueta_df)):
			continue

		if size == len(etiqueta_df):
			start = 0
		else:
			start = np.random.randint(0, len(etiqueta_df)-size)
		#take the rows from the start to the size
		new_df = etiqueta_df.iloc[start:start+size]
      #remove estado column
		new_df = new_df.drop(columns='estado')
		#save the new dataframe to a csv file
		if size >= 2000:
			new_df.to_csv('test_files/'+etiqueta+'/'+etiqueta+'_'+str(size)+'_'+str(size/2000)[:-2]+'s.csv', index=False)
			print('test_files/'+etiqueta+'/'+etiqueta+'_'+str(size)+'_'+str(size/2000)[:-2]+'s.csv')
		else:
			new_df.to_csv('test_files/'+etiqueta+'/'+etiqueta+'_'+str(size)+'_'+str(size/2)[:-2]+'ms.csv', index=False)
			print('test_files/'+etiqueta+'/'+etiqueta+'_'+str(size)+'_'+str(size/2)[:-2]+'ms.csv')
		
	print('-'*50)
	print('Created ' + str(len(sizes)) + ' files for etiqueta ' + etiqueta)
	print('-'*50)