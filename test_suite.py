import os
from predict import predict
import pandas as pd

test_files_path = 'test_files'

def run_test_suite():
	results = []

	for root, dirs, files in os.walk(test_files_path):
		for file in files:
				if file.endswith('.csv'):
					archivo = os.path.join(root, file)

					print('Test: ' + str(len(results) + 1) + '/' + str(len(dirs) * len(files)))
					prediction = predict(archivo)
					# get the real class of the file
					if file.startswith('bal'):
						real_class = 'bal'
					else:
						real_class = '_'.join(file.replace('.csv', '').split('_')[:2])
					# save the results
					results.append({
						'file': file,
						'rows': file.replace('.csv', '').split('_')[-2],
						'time': file.replace('.csv', '').split('_')[-1],
						'correct': 1 if prediction['clase_predominante'] == real_class else 0,
						'prediction': prediction['clase_predominante'],
						'real_class': real_class,
						'accuracy': prediction['porcentaje_confianza']
					})

	#save results in a test_results.csv file
	results_df = pd.DataFrame(results)
	results_df.to_csv('reports/test_results.csv', index=False)
	# print the results
	print(results_df)
	# print the accuracy of the model
	print('Accuracy:', results_df['correct'].mean())
