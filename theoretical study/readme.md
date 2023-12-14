In order to replicate the theoretical study performed in this project it is necessary to follow the following steps:
1) Install the libraries listed in `requirements.txt`.
2) Run the script `src/mat2pickle.py`. This script restructures the data in dataset 5 of the NinaPro database.
3) Run the script `src/statistics.py`. This script provides the first statistical analysis of the signals.
4) Run the script `src/emgAnalysis.py`. This script generates a qualitative analysis of the sEMG signals and a discussion of the labeling of the movements.
5) Run the script `src/featuresCalc.py`. This script is responsible for calculating the consider features. There are several parameters that can be adjusted, among them: the width of the window, the increment, the correctness of the labels, the number of subjects, etc.
6) Run the script `src/mlpAnalysis.py`. This is a detailed study of the MLP model with different architectures. It may take a couple of hours.
7) Run the script `src/featuresAndSensorsAnalysis.py`. This script studies the impact of the features and sensors on the classification of gestures on each of the algorithms used (MLP, SVM and LightGBM). It can take more than 10 hours.
8) Run the script `src/plotsFeaturesAndSensorsAnalysis.py`. It performs a plot of the previously calculated data.

Para replicar el estudio teórico realizado en este proyecto es necesario seguir los siguientes pasos:
1) Instalar las librerías listadas en `requirements.txt`
2) Ejecutar el script `src/mat2pickle.py`. Este script reestructura los datos del dataset 5 de la base de datos NinaPro.
3) Ejecutar el script `src/statistics.py`. Este script brinda el primer análisis estadistico de las señales.
4) Ejecutar el script `src/emgAnalysis.py`. En este script se genera un anáñisis cualitativo de las señales de sEMG y una discusión de sobre el etiquetado de los movimientos.
5) Ejecutar el script `src/featuresCalc.py`. Este script se encarga de calcular las distintas features. Existen diversos parámentros que se dueden ajustar, entre ellos: el ancho de la ventana, el incremento, la corrección de las etiquetas, la cantidad de sujetos, etc.
6) Ejecutar el script `src/mlpAnalysis.py`. En este se hace un estudio a detalle del modelo MLP con distintas arquitecturas. Puede tardar un par de horas.
7) Ejecutar el script `src/featuresAndSensorsAnalysis.py`. En este script se estudia el impacto de las features y los sensores sobre la clasificación de los gestos sobre cada uno de los algoritmos utilziados (MLP, SVM y LightGBM). Puede tardar más de 10 horas.
8) Ejecutar el script `src/plotsFeaturesAndSensorsAnalysis.py`. Este realiza un plor de los datos calculados anteriormente.

Copyright (C) 2023  Renato Sosa Machado Scheeffer. Universidad de la República, Uruguay.
