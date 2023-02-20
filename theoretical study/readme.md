Para replicar el estudio teórico realizado en este proyecto es necesario seguir los siguientes pasos:
1) Instalar las librerías listadas en requirements.txt
2) Ejecutar el script mat2pickle.py. Este script reestructura los datos del dataset 5 de la base de datos NinaPro.
3) Ejecutar el script statistics.py. Este script brinda el primer análisis estadistico de las señales.
4) Ejecutar el script emgAnalysis.py. En este script se genera un anáñisis cualitativo de las señales de sEMG y una discusión de sobre el etiquetado de los movimientos.
5) Ejecutar el script featuresCalc.py. Este script se encarga de calcular las distintas features. Existen diversos parámentros que se dueden ajustar, entre ellos: el ancho de la ventana, el incremento, la corrección de las etiquetas, la cantidad de sujetos, etc.
6) Ejecutar el script mlpAnalysis.py. En este se hace un estudio a detalle del modelo MLP con distintas arquitecturas. Puede tardar un par de horas.
7) Ejecutar el script featuresAndSensorsAnalysis.py. En este script se estudia el impacto de las features y los sensores sobre la clasificación de los gestos sobre cada uno de los algoritmos utilziados (MLP, SVM y LightGBM). Puede tardar más de 10 horas.
8) Ejecutar el script plotsFeaturesAndSensorsAnalysis.py. Este realiza un plor de los datos calculados anteriormente.

Todos los scripts que deben ser ejecutados se encuentran dentro de la carpeta src.
