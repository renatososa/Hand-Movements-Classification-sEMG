Archivos necesarios para diseñar, entrenar y exprotar el algoritmos de machine learning al ESP32. El diseño y entrenamiento se hacen en Python luego el modelo es covnertido a C++.
Para esto hay que realziar los sigueientes pasos:
1) Instalar las librerías listadas en `requirements.txt.`
2) Instalar en Arduino IDE la librería `EloquenTinyML` desde el `Gestor de Bibliotecas`.
3) Instalar en Arduino IDE la librería `filters` desde el [repositorio](https://github.com/MartinBloedorn/libFilter).
4) Datos para el entrenamiento: se pueden utilizar los datos que ya fueron adquiridos o adquirir nuevos datos. Cada movimiento debe estar en un archivo .csv por separado y el nombre del archivo identifica el movimiento. Ej. `1.csv` (registro del movimiento 1).
5) Ejecutar el script `src/generateModel.py`. En este se debe setear la cantidad de movimientos registrados y que se desean clasificar. También se puede modificar el diseño del algoritmo. Luego de ejecutado se generará un archivo model.h en el directorio prototype/ESP32_MLP/src
6) Modificar la linea línea 39 de model.h: <32, 1, arenaSize> por: <32, 9, arenaSize> donde 9 es la cantidad de gestos que se desean clasificar.
7) Subir el modelo a el ESP32 a partir del archivo ESP32_MLP.ino en la carpeta ESP32_MLP. 
8) Capturar datos clasificados para evaluar el desempeño del clasificador implementado en la ESP32. La etiqueta real se realiza con el puslador del módulo y la predicción es realziada por el calasificador. Generar un archivo .csv por movimeinto y nombrearlos como eval_N°.csv, donde N° es el número del movimiento. Ej. eval_1.csv, evaluación del movimeinto 1.
9) Ejecutar el script `src/evaluateModel.py`. Configurar la cantida de movimientos adquiridos.
 
Copyright (C) 2023  Renato Sosa Machado Shceeffer. Universidad de la República, Uruguay.
