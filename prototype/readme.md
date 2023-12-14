Files needed to design, train and export the machine learning algorithm to ESP32. The design and training are done in Python and then the model is converted to C++.
For this the following steps have to be performed:
1) Install the libraries listed in `requirements.txt.`.
2) Install in Arduino IDE the `EloquenTinyML` library from the `Library Manager`.
3) Install in Arduino IDE the `filters` library from the [repository](https://github.com/MartinBloedorn/libFilter).
4) Data for training: you can use data that was already acquired or acquire new data. Each move must be in a separate .csv file and the file name identifies the move. E.g. `1.csv` (record of movement 1).
5) Run the script `src/generateModel.py`. In this script you must set the number of registered movements that you want to classify. You can also modify the design of the algorithm. After executed a `model.h` file will be generated in the `prototype/ESP32_MLP/src` folder.
6) Modify line 39 of `model.h`. Replace `<32, 1, arenaSize>` with `<32, 9, arenaSize>`, where 9 is the number of gestures to be sorted.
7) Upload the model to ESP32 from the `ESP32_MLP/src/ESP32_MLP.ino` file. 
8) Capture classified data to evaluate the performance of the classifier implemented in the ESP32. The actual labeling is performed by the module puslator and the prediction is performed by the classifier. Generate a .csv file per movement and name it `eval_N°.csv`, where N° is the movement number. E.g. `eval_1.csv`, evaluation of movement 1.
9) Run the script `src/evaluateModel.py`. Configure the number of acquired movements.

Archivos necesarios para diseñar, entrenar y exprotar el algoritmos de machine learning al ESP32. El diseño y entrenamiento se hacen en Python luego el modelo es covnertido a C++.
Para esto hay que realziar los sigueientes pasos:
1) Instalar las librerías listadas en `requirements.txt.`
2) Instalar en Arduino IDE la librería `EloquenTinyML` desde el `Gestor de Bibliotecas`.
3) Instalar en Arduino IDE la librería `filters` desde el [repositorio](https://github.com/MartinBloedorn/libFilter).
4) Datos para el entrenamiento: se pueden utilizar los datos que ya fueron adquiridos o adquirir nuevos datos. Cada movimiento debe estar en un archivo .csv por separado y el nombre del archivo identifica el movimiento. Ej. `1.csv` (registro del movimiento 1).
5) Ejecutar el script `src/generateModel.py`. En este se debe setear la cantidad de movimientos registrados y que se desean clasificar. También se puede modificar el diseño del algoritmo. Luego de ejecutado se generará un archivo `model.h` en el directorio `prototype/ESP32_MLP/src`.
6) Modificar la línea 39 de `model.h`. Remplazar `<32, 1, arenaSize>` por `<32, 9, arenaSize>`, donde 9 es la cantidad de gestos que se desean clasificar.
7) Subir el modelo a el ESP32 a partir del archivo `ESP32_MLP/src/ESP32_MLP.ino`. 
8) Capturar datos clasificados para evaluar el desempeño del clasificador implementado en la ESP32. La etiqueta real se realiza con el puslador del módulo y la predicción es realziada por el calasificador. Generar un archivo .csv por movimeinto y nombrearlos como `eval_N°.csv`, donde N° es el número del movimiento. Ej. `eval_1.csv`, evaluación del movimiento 1.
9) Ejecutar el script `src/evaluateModel.py`. Configurar la cantida de movimientos adquiridos.
 
Copyright (C) 2023  Renato Sosa Machado Scheeffer. Universidad de la República, Uruguay.
