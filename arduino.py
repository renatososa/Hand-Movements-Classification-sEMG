# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 22:09:58 2022

@author: Renato
"""


from joblib import load
import tensorflow as tf

FEATURES = load("FEATURES.joblib")
label = load("label.joblib")

red = tf.keras.models.load_model('./')
eval_loss, eval_acc = red.evaluate(FEATURES,  label, verbose=1)
print('Eval accuracy percentage: {:.2f}'.format(eval_acc * 100))
print(red.predict(FEATURES)[0])
red_converter = tf.lite.TFLiteConverter.from_keras_model(red)
tflite_model = red_converter.convert()
open("MLP.tflite", "wb").write(tflite_model)
