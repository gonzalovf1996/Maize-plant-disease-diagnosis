
# Tratamiento de datos
import os
import numpy as np
import pandas as pd

# Preprocesado y modelado
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Tratamiento de imágenes
import cv2
import splitfolders
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from keras.models import Sequential
from keras.callbacks import EarlyStopping 

from utils.funciones import define_x_y, definir_modelo


# 1. Procesamiento de datos
# ================================================================================================================

# Es necesario definir el directorio de trabajo
os.chdir(str(input('Introduce directorio de trabajo:')))

# Dividimos los datos en train y test

if 'train' not in os.listdir(r'data/processed') and 'val' not in os.listdir(r'data/processed'): # Si no se ha ejecutado ya, entonces se ejecutará
    print('Para guardar el modelo, es necesario que los datos de las imágenes estén, dentro de su directorio del proyecto, en "data/raw". Si no lo tiene así guardado, por favor, descárgelos de https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset y guárdelos en "data/raw"')
    path = r'data/raw' # Directorio donde están ubicadas las imágenes sin dividir en train /test /val
    splitfolders.ratio(path, output="data/processed", seed=42, ratio=(.8, .1, .1), move=True) # Así se divide en train /test /val y se guardan en el directoriio indicado
    print('Las imágenes descargadas se han movido a "data/processed" y se han separado en train / val y / test respectivamente')

# Definimos los directorios donde se encuentran las imágenes
path_train = r'data/processed/train'
path_val = r'data/processed/val'
path_test = r'data/processed/test'
tipos = os.listdir(path_train) # Categorías a clasificar


# Tranformaciones



# Definimos el path del directorio y creamos dataset
X_train, y_train = define_x_y(path_train)
X_val, y_val = define_x_y(path_val)
X_test, y_test = define_x_y(path_test)

# Damos un valor numérico a cada categoría
target_dict_train ={k: v for v, k in enumerate(np.unique(y_train))}
target_dict_val ={k: v for v, k in enumerate(np.unique(y_val))}
target_dict_test ={k: v for v, k in enumerate(np.unique(y_test))}

if type(y_train) == list: # Si esta celda no se ha ejectuado aún, entonces se ejecutará

    # Transformamos las variablse target "y_train", "y_val" e "y_test" en numéricas 
    y_train =  [target_dict_train[y_train[i]] for i in range(len(y_train))]
    y_val =  [target_dict_val[y_val[i]] for i in range(len(y_val))]
    y_test =  [target_dict_test[y_test[i]] for i in range(len(y_test))]

    # Transformamos tanto "X" como "y" en numpy arrays
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)

    y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
    y_val = np.asarray(y_val).astype('float32').reshape((-1,1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1,1))



# 2. Definición del modelo
# ================================================================================================================

# Escogemos la red neuronal base:
base_model = 'https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5' 

model = definir_modelo(base_model)

print('Aplicándose transfer learning')



# 3. Entrenamiento del modelo
# ================================================================================================================

# Para prevenir overfitting, establecemos un número alto de 'epoch'
print("Su modelo se está entrenando. Vaya a por un café o tómese un descanso, y vuelva en unos minutos")

earlystopping = EarlyStopping(monitor='val_loss', # Ante datos nuevos (distintos del test), parará
                                mode='min', patience=5,
                                restore_best_weights=True)

model.fit(X_train, y_train, 
                    epochs=100, # el número de veces que se reentrena el modelo
                    batch_size=128, # El número de samples que se procesan antes de que el modelo se actualice
                    verbose=2, 
                    callbacks=[earlystopping], # Parará en el 'epoch' óptimo
                    validation_data=(X_val, y_val) 
                    )


print(model.summary())


# 4. Guardado del modelo
# ================================================================================================================

# Vamos a guardar nuestro modelo
if 'my_model' not in os.listdir(r'model'): # Si no se ha ejecutado esta celda ya, entonces se ejecutará
    model.save("my_model")
    print('su modelo ha sido guardado con éxito')

else:
    print('su modelo ya ha sido guardado previamente')