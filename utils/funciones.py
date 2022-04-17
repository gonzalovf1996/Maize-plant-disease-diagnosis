
# ======================================================================================================================================

# Tratamiento de datos
import os, sys
import numpy as np
import pandas as pd

# Gráficos
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

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

# Evaluación del modelo
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, auc, roc_curve

# Path
path_train = r'C:\Users\gonza\OneDrive\Escritorio\Bootcamp_DS\Alumno\ds_thebridge_1_22\3-Machine_Learning\Entregas\data\processed\train'
path_val = r'C:\Users\gonza\OneDrive\Escritorio\Bootcamp_DS\Alumno\ds_thebridge_1_22\3-Machine_Learning\Entregas\data\processed\val'
path_test = r'C:\Users\gonza\OneDrive\Escritorio\Bootcamp_DS\Alumno\ds_thebridge_1_22\3-Machine_Learning\Entregas\data\processed\test'
tipos = os.listdir(path_train)

# ======================================================================================================================================


# EDA
# ==================================================================================================

def mostrar_imagen_de_cada_tipo(path, introduce_un_numero):
    '''
    -----------------------------------------------------------------------------------------
    Devuelve una figura con imágenes de cada una de las categorías

    Input: 
    "path": directorio común
    "introduce_un_numero": índice de la imagen a visualizar

    Output:
    Figura con una imágen por categoría
    -----------------------------------------------------------------------------------------
    '''
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20,5))
    for i, ax in enumerate(axes.flatten()):
        # plotting all 4 images on the same figure
        image_path = os.listdir(path + '/' + tipos[i])[introduce_un_numero]
        ax.title.set_text(tipos[i])
        ax.imshow(cv2.imread(path + '/' + tipos[i] + '/' + image_path)[:,:,::-1], aspect='auto')    
    plt.show()


def contar_imagenes(path, classes):
    class_count = []
    for i in classes:
        class_count.append(len(os.listdir(path + '/' + i)))
        
    df = pd.DataFrame(columns = ["Class_Name", "No of Images"])
    df['Class_Name'] = classes
    df["No of Images"] = class_count
    return df


# Procesamiento de datos
# ==================================================================================================

def define_x_y(img_folder):
    '''
    -----------------------------------------------------------------------------------------
    Devuelve una "X" con el listado de imágenes a clasificar, y una "y" con la categoría
    de cada una de ellas

    Input: 
    "img_folder": directorio común

    Output:
    "X" e "y" en forma de lista, con valores tipo float
    -----------------------------------------------------------------------------------------
    '''
   
    X = list()
    y = list()

    # Definimos las dimensiones de las imágenes
    img_width, img_height = 224, 224

   # Iteramos en el directorio, para cada una de las carpetas clasificadoras, 
    for i in os.listdir(img_folder):
        new_path = img_folder + '/' + i
        for j in os.listdir(new_path): # Iteramos para alcanzar cada una de las imágenes en cada directorio, y pasarlo a un array
            image_path= new_path + '/' + j # path de cada imagen
            image= cv2.imread(image_path, cv2.COLOR_BGR2RGB) # Lee la imagen y la transforma al formato de color apropiado
            image=cv2.resize(image, (img_width, img_height),interpolation = cv2.INTER_AREA) # Da a cada imagen la dimensión indicada
            image=np.array(image)
            image = image.astype('float32') # Convierte la imagen a un numpy array, tipo float
            image /= 255 # Normalizamos la imagen a valores entre 0 y 1 (por defecto van de 0 a 255), ayudará al modelo
            X.append(image)
            y.append(i)
    
    return X, y


def procesamiento_de_datos_X(X_train, X_val, X_test):
    '''
    ----------------------------------------------------------------------------------------------
    Tranforma "X" para las muestras de train / test / val

    Input:
    X_train, X_val, X_test

    Output:
    X_train, X_val, X_test
    ----------------------------------------------------------------------------------------------
    '''
    # Transformamos "X" en numpy arrays
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)

    return X_train, X_val, X_test


def procesamiento_de_datos_y(y_train,y_val, y_test):
    '''
    ----------------------------------------------------------------------------------------------
    Tranforma "y" para las muestras de train / test / val

    Input:
    y_train, y_val, y_test

    Output:
    y_train, y_val, y_test
    ----------------------------------------------------------------------------------------------
    '''
    # Tranformos las variables categegóricas (y) en numéricas
    target_dict_train ={k: v for v, k in enumerate(np.unique(y_train))}
    target_dict_val ={k: v for v, k in enumerate(np.unique(y_val))}
    target_dict_test ={k: v for v, k in enumerate(np.unique(y_test))}

    # Transformamos las variablse target "y_train", "y_val" e "y_test" en numéricas 
    y_train =  [target_dict_train[y_train[i]] for i in range(len(y_train))]
    y_val =  [target_dict_val[y_val[i]] for i in range(len(y_val))]
    y_test =  [target_dict_test[y_test[i]] for i in range(len(y_test))]

    # Transformamos "y" en numpy arrays
    y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
    y_val = np.asarray(y_val).astype('float32').reshape((-1,1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1,1))

    return  y_train, y_val, y_test


# Diseño del modelo
# ==================================================================================================

def definir_modelo(base_model):
    '''
    ----------------------------------------------------------------------------------------------
    Define las capas del modelo

    Input: 
    "base_model": modelo de entrada

    Output:
    Modelo definido
    ----------------------------------------------------------------------------------------------
    '''
    # Transfer learning
    model = Sequential(
    [hub.KerasLayer(base_model,
                   trainable=False), # Para que se inicialice con todo el aprendizaje que ya posee
    tf.keras.layers.Dense(len(tipos), activation='softmax')
    ])

    # Componemos el modelo
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Evaluación del modelo
# ==================================================================================================

def plot_metrics(history, metric_name, title, ylim=5):
    '''
    ----------------------------------------------------------------------------------------------
    Muestra la diferencia entre la función de pérdidas en "train" y en "val"

    Input: 
    "history": modelo de entrada
    "metric_name": la métrica que se quiere mostrar
    "title": título para la gráfica
    "y_lim": lo que queremos mostrar en el eje y
    ----------------------------------------------------------------------------------------------
    '''
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(history.history[metric_name],color='blue',label=metric_name)
    plt.plot(history.history['val_' + metric_name],color='green',label='val_' + metric_name)
    plt.legend()


def plot_cm(y_real, y_pred):
  '''
  ----------------------------------------------------------------------------------------------
  Muestra la matriz de correlación

  Input: 
  "y_real": categorías reales a las que pertenece cada imagen
  "y_pred": las cateogrías predichas por el modelo
  ----------------------------------------------------------------------------------------------
  '''
  # Matriz de confusión
  cm = confusion_matrix(y_real, y_pred)
  cm_dec = [ cm[i] / np.sum(cm[i]) for i in range(len(cm))] # Lo normalizamos para cada categoría
  cm_per = [ [ int(round(x*100,1)) for x in my_list] for my_list in cm_dec ] # Lo pasamos a %

  # figura
  plt.figure(figsize=(12,8))
  ax = plt.subplot()
  sns.heatmap(cm, annot=cm_per, fmt="d", cmap='Blues')
  ax.set_xlabel('Predicción');ax.set_ylabel('Categoría real'); 
  ax.set_title('Matriz de confusión\n(unidades en porcentaje del total por categoría)'); 
  ax.xaxis.set_ticklabels(tipos, rotation=10); ax.yaxis.set_ticklabels(tipos, rotation=10);


def metricas(y_real, y_pred):
    '''
    ----------------------------------------------------------------------------------------------
    Muestra las métricas de evaluación básicas de un modelo

    Input: 
    "y_real": categorías reales a las que pertenece cada imagen
    "y_pred": las cateogrías predichas por el modelo
    ----------------------------------------------------------------------------------------------
    '''
    print('Accuracy: %.3f' % accuracy_score(y_real, np.argmax(y_pred, axis=1)))
    print('Precision: %.3f' % precision_score(y_real, np.argmax(y_pred, axis=1), average='weighted'))
    print('Recall: %.3f' % recall_score(y_real, np.argmax(y_pred, axis=1), average='weighted'))


def plot_roc_curve(y_real, y_pred):
  '''
  ----------------------------------------------------------------------------------------------
  Muestra la curva ROC, como métrica de evaluación de un modelo

  Input: 
  "y_real": categorías reales a las que pertenece cada imagen
  "y_pred": las cateogrías predichas por el modelo
  ----------------------------------------------------------------------------------------------
  '''
  n_classes = len(tipos)
  y_real = label_binarize(y_real, classes=np.arange(n_classes))

  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  thresholds = dict()
  for i in range(n_classes):
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_real[:, i], y_pred[:, i], drop_intermediate=False)
  roc_auc[i] = auc(fpr[i], tpr[i])

  # First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

  # Finally average it and compute AUC
  mean_tpr /= n_classes

  fpr["weigthed"] = all_fpr
  tpr["weigthed"] = mean_tpr
  roc_auc["weigthed"] = auc(fpr["weigthed"], tpr["weigthed"])

  # Plot all ROC curves
  #plt.figure(figsize=(10,5))
  plt.figure(figsize=(12,7))
  lw = 2

  plt.plot(fpr["weigthed"], tpr["weigthed"],
  label="weigthed-average ROC curve (area = {0:0.2f})".format(roc_auc["weigthed"]),
  color="navy", linestyle=":", linewidth=4,)

  colors = cycle(["red","brown", "orange", "blue"])
  for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    label=("ROC curve de:", tipos[i]))

  plt.plot([0, 1], [0, 1], "k--", lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel("False Positive Rate - Recall")
  plt.ylabel("True Positive Rate - Precision")
  plt.title("Curva ROC")
  plt.legend()


def recall(model_name, y_real, y_pred):
    # Set styles for axes
    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=0.8
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'

    # Plot
    fig, ax = plt.subplots(figsize=(12,5))
    plt.hlines(model_name, xmin=0.4, xmax=y_pred, color='#007acc', alpha=0.5, linewidth=5)

    plt.xticks(rotation=90)
    plt.plot(y_pred, model_name, "o", markersize=5, color='#007acc', alpha=0.9)
    plt.title('Sensibilidad (Recall) en cada modelo')
    plt.xlabel('%')
    plt.show()

# ======================================================================================================================================