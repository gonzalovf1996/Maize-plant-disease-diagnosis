
# LIBRERÍAS--------------------
# -----------------------------
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import streamlit as st
import seaborn as sns
from PIL import Image
import pandas as pd
import numpy as np


# DIAGNÓSTICO -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def preparacion(imagen):
    '''
    ---------------------------------------------------------------------------------------------
    Input: imagen a transformar
    Output: 'image_file' > la imagen preparada acorde al modelo
    ---------------------------------------------------------------------------------------------
    '''
    image_file = Image.open(imagen)
    image_file = tf.image.resize(image_file, size=(224,224))
    image_file = tf.expand_dims(image_file, axis=0)
    image_file = np.array(image_file)
    image_file = np.array(image_file).astype('float32')
    image_file /= 255 # Normalizamos la imagen a valores entre 0 y 1 (por defecto van de 0 a 255)

    return image_file


def prediccion(imagen):
    '''
    ---------------------------------------------------------------------------------------------
    Input: imagen a transformar
    Output: 1. diagnóstico, 2. modelo CNN
    ---------------------------------------------------------------------------------------------
    '''
    # Ejecutamos el diagnóstico
    my_model = keras.models.load_model('src\model\my_model') # cargamos el modelo
    prediction = np.argmax(my_model.predict(imagen), axis=1) # obtenemos la predicción 

    # Los resultados serán numéricos, y debemos pasarlo a categóricos acorde a:
    resultado = {'MARCHITEZ DE STEWART': 0, 'ROYA COMÚN DEL MAÍZ': 1, 'CERCOSPORA ZEAE-MAYDIS \
        (MANCHA GRIS DEL MAÍZ)': 2, 'PLANTA SANA': 3}
    return list(resultado.keys())[prediction[0]], my_model


def comentarios(modelo, imagen):
    '''
    ---------------------------------------------------------------------------------------------
    Input: modelo CNN, imagen transformada
    ---------------------------------------------------------------------------------------------
    '''
    if np.argmax(modelo.predict(imagen), axis=1) == 0:
        st.subheader('Marchitez bacterial de Stewart')
        st.write('Manifestándose en forma de rayas lineales en las hojas de maíz, la marchitez del maíz (mancha bacteriana de \
                    la hoja de maíz) es causada por una bacteria llamada Erwinia stewartii . Las infecciones se clasifican generalmente \
                    en dos tipos basados en el momento en que cada una de ellas ocurre: la etapa de plántula y la etapa de marchitez \
                    de la hoja, que afecta a las plantas más viejas y maduras. Cuando se infecta con la marchitez de Stewart, el \
                    maíz dulce puede morir prematuramente sin importar la edad de la planta, si la infección es severa.')
                
        st.write('Pantoea stewartii es transmitida por la semilla y por algunos coleópteros del maíz (ej. Chaetocnema pulicaria). \
                    Debe controlar los insectos que tiene de inmediato para prevenir más contagios.')

    if np.argmax(modelo.predict(imagen), axis=1) == 1:
        st.subheader('Roya común del maíz')
        st.write('Afecta mayormente plantas que están cercanas a la floración. Esta enfermedad se reconoce por la presencia de \
                    pústulas pequeñas y polvorientas, de color café en ambos lados de las hojas. A medida que la planta madura la epidermis \
                    de la hoja se rompe y las manchas se tornan obscuras. El tejido joven de la planta es más susceptible a la roya que \
                    el tejido maduro. La mayoría de los híbridos de maíz son resistentes a la roya común. Esta roya desarrolla parte \
                    de su ciclo en un hospedero alterno, Oxalis spp., donde produce pústulas de color anaranjado claro. El \
                    desarrollo de pústulas en esta roya es igual de abundante en ambos lados de la hoja, lo que la diferencia \
                    de la roya sureña, la cual tiene muy poca producción de pústulas en el envés de la hoja.')

        st.write('Para un manejo integral, debe acudir a expertos para la aplicación de fungicidas foliares con permiso de uso.')

    if np.argmax(modelo.predict(imagen), axis=1) == 2:
        st.subheader('Mancha gris del maíz')
        st.write('La mancha gris o rectangular del maíz aparece primero con pequeñas manchas necróticas con halos. \
                    Generalmente se expande para convertirse en lesiones rectangulares, de apariencia gris o marrón, de entre \
                    1mm de ancho por hasta 5 u 8 cm de largo. La enfermedad se desarrolla desde la aparación de los estigmas \
                    hasta la madurez. Suele afectar más fuertemente cuando se trata de un monocultivo de maíz, bajo \
                    siembra directa o labranza reducida.')
        st.write('Para un manejo integrado, aplique la rotación de cultivos o \
                    evite prácticas de monocultivo. Además, debe acudir a expertos para la aplicación de fungicidas \
                    foliares con permiso de uso.')

    if np.argmax(modelo.predict(imagen), axis=1) == 3:
        st.subheader('No se ha detectado ninguna enfermedad. Su planta parece estar sana, ¡felicidades!')


def probabilidades(modelo, imagen):
    '''
    ---------------------------------------------------------------------------------------------
    Input: modelo CNN, imagen transformada
    Output: gráfica con probabilidad de pertenencia a cada categoría
    ---------------------------------------------------------------------------------------------
    '''
    # composición de df
    df = pd.DataFrame(modelo.predict(imagen)).T
    df[0] = df[0].round(3)
    df['enfermedad'] = ['Marchitez de Stewart', 'Roya común', 'Mancha gris', 'Planta sana']
    # Barplot
    fig = plt.figure(figsize=(10,4))
    ax = sns.barplot(x=df.iloc[:,0], y=df.iloc[:,1], palette='Blues_d')
    plt.ylabel('', fontsize=18)
    plt.xlabel('', fontsize=18)
    ax.bar_label(ax.containers[0])
    return fig
