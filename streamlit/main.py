
# LIBRERÍAS Y FUNCIONES -------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
from functions import preparacion, prediccion, comentarios, probabilidades
import streamlit as st


# CONFIGURACIÓN DE LA PÁGINA --------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
st.set_page_config(page_title='Diagnóstico móvil', layout='wide', page_icon='chart_with_upwards_trend')

st.sidebar.title("Predice con imágenes de muestra, o sube una foto")

opciones= ['Subir foto', 'Planta de Estados Unidos', 'Planta de Guatemala', 'Planta de México', 'Planta de Puerto Rico']
imageselect = st.sidebar.selectbox("Escoge una imagen", opciones)

# PÁGINA PRINCIPAL ------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
st.header('Diagnóstico de la salud de la planta de maíz')
st.write('Elaborado por Gonzalo Villalón Fornés')

st.image('dataset-cover.jpg', caption='Detectar una enfermedad en la cosecha a tiempo es crucial para \
    procurar una plantación de maíz que crezca sana y fuerte')

with st.expander('¿Qué enfermedades es capaz de diagnosticar este modelo?'):
    st.write('El modelo predecie con altos niveles de precisión entre cuatro categorías:')
    st.write('1. Marchitez bacterial de Stewart')
    st.write('2. Roya común del maíz')
    st.write('3. La mancha gris o rectangular del maíz')
    st.write('4. Plantas de maíz sanas')
    st.write('Por tanto, Como esta aplicación diagnostica únicamente estas enfermedades, recuerde mantener la alerta e \
            informarse si percibe características anómalas que no se ajustan al diagnóstico de presente modelo.')


# DIAGNOSTICO -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

if st.sidebar.button('Diagnóstico'):

    if imageselect == 'Subir foto':
        st.write('Para un diagnóstico más adecuado se aconseja tomar más de una foto a la planta en cuestión \
                y diagnosticar cada foto tomada. Así, se obtendrán resultados más robustos.')
        image_file  = st.file_uploader("Sube una foto de tu planta de maíz enferma", type=["png","jpg","jpeg"])

        if image_file  is not None:
            # To View Uploaded Image
            st.write('Diagnóstico ejecutado para esta imagen:')
            st.image(image_file, width=400)
            image_file = preparacion(image_file)
            diagnostico, my_model = prediccion(image_file)
            st.write('Diagnóstico: ', diagnostico)

            with st.expander('Probabilidad de diagnóstico en esta imagen'):
                fig = probabilidades(my_model, image_file)
                st.pyplot(fig)

            comentarios(my_model, image_file)
            
        else:
            st.write('No dude en insertar una imágen en uno de los siguientes formatos: png, jpg, jpeg.')

    elif imageselect == 'Planta de Estados Unidos':
        image_file  = '1.jpg'
        # To View Uploaded Image
        st.write('Diagnóstico ejecutado para esta imagen:')
        st.image(image_file, width=400)
        image_file = preparacion(image_file)
        diagnostico, my_model = prediccion(image_file)
        st.write('Diagnóstico: ', diagnostico)

        with st.expander('Probabilidad de diagnóstico en esta imagen'):
            fig = probabilidades(my_model, image_file)
            st.pyplot(fig)

        comentarios(my_model, image_file)

    elif imageselect == 'Planta de Guatemala':
        image_file  = '2.jpg'
        # To View Uploaded Image
        st.write('Diagnóstico ejecutado para esta imagen:')
        st.image(image_file, width=400)
        image_file = preparacion(image_file)
        diagnostico, my_model = prediccion(image_file)
        st.write('Diagnóstico: ', diagnostico)
        
        with st.expander('Probabilidad de diagnóstico en esta imagen'):
            fig = probabilidades(my_model, image_file)
            st.pyplot(fig)

        comentarios(my_model, image_file)

    elif imageselect == 'Planta de México':
        image_file  = '3.jpg'
        # To View Uploaded Image
        st.write('Diagnóstico ejecutado para esta imagen:')
        st.image(image_file, width=400)
        image_file = preparacion(image_file)
        diagnostico, my_model = prediccion(image_file)
        st.write('Diagnóstico: ', diagnostico)
        
        with st.expander('Probabilidad de diagnóstico en esta imagen'):
            fig = probabilidades(my_model, image_file)
            st.pyplot(fig)

        comentarios(my_model, image_file)

    elif imageselect == 'Planta de Puerto Rico':
        image_file  = '4.jpg'
        # To View Uploaded Image
        st.write('Diagnóstico ejecutado para esta imagen:')
        st.image(image_file, width=400)
        image_file = preparacion(image_file)
        diagnostico, my_model = prediccion(image_file)
        st.write('Diagnóstico: ', diagnostico)
        
        with st.expander('Probabilidad de diagnóstico en esta imagen'):
            fig = probabilidades(my_model, image_file)
            st.pyplot(fig)

        comentarios(my_model, image_file)


# AUSENCIA DE DIAGNOSTICO -----------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

else: 
    st.write('Escoja una opción de la barra lateral. Verá la imagen escogida o tendrá la opción de subir una foto.')
    st.write('Cuando esté preparado, solicite un diagnóstico')
    st.image('Plant-disease-classifier-with-ai-blog-banner.jpg', caption='¡Diagnostiquemos la salud de su planta de maíz!')
    # st.subheader('¿Se puede evitar una mala cosecha?')
    # st.write('En muchas ocasiones, las malas cosechas están producidas por la propagación de una \
    #     enfermedad entre nuestros cultivos. Identifiquemos la enfermedad que tiene nuestro maíz a tiempo, y adelantémonos \
    #     a posibles contagios.')



# CALIDAD DE DIAGNÓSTICO ------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

barra_hor = st.checkbox('Sobre el modelo utilizado')
if barra_hor:
    st.write('Se trata de una Red Neuronal Convolucional (CNN) a la cual se le ha aplicado Transfer Learning. \
        La red neuronal base aplicada es "mobilenet_v3_large_100_224".')

    st.write('Para su entrenamiento, se han utilizado los datos aportados por __[Kaggle](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)__.')

    st.write('Matriz de confusión ante datos nuevos:')

    st.image('cm_my_model.png')

    st.write('Resultados obtenidos ante datos nuevos:')
    st.write('Accuracy: 91,3%')
    st.write('Precision: 91,1%')
    st.write('Recall: 91,3%')

    
