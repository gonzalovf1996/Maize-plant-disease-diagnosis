# Diagnóstico de enfermedad en la planta de maíz

El maíz es la base principal de la alimentación de muchos pueblos del continente americano desde tiempos ancestrales, tanto es así que los indios tahinos lo llamaron _mahis_, que significa "aquello que sustenta la vida". A día de hoy, continúa siendo un pilar alimentario indispensable, particularmente en áreas rurales de todo el continente. Su buen crecimiento garantiza seguridad alimentaria, mientras que épocas de malas cosechas traen hambrunas. Por ende, diagnosticar a tiempo una enfermedad en su crecimiento puede ser de vital importancia para familias agricultoras. Los avances tecnológicos y el alcance de la tecnología móvil en las áreas más remotas permite el acceso a diagnóstico de enfermedades en la planta del maíz con alto porcentaje de acierto a tan solo un click. 

El presente cuaderno explica cómo se ha trabajado un modelo de diagnóstico de enfermedades en la planta del maíz, a partir de un modelo de Machine Learning.

![Texto alternativo](/src/utils/corn-vs-maize-01-1140x641.jpg)

Se propone un modelo de clasificación de imágenes supervisado, de tal modo que ante una imagen nueva pueda clasificar correctamente a la planta de maíz según cuatro categorías:
- Planta enferma: marchitez de Stewart (_blight_)
- Planta enferma: roya común del maíz (_common rust_)
- Planta enferma: Cercospora zeae-maydis o mancha gris del maíz (_gray leaf spot_)
- Planta sana: no tiene ninguna de las tres enfermedades previas

El modelo utilizado ha sido una red neuronal convolucional.

En cuanto a los datos, se tratan de imágenes a color, extraídas de Kaggle: [Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)

----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------

El modelo obtenido presenta los siguientes resultados:
- Accuracy: 0.913
- Precision: 0.911
- __Recall__: 0.913

Se presta especial atención a la sensibilidad (_Recall_) ya que es la métrica que trata de medir el mayor rendimiento en el diagnostico de enfermedad en la planta cuando realmente está enferma. En este caso, el modelo tiene una sensibilidad del 91,3%.

En cuanto a la matriz de confusión, encontramos ciertas disparidades en la categorización de cada enfermedad:

![Texto alternativo](/src/utils/cm_my_model.png)

Llama la atención dos datos significativos:
- Solamente el 70% de las plantas con Cercospora zeae-maydis o mancha gris del maíz (_gray leaf spot_) han sido categorizadas correctamente.
- El 100% de las plantas sin las enfermedades correspondientes han sido identificadas correctamente.

Esto quiere decir que el modelo sabe identificar perfectamente cuando una planta está enferma y cuando no, y que tiene un alto margen de error (del 30%) a la hora de clasificar a las plantas con la mancha gris del maíz (_gray leaf spot_). Este alto margen de error  coincide con que es la categoría de las que menos imágenes se dispone, tal y como podemos ver a continuación:

![Texto alternativo](/src/utils/number_images.png)

----------------------------------------------------------------------------------------------
En conclusión, el entrenamiento del modelo actual (_"my_model"_) ante una mayor cantidad de imágenes de plantas con la mancha gris del maíz (_gray leaf spot_) en el tamaño muestral podría minimizar el error de predicción existente de forma significativa.
