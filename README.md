# Maize-plant-disease-diagnosis

El maíz es la base principal de la alimentación de muchos pueblos del continente americano desde tiempos ancestrales, tanto es así que los indios tahinos lo llamaron _mahis_, que significa "aquello que sustenta la vida". A día de hoy, continúa siendo un pilar alimentario indispensable, particularmente en áreas rurales de todo el continente. Su buen crecimiento garantiza seguridad alimentaria, mientras que épocas de malas cosechas traen hambrunas. Por ende, diagnosticar a tiempo una enfermedad en su crecimiento puede ser de vital importancia para familias agricultoras. Los avances tecnológicos y el alcance de la tecnología móvil en las áreas más remotas permite el acceso a diagnóstico de enfermedades en la planta del maíz con alto porcentaje de acierto a tan solo un click. 

El presente cuaderno explica cómo se ha trabajado un modelo de diagnóstico de enfermedades en la planta del maíz, a partir de un modelo de Machine Learning.

![Texto alternativo](../utils/corn-vs-maize-01-1140x641.jpg)

Se propone un modelo de clasificación de imágenes supervisado, de tal modo que ante una imagen nueva pueda clasificar correctamente a la planta de maíz según cuatro categorías:
- Planta enferma: marchitez de Stewart (_blight_)
- Planta enferma: roya común del maíz (_common rust_)
- Planta enferma: Cercospora zeae-maydis o mancha gris del maíz (_gray leaf spot_)
- Planta sana: no tiene ninguna de las tres enfermedades previas

El modelo utilizado ha sido una red neuronal convolucional.

En cuanto a los datos, se tratan de imágenes a color, extraídas de Kaggle: [Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)
