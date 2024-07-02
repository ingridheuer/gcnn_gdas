# Redes neuronales definidas en grafos para la predicción de nuevas asociaciones gen-enfermedad

Exploración e implementación de Graph Neural Networks para hallar asociaciones entre genes y enfermedades de origen genético en bases de datos biomédicas.

Las secciones de integración de datos, análisis de redes complejas y procesamiento lenguaje natural se presentaron en el congreso [NetSci-x 2023](https://cnet.fi.uba.ar/netscix23/).
![poster_netsci](https://user-images.githubusercontent.com/61297025/236244985-1911d0fb-7dee-4094-b0f5-44747ba50e21.jpg)

Organización del Proyecto
------------

    ├── license
    ├── README.md                 
    ├── data                      <- Este directorio no es público
    │   ├── external              <- Datos de fuentes externas. Bases de datos originales como DisGeNET y HIPPIE 
    │   ├── interim               <- Pasos intermedios en procesamiento de datos y mapeos de vocabulario manuales.
    │   └── processed             <- Dataset final, contiene el grafo procesado y sus atributos.
    │       
    │       
    ├── models                    <- Evaluación y descripción del modelo final
    │       
    ├── exploration               <- Exploración de datos, notebooks y algunos scripts experimentales (no definitivos)
    |   └── run_in_colab          <- Scripts para correr en Google Colab - Caminatas aleatorias en espacio de hiperparámetros
    │       
    ├── references                <- Data dictionaries, manuals, and all other explanatory materials. (TODO: por ahora solo fuentes originales de los datos)
    │       
    ├── reports                   <- Gráficos, tablas, etc. Este directorio no es público.
    │       
    ├── requirements.txt          <- Requisitos para reproducir el entorno. Generado con `pip freeze > requirements.txt`
    │       
    │
    │
    ├── src                       <- Código fuente del proyecto.
    │   │       
    │   ├── data                  <- Integración y curado de datos, split del dataset.
    │   │          
    │   │       
    │   ├── features              <- Generación de atributos de nodos 
    │   │   └── build_features.py
    │   │
    │   ├── models                <- Modelos, scripts para entrenar y evaluar modelos, scripts para hacer predicciones con modelos.
    │   │   │                 
    │   │   ├── base_model.py     <- Configurar y generar modelos
    │   │   ├── training_utils.py <- Funciones de entrenamiento y evaluación. Implementación de muestreo negativo. Utilidades para cargar datos.
    │   │   ├── prediction.py     <- Utilidades para hacer predicciones y mapear datasets entre formato torch-geometric y pandas. 
    │   │   └── final_model.py    <- Implementación del modelo final 
    │   │
    │   ├── network_analysis      <- Análisis de redes complejas
    │   │
    │   ├── NLP_analysis          <- Procesamiento y análisis de descripciones clínicas de nodos enfermedad
    │   │   │                 
    │   │   ├── preprocess_corpus.py    <- Preprocesamiento de descripciones clínicas
    │   │   ├── vectorize.py            <- Generación de matrices document-term con TF-IDF
    │   │   └── LSA_dim_reduction.py    <- Latent Semantic Analysis
    │   │
    │   │
    │   └── visualization  <- Visualizaciones
    │       ├── visualize_clusters.py     <- Buscador de clusters y pathways a partir de palabras clave (síntomas o partes del cuerpo)
            └── visualize_embeddings.py   <- Ver el espacio latente generado por el modelo
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
