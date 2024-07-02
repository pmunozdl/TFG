# TFG

En este fichero se incluye un breve resumen del proyecto realizado junto a una documentación guiada para su ejecución. 

## Resumen
Se ha realizado un cuadro de mando integral que refleja la actividad de una empresa, segmentada en los perfiles que representan las principales áreas funcionales de la misma. Se ha complementado el análisis con predicciones del año próximo, para la que se han empleados modelos de inteligencia artificial como Redes Neuronales, Regresión Lineal o Suavizado Exponencial. 

En cada una de las ventanas se incluyen elementos visuales, como gráficos e indicadores, que denotan de forma clara e intuitiva el estado de la empresa en cada uno de los aspectos analizados. 

## Tecnologías implementadas
    - Power BI
    - Streamlit
    - Jupyter Notebook
    - Modelos de aprendizaje supervisado, citados a continuación:
        - Regresión Lineal
        - Random Forest
        - Suavización Exponencial
        - Sarimax
        - Red Neuronal
    - Docker

## Composición del proyecto
A continuación, se listan los ficheros que componen el proyecto junto a una breve explicación de estos:
    - **Cuadro de Mando/Nuevo Cuadro de mando.pbix:** Cuadro de mando resultante del análisis de la actividad de la empresa. 
    - **Dataset/VentasEcommerce.xslx:** Conjunto de datos de la actividad de la empresa empleada como base para realizar el análisis. 
    - **/Test:** Contiene los ficheros de las predicciones realizadas y los test cases para comprobar que los resultados esperados coinciden con los reales
    - **nuevaVersion.ipynb:** Fichero en formato de Jupyter Notebook que contiene, de forma extensa, todas las predicciones y las comparaciones entre los resultados de los modelos. 
    - **script.py:** Script insertado en la herramienta PowerBi para incluir la nueva tabla con los resultados de las predicciones. 
    - **streamlit.py:** Cuadro de mando complementario, implementado en el framework Streamlit, que permite analizar los resultados de las predicciones. 

## Ejecución del proyecto
Para ejecutar el proyecto, el único requisito es disponer de un gestor de paquetes. En este caso se ha empleado Docker. Deben seguirse los siguientes pasos:
1. Clonar este repositorio en tu máquina.
    ```bash
    git clone https://github.com/pmunozdl/TFG.git
    ```

2. Acceder al directorio del proyecto.
    ```bash
    cd TFG  
    ```

3. Construir la imagen del Docker.
    ```bash
    docker build -t <nombre_imagen> .
    ```

4. Ejecutar el contenedor docker.
    ```bash
    docker run -d -p 8501:8501 -p 8888:8888 --name <nombre_contenedor> <nombre_imagen>
    ```

5. Comprobar que se ha iniciado correctamente
    ```bash
    docker ps   
    ```

6. Acceder a Streamlit
    ```bash
    http://localhost:8501   
    ```

7. Acceder a Jupyter Notebook
    7.1. Acceder a la terminal
    ```bash
    docker exec -it <nombre_contenedor> /bin/bash   
    ```
    7.2. Obtener el token de inicio de sesión. Copiar el código de 48 cifras que aparece tras "token="
    ```bash
    jupyter notebook list  
    ```
    7.3. Acceder a Jupyter Notebook y añadir el token
    ```bash
    http://localhost:8888   
    ```
8. Ejecutar los test
    8.1. Acceder a la terminal
    ```bash
    docker exec -it <nombre_contenedor> /bin/bash   
    ```
    8.2. Acceder al directorio de testing
    ```bash
    cd /test  
    ```
    8.3. Ejecutar los test
    ```bash
    python -m unittest -v pruebas.py   
    ```