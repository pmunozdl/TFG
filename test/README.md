## Test

Esta carpeta contiene el fichero predicciones.py, donde se ejecutan los modelos de aprendizaje automático para realizar las predicciones, de forma resumida.

También contiene el fichero pruebas.py, que contiene los test cases definidos para comprobar que el resultado esperado, y mostrado en el cuadro de mando, coincide con
los valores que se están obteniendo realmente. Las pruebas unitarias comprueban, de forma independiente y automática, el funcionamiento del código en casos específicos. 

##  Pasos para la ejecución

Para ejecutar las pruebas unitarias se ha utilizado el framework Unittest. Para su ejecución, se necesitan los siguientes pasos:
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

6. Ejecutar los test

    6.1. Acceder a la terminal
    ```bash
    docker exec -it <nombre_contenedor> /bin/bash   
    ```
    6.2. Acceder al directorio de testing
    ```bash
    cd test/  
    ```
    6.3. Ejecutar los test
    ```bash
    python -m unittest -v pruebas.py   
    ```