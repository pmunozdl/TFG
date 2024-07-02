# Usa una imagen base oficial de Python
FROM python:3.12.2

# Establece el directorio de trabajo en el contenedor
WORKDIR /TFG

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libhdf5-dev \
    gfortran \
    libatlas-base-dev \
    && apt-get clean

# Copia los archivos de requerimientos en el directorio de trabajo
COPY ./requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el contenido del repositorio en el directorio de trabajo
COPY . .

# Expon el puerto que Streamlit usa por defecto y Jupyter Notebook
EXPOSE 8501
EXPOSE 8888

# Define el comando para ejecutar Streamlit y Jupyter
CMD ["sh", "-c", "streamlit run streamlit.py & jupyter notebook --ip=0.0.0.0 --allow-root"]
