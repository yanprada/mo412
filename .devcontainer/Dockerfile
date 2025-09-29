# Usa una imagen base más ligera y específica para Python.
# python:3.11-slim-bookworm es ideal ya que es más pequeña que Ubuntu y ya incluye Python.
FROM python:3.12-slim-bookworm

# Configura la zona horaria en una sola línea.
ENV TZ=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Instala las dependencias del sistema en una sola capa para reducir el tamaño de la imagen.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Copia los archivos de configuración y el código fuente.
# Esto optimiza el caché de Docker: si los archivos clave no cambian, no se reconstruyen.
COPY pyproject.toml .
COPY src ./src

# Instala el proyecto y sus dependencias de desarrollo.
# El punto "." le dice a pip que instale el paquete localmente.
# El flag "[dev]" instala el grupo de dependencias opcional de desarrollo.
RUN pip install --no-cache-dir .[dev]

# Configura el directorio de trabajo y copia el resto del proyecto.
WORKDIR /app
COPY . .

# Establece la variable de entorno para que el módulo 'src' esté en la ruta de importación.
ENV PYTHONPATH=/app/src

# Expone el puerto por defecto para Jupyter Notebooks.
EXPOSE 8888

# Comando por defecto para iniciar el contenedor
CMD ["/bin/bash"]