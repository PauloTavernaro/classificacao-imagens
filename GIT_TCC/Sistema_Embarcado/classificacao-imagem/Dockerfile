ARG BASE_NAME=debian
ARG IMAGE_ARCH=linux/arm64/v8
ARG IMAGE_TAG=3-bookworm
ARG DOCKER_REGISTRY=torizon

#Criar a etapa de build do contêiner baseado na imagem fornecida pela Toradex

FROM --platform=$IMAGE_ARCH $DOCKER_REGISTRY/$BASE_NAME:$IMAGE_TAG AS build

# Instalar o QEMU para converter o binário do modelo de machine learning para a arquitetura amr64
COPY --from=multiarch/qemu-user-static /usr/bin/qemu-aarch64-static /usr/bin

## Instalar build tools
RUN apt-get -y update && apt-get install -y \
    cmake build-essential gcc g++ git wget unzip patchelf \
    autoconf automake libtool curl gfortran \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
    
# Definir as variáveis de ambiente para a compilação cruzada
ENV CC=aarch64-linux-gnu-gcc \
    CXX=aarch64-linux-gnu-g++
    
# Copiar o código fonte do binário do modelo
COPY ./src/Modelo_ML /src/Modelo_ML_comp

# Compilar o binário do modelo
WORKDIR /src

RUN make Modelo_ML_comp

# Criar uma nova etapa para a imagem final
FROM --platform=$IMAGE_ARCH $DOCKER_REGISTRY/qt5-wayland-vivante:3

# Copiar o binário compilado do modelo para a imagem final
COPY --from=build /src/Modelo_ML_comp /usr/local/bin/

# Copiar foto a ser classificada
COPY ./src/fotos/Foto_Original.jpg /usr/local/bin/fotos/

# Copiar o código Python da aplicação
COPY . /app

## Instalar Python
RUN apt-get -y update && apt-get install -y \
  python3 python3-dev python3-numpy python3-pybind11 \
  python3-pip python3-setuptools python3-wheel python3.11-venv\
  && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*
  

# Defiir a variável de ambiente para usar o Wayland
ENV QT_QPA_PLATFORM=wayland


## Iniciar programa
WORKDIR /app

## Instalar dependências com apt-get
RUN apt-get -y update && apt-get install -y \
    zlib1g zlib1g-dev libssl-dev \
    imx-gpu-viv-wayland-dev openssl libffi-dev libjpeg-dev \
    python3-pyside2.qtcore \
    python3-pyside2.qtquick \
    python3-pyside2.qtwidgets \
    python3-pyside2.qtqml \
    python3-pyside2.qtnetwork \
    python3-pyside2.qtgui \
    git x11vnc \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

# Criar o ambiente virtual para utilizar o pip
RUN python3 -m venv /opt/venv --system-site-packages

# Copiar o arquivo requirements.txt para o diretório de trabalho no contêiner
COPY requirements.txt .

## Instalar dependências com pip
RUN . /opt/venv/bin/activate && pip install wheel && pip install -r requirements.txt --use-pep517
    
## Remover libqt5gui5-gles, substituir por libqt5gui5 
RUN apt-get remove libqt5opengl5 --allow-change-held-packages \
    && apt-get remove libqt5gui5-gles

## Instalar libqt5gui5 and reinstalar libqt5opengl5
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    libqt5opengl5 libqt5gui5 libqt5printsupport5 libqt5quick5 libqt5quickparticles5 \
    libqt5quickshapes5 libqt5quicktest5 libqt5quickwidgets5 libqt5waylandclient5 \
    libqt5waylandcompositor5 libqt5widgets5 qml-module-qtquick-layouts \
    qml-module-qtquick-particles2 qml-module-qtquick-shapes \
    qml-module-qtquick-window2 qml-module-qtquick2 \
    qml-module-qttest qtwayland5 \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*
    
## Expor porta 5900 para visualizar a tela com VNC Viewer
EXPOSE 5900

# Executar script
CMD . /opt/venv/bin/activate && exec python Usar_Modelo.py
