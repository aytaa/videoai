# CUDA ve cuDNN desteği olan bir imajı baz alıyoruz
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Gerekli sistem paketlerini kuruyoruz
RUN apt-get update && \
    apt-get install -y python3 python3-pip gcc build-essential ffmpeg libsm6 libxext6 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python bağımlılıklarını yükle
COPY requirements.txt ./
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Uygulamanın dinleyeceği port (varsa)
EXPOSE 3000

# Uygulamayı başlat
CMD ["python3", "main.py"]
