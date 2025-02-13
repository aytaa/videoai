FROM python:3.9-slim

# Build argümanı: Cache'i geçersiz kılmak için kullanılabilir
ARG CACHE_DATE=2025-02-12
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Paket listelerini güncelle, gerekli paketleri kur ve apt cache'ini temizle
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/* && \
    echo $CACHE_DATE

# Gereksinimler dosyasını kopyala ve bağımlılıkları yükle
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN apt-get install -y ffmpeg libsm6 libxext6 libgl1-mesa-glx && \
rm -rf /var/lib/apt/lists/* && \
echo $CACHE_DATE

# Tüm uygulama dosyalarını kopyala
COPY . .

# Uygulamanın dinleyeceği portu belirt
EXPOSE 3000

# Uygulamayı başlat (main.py içinde app tanımlı)
CMD ["python", "main.py"]
