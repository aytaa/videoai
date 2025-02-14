from flask import Flask, Response, render_template, session, flash, jsonify, request, redirect, url_for
import cv2
import numpy as np
from scipy.spatial import distance as dist

# -----------------------------
# PyTorch & YOLO ayarları
# -----------------------------
import torch
import torch.serialization
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
torch.serialization.add_safe_globals([DetectionModel, Sequential])
from ultralytics import YOLO

# -----------------------------
# Ek kütüphaneler: netifaces, psutil
# -----------------------------
import netifaces
import psutil
import time

app = Flask(__name__)
app.secret_key = "e7af6fde-a043-4ccb-bc11-56b5c2e78962"

# -----------------------------
# Global sayaçlar
# -----------------------------
totalCars = 0
totalMotorcycles = 0
stream_url = "https://hls.ibb.gov.tr/tkm4/hls/102.stream/chunklist.m3u8"
ALLOWED_CLASSES = ['car', 'motorcycle']

# -----------------------------
# Fonksiyon: Ağ (Network) bilgisi
# -----------------------------
def get_network_info(interface="eth0"):
    info = {
        "ethernet": "N/A",
        "gateway": "N/A",
        "netmask": "N/A",
        "gsm": "Not Found"
    }
    # Varsayılan gateway
    try:
        gateways = netifaces.gateways()
        default_gateway = gateways.get('default')
        if default_gateway:
            gw_addr = default_gateway.get(netifaces.AF_INET)
            if gw_addr:
                info["gateway"] = gw_addr[0]
    except:
        pass

    # IP / Netmask
    try:
        if interface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(interface).get(netifaces.AF_INET)
            if addrs and len(addrs) > 0:
                info["ethernet"] = addrs[0].get('addr', 'N/A')
                info["netmask"] = addrs[0].get('netmask', 'N/A')
    except:
        pass

    return info

# -----------------------------
# Fonksiyon: Sistem (System) bilgisi
# -----------------------------
def get_system_info():
    info = {}
    # CPU, RAM, SWAP, DISK
    info["cpu"] = psutil.cpu_percent(interval=None)
    info["ram"] = psutil.virtual_memory().percent
    info["swap"] = psutil.swap_memory().percent
    info["disk"] = psutil.disk_usage('/').percent

    if hasattr(psutil, "sensors_battery"):
        battery = psutil.sensors_battery()
        if battery is not None:
            info["battery"] = f"{battery.percent}%"
        else:
            info["battery"] = "Not Found"
    else:
        info["battery"] = "Not Found"

    # Uptime
    uptime_seconds = time.time() - psutil.boot_time()
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    info["uptime"] = f"{hours} hrs {minutes} min"

    # Tarih/Saat
    current_time = time.strftime("%d.%m.%Y %H:%M:%S")
    info["datetime"] = current_time

    return info

# -----------------------------
# YOLO model yükleme
# -----------------------------
model = YOLO("yolo11n.pt")  # YOLO11N modeli kullanılıyor

# -----------------------------
# CentroidTracker
# -----------------------------
class CentroidTracker:
    def __init__(self, maxDisappeared=40):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, inputCentroids):
        if len(inputCentroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        if len(self.objects) == 0:
            for centroid in inputCentroids:
                self.register(centroid)
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects

# -----------------------------
# TrackableObject
# -----------------------------
class TrackableObject:
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]
        self.counted = False

# -----------------------------
# Video frame generator (YOLO)
# -----------------------------
def gen_frames():
    global totalCars, totalMotorcycles, stream_url

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Video akışı açılamadı! Yeniden bağlanıyor...")
        cap.release()
        cap = cv2.VideoCapture(stream_url)

    car_tracker = CentroidTracker(maxDisappeared=40)
    motorcycle_tracker = CentroidTracker(maxDisappeared=40)
    trackableCars = {}
    trackableMotorcycles = {}

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Video akışı kesildi, tekrar bağlanılıyor...")
            cap.release()
            cap = cv2.VideoCapture(stream_url)
            continue

        try:
            # 1) YOLO prediction with lower confidence threshold
            results = model.predict(frame, conf=0.3, verbose=False)
            car_centroids = []
            motorcycle_centroids = []

            # 2) Bounding box çizmek ve centroid listesi oluşturmak
            if len(results) > 0:
                det = results[0]
                if det.boxes is not None:
                    for box in det.boxes.data.cpu().numpy():
                        x1, y1, x2, y2, conf, cls_id = box
                        class_name = model.names[int(cls_id)]
                        if class_name in ALLOWED_CLASSES:
                            # Dikdörtgen çiz (yeşil)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                            # Label yaz (class + confidence)
                            cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # Centroid hesapla
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)

                            if class_name == 'car':
                                car_centroids.append((cx, cy))
                            elif class_name == 'motorcycle':
                                motorcycle_centroids.append((cx, cy))

            # 3) Centroid Tracker update
            car_objects = car_tracker.update(np.array(car_centroids) if car_centroids else np.empty((0, 2)))
            motorcycle_objects = motorcycle_tracker.update(np.array(motorcycle_centroids) if motorcycle_centroids else np.empty((0, 2)))

            # 4) Her bir Araba objesi
            for (objectID, centroid) in car_objects.items():
                to = trackableCars.get(objectID, None)
                if to is None:
                    to = TrackableObject(objectID, centroid)
                    totalCars += 1  # Yeni araba tespit edildi, sayacı artır
                else:
                    to.centroids.append(centroid)

                trackableCars[objectID] = to

                # ID yaz (kırmızı)
                cv2.putText(frame, f"Car ID {objectID}", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

            # 5) Sayaçları ekrana yaz
            cv2.putText(frame, f"Total Cars: {totalCars}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Total Motorcycles: {totalMotorcycles}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 6) Görüntüyü MJPEG olarak gönder
            ret2, buffer = cv2.imencode('.jpg', frame)
            if not ret2:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"[Hata]: {str(e)}")
            break

    cap.release()

# -----------------------------
# Rotalar
# -----------------------------
@app.route('/')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    # İlk açılışta gösterilecek veriler
    network_data = get_network_info(interface="eth0")
    system_data = get_system_info()

    return render_template("home.html", network=network_data, system=system_data)

@app.route('/live')
def live():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template("live.html", stream_url=stream_url)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'adem' and password == 'Adem123456':
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('home'))
        else:
            flash('Yanlış kullanıcı adı veya şifre!')
            return redirect(url_for('login'))

    return render_template("login.html")

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/set_stream', methods=['POST'])
def set_stream():
    global stream_url, totalCars, totalMotorcycles
    url = request.form.get('stream_url')
    if url:
        stream_url = url.strip()
        totalCars = 0
        totalMotorcycles = 0
    return redirect(url_for('live'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset_counts', methods=['POST'])
def reset_counts():
    global totalCars, totalMotorcycles
    totalCars = 0
    totalMotorcycles = 0
    return redirect(url_for('live'))

@app.route('/counts')
def counts():
    return jsonify({
        "car": totalCars,
        "motorcycle": totalMotorcycles
    })

# -----------------------------
# Yeni Rota: Gerçek zamanlı sistem bilgisi JSON
# -----------------------------
@app.route('/system_info')
def system_info():
    info = get_system_info()
    return jsonify(info)

# -----------------------------
# Uygulama çalıştırma
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True, port=3000, host='0.0.0.0')