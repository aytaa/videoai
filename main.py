from flask import Flask, Response, render_template, session, flash, jsonify, request, redirect, url_for
import cv2
import numpy as np
from scipy.spatial import distance as dist
from db_connector import get_connection

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
totalTrucks = 0
totalBus = 0
stream_url = "https://hls.ibb.gov.tr/tkm4/hls/102.stream/chunklist.m3u8"

ALLOWED_CLASSES = ['car', 'motorcycle', 'truck', 'bus']


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
    global totalCars, totalMotorcycles, totalTrucks, totalBus, stream_url

    # Video yakalama için daha uzun timeout
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Buffer size'ı ayarla

    if not cap.isOpened():
        print("Video akışı açılamadı! Yeniden bağlanıyor...")
        time.sleep(1)  # Yeniden bağlanma öncesi bekle
        cap.release()
        cap = cv2.VideoCapture(stream_url)

    # Tracker'lar
    trackers = {
        'car': CentroidTracker(maxDisappeared=5),
        'motorcycle': CentroidTracker(maxDisappeared=5),
        'truck': CentroidTracker(maxDisappeared=5),
        'bus': CentroidTracker(maxDisappeared=5)
    }

    trackable_objects = {
        'car': {},
        'motorcycle': {},
        'truck': {},
        'bus': {}
    }

    # Frame işleme için sayaç
    frame_count = 0
    skip_frames = 2

    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Video akışı kesildi, tekrar bağlanılıyor...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                continue

            frame_count += 1
            process_this_frame = frame_count % skip_frames == 0

            if process_this_frame:
                # YOLO tahminini yap
                with torch.no_grad():  # Gradient hesaplamasını devre dışı bırak
                    results = model.predict(frame, conf=0.35, verbose=False)

                centroids = {class_name: [] for class_name in ALLOWED_CLASSES}

                if len(results) > 0:
                    det = results[0]
                    if det.boxes is not None:
                        boxes = det.boxes.data.cpu().numpy()
                        for box in boxes:
                            x1, y1, x2, y2, conf, cls_id = box
                            class_name = model.names[int(cls_id)]

                            if class_name in ALLOWED_CLASSES:
                                # Bounding box ve etiket çizimi
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.putText(frame, f"{class_name}", (int(x1), int(y1) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                                # Centroid hesaplama
                                cx = int((x1 + x2) / 2)
                                cy = int((y1 + y2) / 2)
                                centroids[class_name].append((cx, cy))

                # Her sınıf için tracker güncelleme ve nesne sayma
                for class_name in ALLOWED_CLASSES:
                    current_centroids = np.array(centroids[class_name]) if centroids[class_name] else np.empty((0, 2))
                    objects = trackers[class_name].update(current_centroids)

                    for (objectID, centroid) in objects.items():
                        to = trackable_objects[class_name].get(objectID, None)

                        if to is None:
                            to = TrackableObject(objectID, centroid)
                            if class_name == 'car':
                                totalCars += 1
                            elif class_name == 'motorcycle':
                                totalMotorcycles += 1
                            elif class_name == 'truck':
                                totalTrucks += 1
                            elif class_name == 'bus':
                                totalBus += 1
                        else:
                            to.centroids.append(centroid)
                        trackable_objects[class_name][objectID] = to

            # Sayaçları göster
            cv2.putText(frame, f"Total Cars: {totalCars}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Total Motorcycles: {totalMotorcycles}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Total Trucks: {totalTrucks}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Total Buses: {totalBus}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            # Frame'i JPEG formatında kodla ve gönder
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"[Hata]: {str(e)}")
            time.sleep(0.1)  # Hata durumunda kısa bir bekleme
            continue

    cap.release()


# -----------------------------
# Rotalar
# -----------------------------
@app.route('/')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    network_data = get_network_info(interface="eth0")
    system_data = get_system_info()
    return render_template("home.html", network=network_data, system=system_data, active_page="home")


@app.route('/live')
def live():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    query = "SELECT * FROM cameras WHERE m3u8_url IS NOT NULL AND m3u8_url <> ''"
    cursor.execute(query)
    cameras = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template("live.html", stream_url=stream_url, cameras=cameras, active_page="live")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM users WHERE username = %s"
        cursor.execute(query, (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user:
            if user['password'] == password:
                session['logged_in'] = True
                session['username'] = username
                return redirect(url_for('home'))
            else:
                flash('Yanlış şifre!')
        else:
            flash('Kullanıcı bulunamadı!')
        return redirect(url_for('login'))
    return render_template("login.html")


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/set_stream', methods=['POST'])
def set_stream():
    global stream_url, totalCars, totalMotorcycles, totalTrucks, totalBus
    url = request.form.get('stream_url')
    if url:
        stream_url = url.strip()
        totalCars = 0
        totalMotorcycles = 0
        totalTrucks = 0
        totalBus = 0
    return redirect(url_for('live'))


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/reset_counts', methods=['POST'])
def reset_counts():
    global totalCars, totalMotorcycles, totalTrucks, totalBus
    totalCars = 0
    totalMotorcycles = 0
    totalTrucks = 0
    totalBus = 0
    return redirect(url_for('live'))


@app.route('/counts')
def counts():
    return jsonify({
        "car": totalCars,
        "motorcycle": totalMotorcycles,
        "totalTrucks": totalTrucks,
        "totalBus": totalBus,
    })


@app.route('/settings/cameras')
def settings_cameras():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM cameras")
    cameras = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template("settings_cameras.html", cameras=cameras, active_page="settings_cameras")


@app.route('/settings/cameras/new', methods=['GET', 'POST'])
def new_camera():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        city = request.form.get('city')
        district = request.form.get('district')
        intersection = request.form.get('intersection')
        address = request.form.get('address')
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        ip_address = request.form.get('ip_address')
        m3u8_url = request.form.get('m3u8_url')
        installation_date = request.form.get('installation_date')
        camera_type = request.form.get('camera_type')
        status = request.form.get('status')

        conn = get_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO cameras 
            (name, description, city, district, intersection, address, latitude, longitude, ip_address, m3u8_url, installation_date, camera_type, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            name, description, city, district, intersection, address,
            latitude, longitude, ip_address, m3u8_url, installation_date,
            camera_type, status
        ))
        conn.commit()
        cursor.close()
        conn.close()

        flash("Kamera başarıyla eklendi.")
        return redirect(url_for('settings_cameras'))

    return render_template("settings_cameras_new.html")


@app.route('/settings/user')
def settings_users():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template("settings_users.html", users=users, active_page="settings_users")


@app.route('/settings/user/new', methods=['GET', 'POST'])
def new_user():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = 'admin'

        conn = get_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO users 
            (username, password, role)
            VALUES (%s, %s, %s)
        """
        cursor.execute(query, (
            username, password, role
        ))
        conn.commit()
        cursor.close()
        conn.close()

        flash("Kullanıcı başarıyla eklendi.")
        return redirect(url_for('settings_users'))

    return render_template("settings_users_new.html")


# -----------------------------
# Yeni Rota: Gerçek zamanlı sistem bilgisi JSON
# -----------------------------

@app.route('/system_info')
def system_info():
    info = get_system_info()
    return jsonify(info)


@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404

# -----------------------------
# Uygulama çalıştırma
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True, port=3000, host='0.0.0.0')
