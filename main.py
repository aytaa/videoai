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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    info = {"cpu": psutil.cpu_percent(interval=None), "ram": psutil.virtual_memory().percent,
            "swap": psutil.swap_memory().percent, "disk": psutil.disk_usage('/').percent}
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            info["gpu"] = []
            for gpu in gpus:
                info["gpu"].append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": f"{gpu.load * 100:.1f}%",
                    "free_memory": f"{gpu.memoryFree}MB",
                    "used_memory": f"{gpu.memoryUsed}MB",
                    "total_memory": f"{gpu.memoryTotal}MB",
                    "temperature": f"{gpu.temperature}°C",
                    "utilization": f"{gpu.utilization * 100:.1f}%"
                })
        else:
            info["gpu"] = "No GPU found"
    except Exception as e:
        info["gpu"] = f"Error fetching GPU info: {str(e)}"

    # Uptime, Tarih/Saat bilgisi
    uptime_seconds = time.time() - psutil.boot_time()
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    info["uptime"] = f"{hours} hrs {minutes} min"
    current_time = time.strftime("%d.%m.%Y %H:%M:%S")
    info["datetime"] = current_time

    return info


# -----------------------------
# YOLO model yükleme
# -----------------------------

model = YOLO("yolo11n.pt")
model.half().fuse().eval()
_ = model(torch.zeros(1, 3, 640, 640).half().to(device))

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

    # GPU Optimize Video Yakalama
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

    # Tracker'lar
    trackers = {cls: CentroidTracker(maxDisappeared=5) for cls in ALLOWED_CLASSES}
    trackable_objects = {cls: {} for cls in ALLOWED_CLASSES}

    # GPU Optimizasyonları
    frame_skip = 2
    batch_size = 4
    frame_buffer = []
    device = torch.device("cuda:0")

    # Model Warmup
    _ = model(torch.zeros((1, 3, 640, 640)).half().to(device))

    while True:
        try:
            # Batch boyutuna kadar frame topla
            while len(frame_buffer) < batch_size:
                ret, frame = cap.read()
                if ret:
                    frame_buffer.append(frame)
                else:
                    cap.release()
                    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                    continue

            # GPU'da Batch İşleme
            with torch.no_grad(), torch.cuda.amp.autocast():
                input_tensor = torch.stack([
                    torch.from_numpy(
                        cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (640, 640))
                    ).permute(2, 0, 1).half().to(device) / 255.0
                    for f in frame_buffer
                ])

                results = model.predict(input_tensor,
                                        conf=0.35,
                                        device=device,
                                        classes=[2, 3, 5, 7],  # DÜZELTME: Parantez eklendi
                                        verbose=False)

            # Sonuçları İşle
            for i, det in enumerate(results):
                frame = frame_buffer[i]
                centroids = {cls: [] for cls in ALLOWED_CLASSES}

                if det.boxes is not None:
                    boxes = det.boxes.xyxy.cpu().numpy()
                    classes = det.boxes.cls.cpu().numpy()

                    for box, cls_id in zip(boxes, classes):
                        class_name = model.names[int(cls_id)]
                        if class_name in ALLOWED_CLASSES:
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            centroids[class_name].append(((x1 + x2) // 2, (y1 + y2) // 2))

                # DÜZELTME: Parantez hatası giderildi
                for cls in ALLOWED_CLASSES:
                    current_centroids = np.array(centroids[cls]) if centroids[cls] else np.empty((0, 2))
                    objects = trackers[cls].update(current_centroids)

                    for obj_id, centroid in objects.items():
                        if obj_id not in trackable_objects[cls]:
                            globals()[f'total{cls.capitalize()}'] += 1
                            trackable_objects[cls][obj_id] = TrackableObject(obj_id, centroid)

                # Sayaç Görüntüleme
                y_pos = 30
                for cls in ALLOWED_CLASSES:
                    cv2.putText(frame,
                                f"{cls.capitalize()}: {globals()[f'total{cls.capitalize()}']}",
                                (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2)
                    y_pos += 30

                # Frame Kodlama
                _, buffer = cv2.imencode('.jpg', frame, [
                    cv2.IMWRITE_JPEG_QUALITY, 70,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1
                ])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            frame_buffer.clear()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Hata: {str(e)}")
            frame_buffer.clear()
            cap.release()
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
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
    torch.backends.cudnn.benchmark = True
    app.run(debug=True, port=3000, host='0.0.0.0')
