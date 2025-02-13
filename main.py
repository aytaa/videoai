from flask import Flask, Response, render_template, session, flash, jsonify, request, redirect, url_for
import cv2
import numpy as np
from scipy.spatial import distance as dist

# PyTorch ve "safe load" ayarları
import torch
import torch.serialization

# Ultralytics içinde kullanılan sınıfları import edin
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential

# Bu sınıfları PyTorch’un güvenli yükleme (allowlist) listesine ekleyin
torch.serialization.add_safe_globals([DetectionModel, Sequential])

from ultralytics import YOLO

# Modeli yükle
model = YOLO("yolo11n.pt")

# ------------------- Centroid Tracker -------------------
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


# ------------------ TrackableObject ------------------
class TrackableObject:
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]
        self.counted = False


# ------------------ Flask Setup ------------------
app = Flask(__name__)
app.secret_key = "e7af6fde-a043-4ccb-bc11-56b5c2e78962"

# Global sayaçlar
totalCars = 0
totalMotorcycles = 0

# Varsayılan HLS akışı (İBB örnek)
stream_url = "https://hls.ibb.gov.tr/tkm4/hls/102.stream/chunklist.m3u8"

# Yalnızca bu sınıfları tespit & say
ALLOWED_CLASSES = ['car', 'motorcycle']


def gen_frames():
    global totalCars, totalMotorcycles, stream_url

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Video akışı açılamadı!")
        return

    # Centroid tracker
    car_tracker = CentroidTracker(maxDisappeared=40)
    motorcycle_tracker = CentroidTracker(maxDisappeared=40)
    trackableCars = {}
    trackableMotorcycles = {}

    ret, frame = cap.read()
    if not ret:
        print("İlk frame alınamadı!")
        cap.release()
        return

    frameHeight, frameWidth = frame.shape[:2]
    counting_line = int(frameHeight * 0.8)

    while True:
        try:
            results = model.predict(frame, conf=0.5, verbose=False)
            car_centroids = []
            motorcycle_centroids = []

            if len(results) > 0:
                det = results[0]
                if det.boxes is not None:
                    for box in det.boxes.data.cpu().numpy():
                        x1, y1, x2, y2, conf, cls_id = box
                        if conf < 0.5:
                            continue
                        class_name = model.names[int(cls_id)]
                        if class_name in ALLOWED_CLASSES:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            if class_name == 'car':
                                car_centroids.append((cx, cy))
                            elif class_name == 'motorcycle':
                                motorcycle_centroids.append((cx, cy))

            # Tracker güncelle
            car_objects = car_tracker.update(
                np.array(car_centroids) if car_centroids else np.empty((0, 2))
            )
            motorcycle_objects = motorcycle_tracker.update(
                np.array(motorcycle_centroids) if motorcycle_centroids else np.empty((0, 2))
            )

            # Araba sayımı
            for (objectID, centroid) in car_objects.items():
                to = trackableCars.get(objectID, None)
                if to is None:
                    to = TrackableObject(objectID, centroid)
                else:
                    to.centroids.append(centroid)
                    if not to.counted:
                        # Basit mantık: centroid Y > counting_line => say
                        y_coords = [c[1] for c in to.centroids]
                        direction = centroid[1] - np.mean(y_coords)
                        if direction > 0 and centroid[1] > counting_line:
                            totalCars += 1
                            to.counted = True
                trackableCars[objectID] = to

                # Debug çizim
                cv2.putText(frame, f"Car ID {objectID}", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

            # Motorsiklet sayımı
            for (objectID, centroid) in motorcycle_objects.items():
                to = trackableMotorcycles.get(objectID, None)
                if to is None:
                    to = TrackableObject(objectID, centroid)
                else:
                    to.centroids.append(centroid)
                    if not to.counted:
                        y_coords = [c[1] for c in to.centroids]
                        direction = centroid[1] - np.mean(y_coords)
                        if direction > 0 and centroid[1] > counting_line:
                            totalMotorcycles += 1
                            to.counted = True
                trackableMotorcycles[objectID] = to

                cv2.putText(frame, f"Motor ID {objectID}", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)

            # Sayım çizgisi & sayaç
            cv2.line(frame, (0, counting_line), (frameWidth, counting_line), (0, 255, 255), 2)
            cv2.putText(frame, f"Araba: {totalCars}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Motor: {totalMotorcycles}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # MJPEG çıkışı
            ret2, buffer = cv2.imencode('.jpg', frame)
            if not ret2:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Sonraki kare
            ret, frame = cap.read()
            if not ret:
                print("Akış koptu veya bitti; yeniden başlatılıyor...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
        except Exception as e:
            print(f"[Hata]: {str(e)}")
            break

    cap.release()

@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template("index.html", stream_url=stream_url)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'adem' and password == 'Adem123456':
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
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
        print("Yeni stream URL ayarlandı:", stream_url)
    return redirect(url_for('index'))


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/reset_counts', methods=['POST'])
def reset_counts():
    global totalCars, totalMotorcycles
    totalCars = 0
    totalMotorcycles = 0
    print("Sayımlar sıfırlandı!")
    return redirect(url_for('index'))


@app.route('/counts')
def counts():
    return jsonify({"car": totalCars, "motorcycle": totalMotorcycles})


if __name__ == '__main__':
    app.run(debug=True, port=3000, host='0.0.0.0')
