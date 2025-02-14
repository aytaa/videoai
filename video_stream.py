# video_stream.py
import cv2
import numpy as np
import time
from scipy.spatial import distance as dist
from ultralytics import YOLO

# -----------------------------
# Global Ayarlar ve Değişkenler
# -----------------------------
ALLOWED_CLASSES = ['car', 'motorcycle']
stream_url = "https://hls.ibb.gov.tr/tkm4/hls/102.stream/chunklist.m3u8"

# Global sayaçlar (isteğe bağlı: daha sonra dışarıdan erişilebilir yapmak için sınıf veya fonksiyonla sarmalanabilir)
totalCars = 0
totalMotorcycles = 0

# -----------------------------
# CentroidTracker Sınıfı
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
# TrackableObject Sınıfı
# -----------------------------
class TrackableObject:
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]
        self.counted = False

# -----------------------------
# YOLO Model Yükleme
# -----------------------------
model = YOLO("yolo11n.pt")  # Model dosya yolunu ihtiyacınıza göre ayarlayın

# -----------------------------
# Video Frame Generator (YOLO ile)
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
            # 1) YOLO prediction
            results = model.predict(frame, conf=0.3, verbose=False)
            car_centroids = []
            motorcycle_centroids = []

            if len(results) > 0:
                det = results[0]
                if det.boxes is not None:
                    for box in det.boxes.data.cpu().numpy():
                        x1, y1, x2, y2, conf, cls_id = box
                        class_name = model.names[int(cls_id)]
                        if class_name in ALLOWED_CLASSES:
                            # Dikdörtgen çizme ve label ekleme
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # Centroid hesaplama
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            if class_name == 'car':
                                car_centroids.append((cx, cy))
                            elif class_name == 'motorcycle':
                                motorcycle_centroids.append((cx, cy))

            # 2) Centroid Tracker güncelleme
            car_objects = car_tracker.update(np.array(car_centroids) if car_centroids else np.empty((0, 2)))
            motorcycle_objects = motorcycle_tracker.update(np.array(motorcycle_centroids) if motorcycle_centroids else np.empty((0, 2)))

            # 3) Her bir araba için
            for (objectID, centroid) in car_objects.items():
                to = trackableCars.get(objectID, None)
                if to is None:
                    to = TrackableObject(objectID, centroid)
                    totalCars += 1  # Yeni araba tespit edildi
                else:
                    to.centroids.append(centroid)

                trackableCars[objectID] = to
                cv2.putText(frame, f"Car ID {objectID}", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

            # 4) Sayaçları ekrana yazdırma
            cv2.putText(frame, f"Total Cars: {totalCars}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Total Motorcycles: {totalMotorcycles}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 5) Frame'i MJPEG formatında kodlama
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
