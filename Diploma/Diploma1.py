import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import Model
import pickle
import time
import os
import warnings
warnings.filterwarnings('ignore')  

print(" Energy Efficient Face Recognition System (Diploma Demo)")

base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
feature_model = Model(base_model.input, x)
print(" Model loaded: 3.5M params, 0.3 GFLOPs")

DB_FILE = "faces_database.pkl" 

def load_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_database(db):
    with open(DB_FILE, 'wb') as f:
        pickle.dump(db, f)

database = load_database()
print(f"📊 Database loaded: {len(database)} persons")

CASCADE_PATH = "haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_PATH):
    print(" Downloading cascade...")
    import urllib.request
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, CASCADE_PATH)
    print(" Cascade downloaded!")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("Cascade failed. Using DNN detector...")
    try:
        net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt",  
            "res10_300x300_ssd_iter_140000.caffemodel"
        )
        USE_DNN = True
    except:
        USE_DNN = False
else:
    USE_DNN = False
    print(" Haar Cascade loaded")

fps_counter = 0
start_time = time.time()
total_inference_time = 0

def recognize_face(embedding, threshold=0.6):
    best_name = None
    best_score = 0
    for name, known_embeddings in database.items():
        for known_emb in known_embeddings:
            score = np.dot(embedding, known_emb) / (np.linalg.norm(embedding) * np.linalg.norm(known_emb))
            if score > best_score and score > threshold:
                best_score = score
                best_name = name
    return best_name, best_score

print("Press 'r' = REGISTER, 'q' = QUIT")
print("oneDNN warnings = NORMAL")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Детекор лица
    if not USE_DNN:
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
    else:
        faces = []
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
    
    
    inference_start = time.time()
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        
        # Энергоэфф. признак
        face_resized = cv2.resize(face_roi, (224, 224))
        face_input = np.expand_dims(face_resized, 0).astype(np.float32)
        face_input = preprocess_input(face_input)
        
        embedding = feature_model.predict(face_input, verbose=0)[0]
        
        # Распознавание лица
        name, confidence = recognize_face(embedding)
        color = (0, 255, 0) if name else (0, 0, 255)
        label = f"{name or 'Unknown'}: {confidence:.1f}" if name else "Unknown"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    inference_time = time.time() - inference_start
    total_inference_time += inference_time
    
    #  
    fps_counter += 1
    if fps_counter % 30 == 0:
        fps = 30 / (time.time() - start_time)
        avg_inference_ms = total_inference_time / 30 * 1000
        print(f"📊 FPS: {fps:.1f} | Inference: {avg_inference_ms:.1f}ms | FLOPs: 0.3G")
        start_time = time.time()
        total_inference_time = 0
    
    cv2.imshow("EnergyEfficient Face Rec [r=Register]", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and len(faces) > 0:
        name = input("Имя для регистрации: ")
        face_roi = frame[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]
        embedding = feature_model.predict(preprocess_input(np.expand_dims(cv2.resize(face_roi, (224,224)), 0)), verbose=0)[0]
        
        if name not in database:
            database[name] = []
        database[name].append(embedding)
        save_database(database)
        print(f"✅ '{name}' зарегистрировано! Всего: {len(database)} человек")

cap.release()
cv2.destroyAllWindows()
print("ДЕМО")
print(f"📈 База: {len(database)} человек в 'faces_database.pkl'")
