!pip install opencv-python ultralytics torch numpy
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity

# Load YOLO model
model = YOLO("best.pt")  
features_db = []
id_counter = 0
track_memory = {}

def extract_feature(crop):
    
    return cv2.resize(crop, (32, 64)).flatten()

def assign_id(new_feature):
    global id_counter
    if not features_db:
        features_db.append((id_counter, new_feature))
        return id_counter

    similarities = [cosine_similarity([new_feature], [f[1]])[0][0] for f in features_db]
    max_sim_idx = int(np.argmax(similarities))
    max_sim_val = similarities[max_sim_idx]
    print(max_sim_val)

    if max_sim_val > 0.9:  # similarity threshold
        return features_db[max_sim_idx][0]
    else:
        id_counter += 1
        features_db.append((id_counter, new_feature))
        return id_counter
cap = cv2.VideoCapture("/content/15sec_input_720p.mp4")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
out = cv2.VideoWriter('output_video.mp4', fourcc, fps/6, (width, height)) 

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    results = model(frame)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            feature = cv2.resize(crop, (32, 64)).flatten()
            player_id = assign_id(feature)
          
           
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f'ID: {player_id}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)


    out.write(frame)

cap.release()
out.release()
