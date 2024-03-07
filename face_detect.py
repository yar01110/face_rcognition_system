from ultralytics import YOLO


facemodel = YOLO('yolov8n-face.pt')


def detect(frame):
    face_result = facemodel.predict(frame, conf=0.30)
    return face_result
        

