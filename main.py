import cv2
import cvzone
from face_detect import detect 
from face_reco_cossim import recognize_face
while True:
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            break

        mainframe = frame.copy()

        face_result=detect(mainframe)
        for info in face_result:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                name=recognize_face(mainframe[y1:y2, x1:x2],0.6)
                h, w = y2 - y1, x2 - x1
                
                cvzone.cornerRect(mainframe, [x1, y1, w, h], l=9, rt=3)
                cvzone.putTextRect(mainframe, name, [x1, y1 - 10])
                cvzone.putTextRect(mainframe, 'Input Video', [0, 36])

        cv2.imshow('mainframe', mainframe)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()