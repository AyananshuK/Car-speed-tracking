import cv2
from ultralytics import YOLO
import cvzone
import math
from sort import *  #https://github.com/abewley/sort
import numpy as np
import time


cap = cv2.VideoCapture("car highway.mp4")

model = YOLO("yolo weights/yolov8l.pt").to('cuda')

classnames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
"handbag","tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
"baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
"fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
"carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
"diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
"teddy bear", "hair drier", "toothbrush"
]

tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

up_enter_line = [475, 269, 1000, 269]
up_leave_line = [475, 200, 750, 200]
down_enter_line = [130, 200, 455, 200]
down_leave_line = [0, 269, 445, 269]

upCars={}
downCars={}
sId={}
upOverSpeed={}
downOverSpeed={}


while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            
            w,h = x2-x1, y2-y1

            #Confidence
            conf = math.ceil(box.conf[0]*100)/100

            #Class name
            cls = int(box.cls[0])

            currentClass = classnames[cls]

            if currentClass in ["car", "bus", "truck", "motorbike", "bicycle"]:
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))


    cv2.line(img, (up_enter_line[0], up_enter_line[1]), (up_enter_line[2], up_enter_line[3]), (91, 200, 237), 5)
    cv2.line(img, (up_leave_line[0], up_leave_line[1]), (up_leave_line[2], up_leave_line[3]), (255,0,0), 5)
    cv2.line(img, (down_enter_line[0], down_enter_line[1]), (down_enter_line[2], down_enter_line[3]), (91, 200, 237), 5)
    cv2.line(img, (down_leave_line[0], down_leave_line[1]), (down_leave_line[2], down_leave_line[3]), (255,0,0), 5)


    trackerResults = tracker.update(detections)
    for res in trackerResults:
        x1,y1,x2,y2,id = res
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        w,h = x2-x1, y2-y1
        centre_x = (x1+x2)//2
        centre_y = (y1+y2)//2
     
        #Up
        if up_enter_line[0] < centre_x < up_enter_line[2] and up_enter_line[1]-10 < centre_y < up_enter_line[3]+10:
            if id not in upCars:
                upCars[id]=time.time()
                cv2.line(img, (up_enter_line[0], up_enter_line[1]), (up_enter_line[2], up_enter_line[3]), (0, 255, 0), 5)
        
        if id in upCars:
            if up_leave_line[0] < centre_x < up_leave_line[2] and up_leave_line[1]-10 < centre_y < up_leave_line[3]+10:
                if id not in sId:
                    elapsed_time = time.time() - upCars[id]
                    d = 36 #meters
                    s = d//elapsed_time  
                    s = s * 3.6 #km/h
                    sId[id] = s
               
            if up_enter_line[0]-10 < centre_x < up_enter_line[2]+10 and up_enter_line[1]+10 > centre_y > up_leave_line[1]-60:
                cv2.circle(img, (centre_x, centre_y), 5, (0,0,255), cv2.FILLED)
                cvzone.cornerRect(img, (x1,y1,w,h), l=9, rt=2, colorR=(255,255,255))
                if id in sId:
                    if sId[id]>60:
                        cvzone.putTextRect(img, f"Id:{int(id)} S:{int(sId[id])}km/h", (max(0,x1), max(35,y1)), scale=1, thickness=1, offset=6, colorR=(0,0,255), colorT=(255,255,255))
                        upOverSpeed[id]=sId[id]
                    else:
                        cvzone.putTextRect(img, f"Id:{int(id)} S:{int(sId[id])}km/h", (max(0,x1), max(35,y1)), scale=1, thickness=1, offset=6, colorR=(255,255,255), colorT=(0,0,0))
                else:
                    cvzone.putTextRect(img, f"Id:{int(id)}", (max(0,x1), max(35,y1)), scale=1, thickness=1, offset=6, colorR=(255,255,255), colorT=(0,0,0))


        #Down
        if down_enter_line[0] < centre_x < down_enter_line[2] and down_enter_line[1]-10 < centre_y < down_enter_line[3]+10:
            if id not in downCars:
                downCars[id]=time.time()  
                cv2.line(img, (down_enter_line[0], down_enter_line[1]), (down_enter_line[2], down_enter_line[3]), (0, 255, 0), 5)

        if id in downCars:
            if down_leave_line[0] < centre_x < down_leave_line[2] and down_leave_line[1]-10 < centre_y < down_leave_line[3]+10:
                if id not in sId:
                    elapsed_time = time.time() - downCars[id]
                    d = 36 #meters
                    s = d//elapsed_time  
                    s = s * 3.6 #km/h
                    sId[id] = s  
        
            if down_enter_line[0]-10 < centre_x < down_enter_line[2]+10 and down_enter_line[1]-10 < centre_y < down_leave_line[1]+90:
                cv2.circle(img, (centre_x, centre_y), 5, (0,0,255), cv2.FILLED)
                cvzone.cornerRect(img, (x1,y1,w,h), l=9, rt=2, colorR=(255,255,255))
                if id in sId:
                    if sId[id]>60:
                        cvzone.putTextRect(img, f"Id:{int(id)} S:{int(sId[id])}km/h", (max(0,x1), max(15,y1)), scale=1, thickness=1, offset=6, colorR=(0,0,255), colorT=(255,255,255))
                        downOverSpeed[id]=sId[id]
                    else:
                        cvzone.putTextRect(img, f"Id:{int(id)} S:{int(sId[id])}km/h", (max(0,x1), max(15,y1)), scale=1, thickness=1, offset=6, colorR=(255,255,255), colorT=(0,0,0))
                else:
                    cvzone.putTextRect(img, f"Id:{int(id)}", (max(0,x1), max(35,y1)), scale=1, thickness=1, offset=6, colorR=(255,255,255), colorT=(0,0,0))

    
    
    cvzone.putTextRect(img, f"Speed limit: 60km/h", (380,50), 1.5, 2, colorR=(255,255,255), colorT=(0,0,255))

    cvzone.putTextRect(img, f"Up count: {len(upCars)}", (720,50), 2, 2, colorR=(255,255,255), colorT=(0,0,0))
    cvzone.putTextRect(img, f"Down count: {len(downCars)}", (36,50), 2, 2, colorR=(255,255,255), colorT=(0,0,0))

    cvzone.putTextRect(img, f"Up overspeed: {len(upOverSpeed)}", (720,100), 1.5, 2, colorR=(255,255,255), colorT=(0,0,255))
    cvzone.putTextRect(img, f"Down overspeed: {len(downOverSpeed)}", (36,100), 1.5, 2, colorR=(255,255,255), colorT=(0,0,255))
 
    cv2.imshow("Image", img) 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()