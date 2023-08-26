import cv2
#opencv.DNN
net=cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model=cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320),scale=1/255)
#load class lists
classes= []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
       class_name=class_name.strip()
       classes.append(class_name)
print("object list")
print(classes[0])


#intialize camera
cap =cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
def click_button(event,x,y,flags,parms):
    if event ==cv2.EVENT_LBUTTONDOWN:
        print(x,y)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame",click_button)

while True:
    # get frames
    ret, frame = cap.read()
    #object detection
    (class_ids,scores,bboxes)=model.detect(frame)
    #  zip is function to extract muliple arrays at a time
    for class_id, score, bbox in zip(class_ids,scores,bboxes):
        (x,y,w,h)=bbox
        class_name=classes[class_id]
        cv2.putText(frame,class_name,(x,y-5),cv2.FONT_HERSHEY_PLAIN,2,(200,0,0),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(200,0,0),1)






    #print("class ids",class_ids)
    #print("Score", scores)
    #print("bboxes", bboxes)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)# streams the frame for every milli second




