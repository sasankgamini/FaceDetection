import cv2
facecascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyecascade=cv2.CascadeClassifier('haarcascade_eye.xml')
smilecascade=cv2.CascadeClassifier('haarcascade_smile.xml')
cap=cv2.VideoCapture(0)
while cap.isOpened():
    retval,img=cap.read()
    grayscale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=facecascade.detectMultiScale(img,1.1,5)
    for (fx,fy,fw,fh) in face:
        cv2.rectangle(img,(fx,fy),(fx+fw,fy+fh),(255,255,255),2)
        rof = img[fy:fy+fh, fx:fx+fw]
##      rofgray=grayscale[fy:fy+fh, fx:fx+fw]
        eyes = eyecascade.detectMultiScale(rof)
        smile=smilecascade.detectMultiScale(rof,1.7,22,minSize=(25,25))
        for (ex,ey,ew,eh) in eyes:    
            cv2.rectangle(rof,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(rof,(sx,sy),(sx+sw,sy+sh),(0,255,255),2)
    cv2.imshow('face',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    
