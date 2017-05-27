import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cv2.namedWindow("image")
a = 0
cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
print cascade
while (cap.isOpened()):
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cascade.detectMultiScale(gray, scaleFactor=1.3,
                                      minNeighbors=4,
                                      minSize=(30, 30),
                                      flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    for (x, y, w, h) in face:

        cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0, 0), 2)

    cv2.imshow("image", img)
    keypress = cv2.waitKey(20) & 0xFF
    if keypress == 27:
        break
    elif keypress == ord('p'):
        cv2.imwrite("imgs\img" + str(a) + ".jpg", img[:640, :480])
        a = a + 1

cap.release()
cv2.destroyAllWindows()
