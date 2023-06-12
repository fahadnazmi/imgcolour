import cv2

# Opens the Video file
cap= cv2.VideoCapture('F:\\DTU\Sem 6\\IT420 Computer Vision\\Colorize\\test\\landscape.mp4')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('kang'+str(i)+'.jpg',frame)
    i+=1

cap.release()
cv2.destroyAllWindows()