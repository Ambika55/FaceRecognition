# reading a video stream from web camera

import cv2

cap=cv2.VideoCapture(0)

while True:

	ret,frame=cap.read()
   
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret== False:
		continue

	cv2.imshow('video frame',frame)
	cv2.imshow('gray frame',gray)

	# wait for user input-q  to stop the capture
	key_pressed=cv2.waitKey(1) & 0xFF

	if key_pressed==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
