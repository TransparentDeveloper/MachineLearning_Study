import cv2

cap = cv2.VideoCapture('Week1/04.mp4')

while True:
	ret, img = cap.read()

	if ret == False:
		break

	cv2.imshow('result', img)
    
	if cv2.waitKey(10) == ord('q'):
		break