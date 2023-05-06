# importing the modules
import os
import cv2
import numpy as np

# set Width and Height of output Screen
frameWidth = 640
frameHeight = 480
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
haar_eye_model = os.path.join(cv2_base_dir, 'data/haarcascade_eye.xml')
# 实例化人脸分类器
face_cascade = cv2.CascadeClassifier(haar_model)#xml来源于资源文件。
eye_cascade = cv2.CascadeClassifier(haar_eye_model)#xml来源于资源文件。

# capturing Video from Webcam
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# set brightness, id is 10 and
# value can be changed accordingly
cap.set(10,150)

# draws your action on virtual canvas
def drawOnCanvas(myPoints, myColorValues):
	for point in myPoints:
		cv2.circle(imgResult, (point[0], point[1]),
				10, myColorValues[point[2]], cv2.FILLED)
	
# running infinite while loop so that
# program keep running until we close it
while True:
	success, img = cap.read()
	# 读取测试图片
	# img = cv2.imread('faces.jpg',cv2.IMREAD_COLOR)
	imgResult = img.copy()
	# 将原彩色图转换成灰度图
	gray = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
	# 开始在灰度图上检测人脸，输出是人脸区域的外接矩形框
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1)
	# 遍历人脸检测结果
	for (x,y,w,h) in faces:
		# 在原彩色图上画人脸矩形框
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = imgResult[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		if len(eyes) != 2:
			break
		cv2.rectangle(imgResult,(x,y),(x+w,y+h),(255,255,0),2)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	# displaying output on Screen
	cv2.imshow("Result", imgResult)
	
	# condition to break programs execution
	# press q to stop the execution of program
	if cv2.waitKey(1) and 0xFF == ord('q'):
		break
