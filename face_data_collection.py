import cv2
import numpy as np

cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip=0

face_data=[]
dataset_path='./data/'
file_name=input("Enter the name of the person :")

while True:

	ret,frame=cap.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


	if ret==False:
		continue

	faces=face_cascade.detectMultiScale(gray,1.3,5)

	if len(faces)==0:
		continue
	faces=sorted(faces,key=lambda f:f[2]*f[3])

	
	# picking the last face which has largest area f[2]*f[3]
	for face in faces[-1:]:
		x,y,w,h=face
		cv2.rectangle(frame,(x,y),((x+w),(y+h)),(0,255,255),2)


		# extract(crop out the required face) :region of interest(ROI)

		off_set=10
		face_region=frame[y-off_set:y+h+off_set,x-off_set:x+w+off_set]
		face_region=cv2.resize(face_region,(100,100))

		skip+=1

		# capturing every 10th frame
		if skip%10==0:
			face_data.append(face_region)
			print(len(face_data))

	

	

	cv2.imshow('frame',frame)
	cv2.imshow('face section',face_region)

	key_pressed=cv2.waitKey(1) & 0xFF
	if key_pressed==ord('q'):
		break


# convert our face list into numpy array
face_data=np.array(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)


# save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("data saved sucessfully at "+dataset_path+file_name+'.npy')


cap.release()
cv2.destroyAllWindows()
