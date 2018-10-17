import cv2

import numpy as np
import os

######################  KNN CODE   ########################
def distance(x1,x2):
	return np.sqrt(((x1-x2)**2).sum())

def KNN(train,test,k=5):

	dist=[]
	for ix in range(train.shape[0]):

		x=train[ix,:-1]
		y=train[ix,-1]

		d=distance(test,x)
		dist.append((d,y))

	# print(dist)

	dist=sorted(dist,key=lambda x:x[0])
	dist=dist[:k]

	labels=np.array(dist)
	labels=labels[:,-1]

	output=np.unique(labels,return_counts=True)

	index=np.argmax(output[1])

	return output[0][index]


###############################################################

# init camera

cap=cv2.VideoCapture(0)

#Face Detection

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip=0

dataset_path='./data/'

face_data=[]
labels=[]

class_id=0  #labels for the given file
names={}   #mapping between id - name

# Data preparation

for fx in os.listdir(dataset_path):

	if fx.endswith('.npy'):

		# mapping class id with name

		names[class_id]=fx[:-4] #storing name of the file

		print("loaded "+fx)

		data_item=np.load(dataset_path+fx)
		face_data.append(data_item)

		# creating labels for the classes

		target=class_id*np.ones((data_item.shape[0],))
		class_id +=1

		labels.append(target)

face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset=np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)


# Testing

while True:

	ret,frame=cap.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret==False:
		continue

	faces=face_cascade.detectMultiScale(gray,1.3,5)

	for face in faces:
		x,y,w,h=face

		# get the face ROI

		offset=10
		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))

		# predicted labels 
		out=KNN(trainset,face_section.flatten())

		# display on the screen the name and rectangle around it
		pred_name=names[int(out)] 
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)

		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

	cv2.imshow("Faces",frame)

	key_pressed=cv2.waitKey(1) & 0xFF

	if key_pressed==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
