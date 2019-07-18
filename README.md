# OpenCV
OpenCV is an open source computer vision and machine learning software python library.
The Library provides more than 2500 algorithms that include machine learning tools for classification and clustering, image processing and vision algorithm, basic algorithms and drawing functions, GUI and I/O functions for images and videos. Some applications of these algorithms include face detection, object recognition, extracting 3D models, image processing, camera calibration, motion analysis etc.

# Face Detection
Face detection has gained a lot of attention due to its real-time applications. A lot of research has been done and still going on for improved and fast implementation of the face detection algorithm. Why is face detection difficult for a machine? Face detection is not as easy as it seems due to lots of variations of image appearance, such as pose variation (front, non-front), occlusion, image orientation, illumination changes and facial expression.

OpenCV contains many pre-trained classifiers for face, eyes, smile etc.

In this project I have used the haarcascade classifier which uses it's haar features to detect face.
![image](https://user-images.githubusercontent.com/41102775/61474351-1e540980-a9a6-11e9-852a-98b8f7fd9197.png)  
These are some Haar features which are similar to convolutional kernals(used in convolutional neural network).


# Face Recognition
Using the above method the data of the detected face gets stored in the data folder in the same directory as that of the dectection code.Then one can use multiple Classification algorthms to recognize and predict the name of the user.

In this, I have used <b>KNN algorithm</b> to classify the name of the user after detecting their face.

# About Project Files
- **data folder**  
It contains the face data along with the names of the user. This data is used for training our face detection model.

- **FaceRecognition_usingKNN.py**  
It conatins python code for classifying the name of the user after detecting their face.

- **ReadingVideoStream.py**  
It contains basic code for reading Video from front camera of Laptops/PC's using OpenCV.

- **face_data_collection.py**  
This python script is used for collecting face data from the user along with their names.

- **face_detection.py**  
This python script is used for detecting face and making a rectangular box around it.
