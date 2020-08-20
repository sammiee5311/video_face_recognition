# video_face_recognition(+blur)

## How to use
+ download main.py and requirements.txt
+ download required libraries(simply write 'pip install -r requirements.txt' on console)
+ put images file you want to recognize faces in 'images' folder(or any where you want)
+ change the images names to person's name that you want to recognize
+ write the images path ,the video file name and names that you want to blur on main.py
+ if video file name is empty, it's automatically changed to Webcam.

## Requirements
+ numpy
+ opencv-python
+ cmake
+ dlib
+ face-recognition

## Images

### if people are blurred, their names will be changed to UNKNOWN.
![](./images/blur.png)

### Reference
+ https://github.com/ageitgey/face_recognition
+ https://youtu.be/sz25xxF_AVE
+ https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/
