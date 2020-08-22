import cv2
import numpy as np
import face_recognition
from glob import glob


class face_recog:
    def __init__(self, path, video_file='', names=[]):
        self.path = path
        self.images = []
        self.names_list = []
        self.video_file = video_file
        self.people = glob(path + '/*.jpg')
        self.blurred = set()
        for name in names:
            self.blurred.add(name.upper())

    def find_face_encodings(self):
        encodings_list = []
        for img in self.images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings_list.append(face_recognition.face_encodings(img)[0])

        return encodings_list

    def start_face_recog(self):
        for person in self.people:
            cur = cv2.imread(person)
            self.images.append(cur)
            self.names_list.append(person.split('\\')[1][:-4])

        print(self.names_list)

        encode_known = self.find_face_encodings()
        print('Encoding Complete')

        self.video_capture(encode_known, self.video_file)

    def make_blur(self, img, blocks=3):
        (h, w) = img.shape[:2]
        x = np.linspace(0, w, blocks + 1, dtype="int")
        y = np.linspace(0, h, blocks + 1, dtype="int")
        for i in range(1, len(y)):
            for j in range(1, len(x)):
                left = x[j - 1]
                top = y[i - 1]
                right = x[j]
                bottom = y[i]
                roi = img[top:bottom, left:right]
                (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
                cv2.rectangle(img, (left, top), (right, bottom),
                              (B, G, R), -1)
        return img

    def video_capture(self, encode_known, video_file):
        if video_file:
            cap = cv2.VideoCapture(video_file)
        else:
            cap = cv2.VideoCapture(0)

        while True:
            success, img = cap.read()
            resized_img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

            cur_locations = face_recognition.face_locations(resized_img)
            cur_encodings = face_recognition.face_encodings(resized_img, cur_locations)

            for encodings_face, location in zip(cur_encodings, cur_locations):
                check = face_recognition.compare_faces(encode_known, encodings_face)
                face_distance = face_recognition.face_distance(encode_known, encodings_face)
                matched_idx = np.argmin(face_distance)

                name = self.names_list[matched_idx].upper() if check[matched_idx] else 'UNKNOWN'
                top, right, bottom, left = location
                top, right, bottom, left = 4 * top, 4 * right, 4 * bottom, 4 * left
                if name in self.blurred:
                    face = img[top:bottom, left:right]
                    face = self.make_blur(face, blocks=6)
                    img[top:bottom, left:right] = face
                    name = 'UNKNOWN'
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('img', img)
            cv2.waitKey(1)


# Write the images path / Write the video file name / Write names that you want to blur
# if video file name is empty, it's automatically changed to Webcam.
video_face_recog = face_recog('path', 'video', ['name'])
# video_face_recog = face_recog('./Images','video.mp4',['Jack'])
video_face_recog.start_face_recog()
