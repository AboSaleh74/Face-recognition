#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import glob
import numpy as np
import os
import face_recognition


# In[6]:


class Simple_facerec:
    def __init__(self):
        self.known_face_encoding=[]
        self.known_face_name=[]
        self.frame_resize = 0.25
        
    def load_encoding_image(self,Img_path):
        Img_path = glob.glob(os.path.join(Img_path,'*.*'))
        print(f'{len(Img_path)} images found')
        for img_path in Img_path:
            image = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            
            basename = os.path.basename(img_path)
            filename,ext = os.path.splitext(basename)
            
            img_encoding = face_recognition.face_encodings(rgb_img)[0]
            
            self.known_face_encoding.append(img_encoding)
            self.known_face_name.append(filename)
    def detect_known_face(self,frame):
        #frame = cv2.resize(frame,(0,0),fx=self.frame_resize,fy=self.frame_resize)
        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) 
        face_loc = face_recognition.face_locations(frame_rgb)
        face_encodings = face_recognition.face_encodings(frame_rgb,face_loc)
        
        face_names = []
        for face_encoding in face_encodings:
            match = face_recognition.compare_faces(self.known_face_encoding,face_encoding)
            name = 'Unknown'
            
            face_distances = face_recognition.face_distance(self.known_face_encoding,face_encoding)
            best_match = np.argmin(face_distances)
            if match[best_match]:
                name = self.known_face_name[best_match]
            face_names.append(name)
        face_loc = np.array(face_loc)
        #face_loc = face_loc/self.frame_resize
        return face_loc.astype(int),face_names

