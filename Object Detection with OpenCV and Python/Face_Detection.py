
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


nadia = cv2.imread('Nadia_Murad.jpg', 0)
denis = cv2.imread('Denis_Mukwege.jpg', 0)
solvay = cv2.imread('solvay_conference.jpg', 0)


# In[8]:


plt.imshow(nadia, cmap = 'gray')


# In[9]:


plt.imshow(denis, cmap = 'gray')


# In[10]:


plt.imshow(solvay, cmap = 'gray')


# In[12]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[13]:


def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)
    
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
        
    return face_img


# In[14]:


result = detect_face(nadia)
plt.imshow(result, cmap = 'gray')


# In[15]:


result = detect_face(denis)
plt.imshow(result, cmap = 'gray')


# In[16]:


result = detect_face(solvay)
plt.imshow(result, cmap = 'gray')


# In[20]:


def adjusted_detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor = 1.2, minNeighbors = 5)
    
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
        
    return face_img


# In[21]:


result = adjusted_detect_face(solvay)
plt.imshow(result, cmap = 'gray')


# In[37]:


eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# In[44]:


def detect_eyes(img):
    
    face_img = img.copy()
    eyes_rects = eye_cascade.detectMultiScale(face_img, scaleFactor = 1.2, minNeighbors = 5)
    
    for (x, y, w, h) in eyes_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
        
    return face_img


# In[45]:


result = detect_eyes(nadia)
plt.imshow(result, cmap = 'gray')


# In[46]:


result = detect_eyes(denis)
plt.imshow(result, cmap = 'gray')


# In[48]:


cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read(0)
    
    frame = detect_face(frame)
    
    cv2.imshow('Video Face Detect', frame)
    
    k = cv2.waitKey(1)
    if k == 27:
        break
        
cap.release()
cv2.destroyAllWindows()

