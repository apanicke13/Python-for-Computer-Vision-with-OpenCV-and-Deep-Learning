
# coding: utf-8

# In[2]:


import cv2
import time

cap = cv2.VideoCapture('finger_move.mp4')
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if cap.isOpened() == False:
    print('ERROR FILE NOT FOUND!')
    
while cap.isOpened():
    
    
    time.sleep(1/frames)
    ret, frame = cap.read()
    
    if ret == True:
        cv2.imshow('frame', frame)
    
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    else:
        break

cap.release()
cv2.destroyAllWindows()

