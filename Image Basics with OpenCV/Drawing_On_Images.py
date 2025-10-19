
# coding: utf-8

# In[1]:


import cv2
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


blank_img = np.zeros(shape = (512,512,3), dtype = np.int16)


# In[9]:


plt.imshow(blank_img)


# In[10]:


cv2.rectangle(blank_img, pt1 = (384, 10), pt2 = (500, 150), color = (0, 255, 0), thickness = 10)


# In[11]:


plt.imshow(blank_img)


# In[14]:


cv2.rectangle(blank_img, pt1 = (200, 200), pt2 = (300, 300), color = (0, 0, 255), thickness = 10)


# In[15]:


plt.imshow(blank_img)


# In[19]:


cv2.circle(blank_img, center = (100, 100), radius = 50, color = (255, 0, 0), thickness = 8)


# In[20]:


plt.imshow(blank_img)


# In[21]:


cv2.circle(blank_img, center = (400, 400), radius = 50, color = (255, 0, 0), thickness = -1)
plt.imshow(blank_img)


# In[24]:


cv2.line(blank_img, pt1 = (0, 0), pt2 = (512, 512), color = (102, 255, 255), thickness = 5)
plt.imshow(blank_img)


# In[27]:


font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(blank_img, text = 'Hello', org = (10, 500), fontFace = font, fontScale = 4, color = (255, 255, 255), thickness = 3, lineType = cv2.LINE_AA)
9shapeplt.imshow(blank_img)


# In[29]:


blank_img = np.zeros(shape=(512, 512, 3), dtype = np.int32)


# In[30]:


plt.imshow(blank_img)


# In[32]:


vertices = np.array([[100, 300], [200, 200], [400, 300], [200, 400]], dtype = np.int32)
plt.imshow(blank_img)


# In[33]:


vertices


# In[34]:


vertices.shape


# In[35]:


pts = vertices.reshape((-1, 1, 2))


# In[36]:


vertices.shape


# In[37]:


pts.shape


# In[38]:


pts


# In[40]:


cv2.polylines(blank_img, [pts], isClosed = True, color = (255, 0, 0), thickness = 5)
plt.imshow(blank_img)

