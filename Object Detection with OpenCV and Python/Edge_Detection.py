
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


img = cv2.imread('sammy_face.jpg')
plt.imshow(img)


# In[3]:


edges = cv2.Canny(image = img, threshold1 = 127, threshold2 = 127)


# In[4]:


plt.imshow(edges)


# In[5]:


edges = cv2.Canny(image = img, threshold1 = 0, threshold2 = 255)
plt.imshow(edges)


# In[6]:


med_val = np.median(img)


# In[7]:


med_val


# In[8]:


# SET THRESHOLD TO 0 OR 70% OF MEDIAN VALUE WHICH EVER IS GREATER
lower = int(max(0, 0.7 * med_val))
# SET THRESHOLD TO 130% OF THE MEDIAN OR THE MAX 255
upper = int(min(255, 1.3 * med_val))


# In[9]:


edges = cv2.Canny(image = img, threshold1 = lower, threshold2 = upper)
plt.imshow(edges)


# In[10]:


edges = cv2.Canny(image = img, threshold1 = lower, threshold2 = upper + 100)
plt.imshow(edges)


# In[14]:


blur = cv2.blur(img, ksize = (7, 7))


# In[16]:


edges = cv2.Canny(image = blur, threshold1 = lower, threshold2 = upper + 50)
plt.imshow(edges)

