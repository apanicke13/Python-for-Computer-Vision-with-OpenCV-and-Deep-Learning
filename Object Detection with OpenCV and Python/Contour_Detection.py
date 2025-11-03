
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


img = cv2.imread('internal_external.png', 0)
plt.imshow(img, cmap = 'gray')


# In[4]:


image, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


# In[5]:


type(contours)


# In[6]:


len(contours)


# In[7]:


type(hierarchy)


# In[8]:


hierarchy


# In[9]:


external_contours = np.zeros(img.shape)


# In[11]:


external_contours


# In[13]:


external_contours.shape


# In[15]:


plt.imshow(image)


# In[16]:


for i in range(len(contours)):
    
    # External
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(external_contours, contours, i, 255, -1)


# In[19]:


plt.imshow(external_contours, cmap = 'gray')


# In[21]:


internal_contours = np.zeros(img.shape)

for i in range(len(contours)):
    
    # External
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(internal_contours, contours, i, 255, -1)
        
        
plt.imshow(internal_contours, cmap = 'gray')


# In[22]:


internal_contours = np.zeros(img.shape)

for i in range(len(contours)):
    
    # External
    if hierarchy[0][i][3] == 4:
        cv2.drawContours(internal_contours, contours, i, 255, -1)
        
        
plt.imshow(internal_contours, cmap = 'gray')


# In[23]:


internal_contours = np.zeros(img.shape)

for i in range(len(contours)):
    
    # External
    if hierarchy[0][i][3] == 0:
        cv2.drawContours(internal_contours, contours, i, 255, -1)
        
        
plt.imshow(internal_contours, cmap = 'gray')

