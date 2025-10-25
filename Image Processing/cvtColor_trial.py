
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


img = cv2.imread('00-puppy.jpg')


# In[3]:


plt.imshow(img)


# In[5]:


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[6]:


plt.imshow(img)


# In[9]:


img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
plt.imshow(img)


# In[10]:


img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
plt.imshow(img)

