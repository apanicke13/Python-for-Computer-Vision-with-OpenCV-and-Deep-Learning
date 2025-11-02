
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


flat_chess = cv2.imread('flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)
plt.imshow(flat_chess)


# In[3]:


gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_flat_chess, cmap = 'gray')


# In[4]:


real_chess = cv2.imread('real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
plt.imshow(real_chess)


# In[5]:


gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_real_chess, cmap = 'gray')


# In[13]:


corners = cv2.goodFeaturesToTrack(gray_flat_chess, 64, 0.01, 10)


# In[14]:


corners


# In[15]:


corners = np.int0(corners)


# In[16]:


corners


# In[17]:


for i in corners:
    x, y = i.ravel()
    cv2.circle(flat_chess, (x, y), 3, (255, 0, 0), -1)
    


# In[18]:


plt.imshow(flat_chess)


# In[25]:


corners = cv2.goodFeaturesToTrack(gray_real_chess, 100, 0.01, 10)


# In[26]:


corners = np.int0(corners)


# In[27]:


for i in corners:
    x, y = i.ravel()
    cv2.circle(real_chess, (x, y), 3, (255, 0, 0), -1)
    


# In[28]:


plt.imshow(real_chess)

