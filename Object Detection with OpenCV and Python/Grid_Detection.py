
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


flat_Chess = cv2.imread('flat_chessboard.png')


# In[3]:


plt.imshow(flat_Chess)


# In[4]:


found, corners = cv2.findChessboardCorners(flat_Chess, (7, 7))


# In[5]:


found


# In[7]:


corners


# In[8]:


cv2.drawChessboardCorners(flat_Chess, (7, 7), corners, found)


# In[9]:


plt.imshow(flat_Chess)


# In[11]:


dots = cv2.imread('dot_grid.png')


# In[12]:


plt.imshow(dots)


# In[13]:


found, corners = cv2.findCirclesGrid(dots, (10, 10), cv2.CALIB_CB_SYMMETRIC_GRID)


# In[14]:


found


# In[15]:


corners


# In[16]:


cv2.drawChessboardCorners(dots, (10, 10), corners, found)


# In[17]:


plt.imshow(dots)

