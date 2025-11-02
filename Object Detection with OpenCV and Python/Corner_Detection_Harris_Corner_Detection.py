
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


# In[4]:


gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_flat_chess, cmap = 'gray')


# In[5]:


real_chess = cv2.imread('real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
plt.imshow(real_chess)


# In[6]:


gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_real_chess, cmap = 'gray')


# In[7]:


gray_flat_chess


# In[8]:


gray = np.float32(gray_flat_chess)


# In[9]:


dst = cv2.cornerHarris(src = gray, blockSize = 2, ksize = 3, k = 0.04)
plt.imshow(dst)


# In[10]:


dst = cv2.dilate(dst, None)
plt.imshow(dst)


# In[11]:


flat_chess[dst > 0.01 * dst.max()] = [255, 0, 0]
plt.imshow(flat_chess)


# In[13]:


gray = np.float32(gray_real_chess)

dst = cv2.cornerHarris(src = gray, blockSize = 2, ksize = 3, k = 0.04)
dst = cv2.dilate(dst, None)
real_chess[dst > 0.01 * dst.max()] = [255, 0, 0]
plt.imshow(real_chess)

