
# coding: utf-8

# In[14]:


import cv2
import numpy as py
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


img = cv2.imread('rainbow.jpg', 0)


# In[20]:


img


# In[23]:


plt.imshow(img, cmap = 'gray')


# In[25]:


img.max()


# In[26]:


ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)


# In[27]:


ret


# In[28]:


plt.imshow(thresh1, 'gray')


# In[33]:


ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)


# In[34]:


ret


# In[35]:


plt.imshow(thresh1, 'gray')


# In[39]:


ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
plt.imshow(thresh1, 'gray')


# In[40]:


ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
plt.imshow(thresh1, 'gray')


# In[41]:


ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
plt.imshow(thresh1, 'gray')


# In[42]:


img = cv2.imread('crossword.jpg', 0)


# In[44]:


plt.imshow(img, 'gray')


# In[49]:


def show_pic(img):
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img, 'gray')


# In[50]:


show_pic(img)


# In[55]:


ret, th1 = cv2.threshold(img, 145, 255, cv2.THRESH_BINARY)
show_pic(th1)


# In[58]:


th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
show_pic(th2)


# In[60]:


blended = cv2.addWeighted(src1 = th1, alpha = 0.6, src2 = th2, beta = 0.4, gamma = 0)
show_pic(blended)

