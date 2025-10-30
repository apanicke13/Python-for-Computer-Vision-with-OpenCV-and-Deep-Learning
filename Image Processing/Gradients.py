
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


def display_img(img):
    fig = plt.figure(figsize = (12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, 'gray')


# In[7]:


img = cv2.imread('sudoku.jpg', 0)
display_img(img)


# In[8]:


sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)


# In[11]:


display_img(sobelx)


# In[12]:


sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)
display_img(sobely)


# In[13]:


laplace = cv2.Laplacian(img, cv2.CV_64F)
display_img(laplace)


# In[15]:


blended = cv2.addWeighted(src1 = sobelx, alpha = 0.5, src2 = sobely, beta = 0.5, gamma = 0)


# In[16]:


display_img(blended)


# In[17]:


ret, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
display_img(th1)


# In[20]:


kernel = np.ones((4, 4), np.uint8)
grad = cv2.morphologyEx(blended, cv2.MORPH_GRADIENT, kernel)
display_img(grad)

