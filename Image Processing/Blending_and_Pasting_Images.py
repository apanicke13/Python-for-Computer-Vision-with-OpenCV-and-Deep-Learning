
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


img1 = cv2.imread('dog_backpack.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


# In[6]:


plt.imshow(img1)


# In[7]:


plt.imshow(img2)


# In[8]:


img1.shape


# In[9]:


img2.shape


# In[10]:


# Blending images of the same size
img1 = cv2.resize(img1, (1200, 1200))
img2 = cv2.resize(img2, (1200, 1200)) 


# In[11]:


plt.imshow(img1)


# In[12]:


plt.imshow(img2)


# In[13]:


blended = cv2.addWeighted(src1 = img1, alpha = 0.5, src2 = img2, beta = 0.5, gamma = 0)


# In[16]:


plt.imshow(blended)


# In[17]:


blended = cv2.addWeighted(src1 = img1, alpha = 0.8, src2 = img2, beta = 0.5, gamma = 0)
plt.imshow(blended)


# In[19]:


# Overlay a small image on top of a larger image (no blending)
# Numpy reassignment opperation
img1 = cv2.imread('dog_backpack.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


# In[21]:


img2 = cv2.resize(img2, (600, 600))
plt.imshow(img2)


# In[23]:


plt.imshow(img1)


# In[24]:


large_img = img1
small_img = img2


# In[25]:


x_offset = 0
y_offset = 0


# In[29]:


x_end = x_offset + small_img.shape[1]
y_end = y_offset + small_img.shape[0]


# In[30]:


large_img[y_offset:y_end, x_offset:x_end] = small_img


# In[31]:


plt.imshow(large_img)


# In[32]:


# How you an blend together images of different sizes
img1 = cv2.imread('dog_backpack.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


# In[33]:


img2 = cv2.resize(img2, (600, 600))
plt.imshow(img2)


# In[34]:


plt.imshow(img1)


# In[35]:


img1.shape


# In[36]:


x_offset = 934 - 600
y_offset = 1401 - 600


# In[37]:


x_end = 934
y_end = 1401


# In[38]:


img2.shape


# In[41]:


rows,cols,channels = img2.shape


# In[43]:


rows


# In[44]:


cols


# In[45]:


channels


# In[46]:


roi = img1[y_offset:1401, x_offset:943]


# In[47]:


roi


# In[48]:


plt.imshow(roi)


# In[51]:


img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
plt.imshow(img2gray, cmap = 'gray')


# In[52]:


mask_inv = cv2.bitwise_not(img2gray)


# In[54]:


plt.imshow(mask_inv, cmap = 'gray')


# In[56]:


mask_inv.shape


# In[57]:


import numpy as np


# In[58]:


white_bg = np.full(img2.shape, 255, dtype = np.uint8)


# In[59]:


plt.imshow(white_bg)


# In[60]:


white_bg.shape


# In[61]:


bk = cv2.bitwise_or(white_bg, white_bg, mask = mask_inv)


# In[62]:


bk.shape


# In[63]:


plt.imshow(bk)


# In[64]:


fg = cv2.bitwise_or(img2, img2, mask = mask_inv)


# In[65]:


plt.imshow(fg)


# In[66]:


final_roi = cv2.bitwise_or(roi, fg)


# In[67]:


plt.imshow(final_roi)


# In[68]:


large_img = img1
small_img = final_roi


# In[71]:


large_img[y_offset:y_offset+small_img.shape[0], x_offset:x_offset+small_img.shape[1]] = small_img


# In[72]:


plt.imshow(large_img)

