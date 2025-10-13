
# coding: utf-8

# In[1]:


import cv2


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


img = cv2.imread('00-puppy.jpg')


# In[14]:


type(img)


# In[16]:


img.shape


# In[17]:


plt.imshow(img)


# In[19]:


# MATPLOTLOB --> R G B order
# OPENCV --> B G R order
# Thats why the image colour issue

# So what we need to do is
fix_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


# In[20]:


plt.imshow(fix_img)


# In[21]:


img_gray = cv2.imread('00-puppy.jpg',cv2.IMREAD_GRAYSCALE)


# In[23]:


plt.imshow(img_gray, cmap='gray')


# In[24]:


plt.imshow(img_gray, cmap='magma')


# In[25]:


plt.imshow(fix_img)


# In[27]:


new_img = cv2.resize(fix_img, (1000,400))


# In[28]:


plt.imshow(new_img)


# In[35]:


w_ratio = 0.8 # 80% of original width
h_ratio = 0.5


# In[36]:


new_img = cv2.resize(fix_img, (0,0), fix_img, w_ratio, h_ratio)


# In[37]:


plt.imshow(new_img)


# In[38]:


new_img = cv2.flip(fix_img, 0)
plt.imshow(new_img)


# In[39]:


new_img = cv2.flip(fix_img, 1)
plt.imshow(new_img)


# In[40]:


new_img = cv2.flip(fix_img, -1)
plt.imshow(new_img)


# In[41]:


type(fix_img)


# In[42]:


cv2.imwrite('totally_new.jpg', fix_img)


# In[43]:


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.imshow(fix_img)

