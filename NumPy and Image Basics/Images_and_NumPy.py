
# coding: utf-8

# In[5]:


import numpy as np


# In[30]:


import matplotlib.pyplot as plt


# In[7]:


from PIL import Image


# In[11]:


pic = Image.open('00-puppy.jpg')


# In[37]:


pic


# In[56]:


pic_arr = np.asarray(pic)


# In[57]:


type(pic_arr)


# In[58]:


pic_arr.shape


# In[59]:


plt.imshow(pic_arr)
plt.show()


# In[60]:


pic_red = pic_arr.copy()


# In[61]:


plt.imshow(pic_red)
plt.show()


# In[62]:


pic_red.shape


# In[69]:


# R G B
# RED channel values 0 means no red - pure black and 255 means full pure red - white
# So since ears have red it will be white in gray scale
plt.imshow(pic_red[:,:,0], cmap='gray') # closer, more red
plt.show()


# In[70]:


pic_red[:,:,0]


# In[71]:


plt.imshow(pic_red[:,:,1], cmap='gray')
plt.show()


# In[72]:


plt.imshow(pic_red[:,:,2], cmap='gray')
plt.show()


# In[74]:


# GREEN CHANNEL - because we removed green
pic_red[:,:,1] = 0


# In[75]:


plt.imshow(pic_red)
plt.show()


# In[78]:


# Getting rid of blue
pic_red[:,:,2] = 0


# In[79]:


plt.imshow(pic_red)
plt.show()

