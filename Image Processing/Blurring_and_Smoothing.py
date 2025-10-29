
# coding: utf-8

# In[10]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


def load_img():
    img = cv2.imread('bricks.jpg').astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# In[12]:


# load_img()


# In[13]:


def display_img(img):
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img, 'gray')


# In[14]:


i = load_img()
display_img(i)


# In[18]:


gamma = 1/10


# In[19]:


result = np.power(i, gamma)
display_img(result)


# In[20]:


gamma = 4
result = np.power(i, gamma)
display_img(result)


# In[23]:


img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, 'bricks', org = (10, 600), fontFace = font, fontScale = 10, color = (255, 0, 0), thickness = 4)


# In[24]:


display_img(img)


# In[25]:


kernel = np.ones(shape = (5, 5), dtype = np.float32) / 25


# In[26]:


kernel


# In[27]:


1/25


# In[28]:


dst = cv2.filter2D(img, -1, kernel)


# In[30]:


display_img(dst)


# In[31]:


img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, 'bricks', org = (10, 600), fontFace = font, fontScale = 10, color = (255, 0, 0), thickness = 4)
print('reset')


# In[33]:


blurred = cv2.blur(img, ksize = (10, 10))
display_img(blurred)


# In[35]:


img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, 'bricks', org = (10, 600), fontFace = font, fontScale = 10, color = (255, 0, 0), thickness = 4)
print('reset')
display_img(img)


# In[40]:


blur = cv2.GaussianBlur(img, (5, 5), 10)
display_img(blur)


# In[41]:


img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, 'bricks', org = (10, 600), fontFace = font, fontScale = 10, color = (255, 0, 0), thickness = 4)
print('reset')
display_img(img)


# In[42]:


blurs = cv2.medianBlur(img, 5)
display_img(blurs)


# In[43]:


img = cv2.imread('sammy.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
display_img(img)


# In[44]:


noise_img = cv2.imread('sammy_noise.jpg')
display_img(noise_img)


# In[45]:


median = cv2.medianBlur(noise_img, 5)
display_img(median)


# In[46]:


img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, 'bricks', org = (10, 600), fontFace = font, fontScale = 10, color = (255, 0, 0), thickness = 4)
print('reset')
display_img(img)


# In[47]:


bilateral = cv2.bilateralFilter(img, 9, 75, 75)


# In[48]:


display_img(bilateral)

