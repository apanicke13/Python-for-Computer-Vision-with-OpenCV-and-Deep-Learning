
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


def load_img():
    blank_img = np.zeros((600, 600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img, text = 'ABCDE', org = (50, 300), fontFace = font, fontScale = 5, color = (255, 255, 255), thickness = 25)
    return blank_img


# In[21]:


def display_img(img):
    fig = plt.figure(figsize = (12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, 'gray')


# In[22]:


img = load_img()
display_img(img)


# In[23]:


kernel = np.ones((5, 5), dtype = np.uint8)


# In[24]:


kernel


# In[26]:


result = cv2.erode(img, kernel, iterations = 1)
display_img(result)


# In[27]:


img = load_img()
display_img(img)


# In[28]:


result = cv2.erode(img, kernel, iterations = 4)
display_img(result)


# In[29]:


img = load_img()
display_img(img)


# In[31]:


white_noise = np.random.randint(low = 0, high = 2, size= (600, 600))


# In[32]:


white_noise


# In[33]:


display_img(white_noise)


# In[34]:


img = load_img()
display_img(img)


# In[35]:


img.max()


# In[36]:


white_noise = white_noise * 255


# In[37]:


white_noise


# In[38]:


display_img(white_noise)


# In[39]:


noise_img = white_noise + img


# In[40]:


display_img(noise_img)


# In[41]:


opening = cv2.morphologyEx(noise_img, cv2.MORPH_OPEN, kernel)


# In[42]:


display_img(opening)


# In[43]:


img = load_img()
display_img(img)


# In[44]:


black_noise = np.random.randint(low = 0, high = 2, size = (600, 600))


# In[46]:


black_noise = black_noise * -255


# In[47]:


black_noise


# In[48]:


black_noise_img = img + black_noise


# In[49]:


black_noise_img


# In[50]:


black_noise_img[black_noise == -255] = 0


# In[51]:


black_noise_img


# In[52]:


display_img(black_noise_img)


# In[53]:


closing = cv2.morphologyEx(black_noise_img, cv2.MORPH_CLOSE, kernel)


# In[54]:


display_img(closing)


# In[55]:


img = load_img()
display_img(img)


# In[56]:


gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
display_img(gradient)

