
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


dark_horse = cv2.imread('horse.jpg')
show_horse = cv2.cvtColor(dark_horse, cv2.COLOR_BGR2RGB)

rainbow = cv2.imread('rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)

blue_bricks = cv2.imread('bricks.jpg')
show_bricks = cv2.cvtColor(blue_bricks, cv2.COLOR_BGR2RGB)


# In[6]:


plt.imshow(show_horse)


# In[7]:


plt.imshow(show_rainbow)


# In[8]:


plt.imshow(show_bricks)


# In[9]:


# OPENCV BGR
hist_values = cv2.calcHist([blue_bricks], channels = [0], mask = None, histSize = [256], ranges = [0, 256])


# In[10]:


hist_values.shape


# In[11]:


plt.plot(hist_values)


# In[12]:


# OPENCV BGR
hist_values = cv2.calcHist([dark_horse], channels = [0], mask = None, histSize = [256], ranges = [0, 256])
plt.plot(hist_values)


# In[13]:


img = blue_bricks


# In[14]:


color = ('b', 'g', 'r')

for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color = col)
    plt.xlim([0, 256])
    
plt.title('HISTOGRAM FOR BLUE BRICK')


# In[16]:


img = dark_horse
color = ('b', 'g', 'r')

for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color = col)
    plt.xlim([0, 50])
    plt.ylim([0, 500000])
    
plt.title('HISTOGRAM FOR DARK HORSE')


# In[17]:


rainbow = cv2.imread('rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)


# In[18]:


img = rainbow


# In[19]:


img.shape


# In[21]:


img.shape[:2]


# In[24]:


mask = np.zeros(img.shape[:2], np.uint8)


# In[26]:


# plt.imshow(mask, 'gray')
mask[300:400, 100:400] = 255


# In[27]:


plt.imshow(mask, 'gray')


# In[28]:


plt.imshow(show_rainbow)


# In[29]:


masked_img = cv2.bitwise_and(img, img, mask = mask)


# In[30]:


show_masked_img = cv2.bitwise_and(show_rainbow, show_rainbow, mask = mask)


# In[31]:


plt.imshow(show_masked_img)


# In[33]:


hist_mask_values_red = cv2.calcHist([rainbow], channels = [2], mask = mask, histSize = [256], ranges = [0, 256])


# In[37]:


hist_values_red = cv2.calcHist([rainbow], channels = [2], mask = None, histSize = [256], ranges = [0, 256])


# In[38]:


plt.plot(hist_mask_values_red)
plt.title('RED HISTOGRAM FOR MASKED RAINBOW')


# In[39]:


plt.plot(hist_values_red)
plt.title('HISTOGRAM FOR FULL RAINBOW IMAGE')


# In[40]:


gorilla = cv2.imread('gorilla.jpg', 0)


# In[41]:


def display_img(img):
    fig = plt.figure(figsize = (12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, 'gray')


# In[43]:


display_img(gorilla)


# In[44]:


gorilla.shape


# In[45]:


hist_values = cv2.calcHist([gorilla], channels = [0], mask = None, histSize = [256], ranges = [0, 256])


# In[46]:


plt.plot(hist_values)


# In[47]:


eq_gorilla = cv2.equalizeHist(gorilla)


# In[50]:


display_img(eq_gorilla)


# In[51]:


hist_values = cv2.calcHist([eq_gorilla], channels = [0], mask = None, histSize = [256], ranges = [0, 256])


# In[52]:


plt.plot(hist_values)


# In[61]:


color_gorilla = cv2.imread('gorilla.jpg')


# In[62]:


show_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2RGB)


# In[63]:


plt.imshow(show_gorilla)


# In[64]:


hsv = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2HSV)


# In[65]:


hsv[:,:,2]


# In[66]:


hsv[:,:,2].max()


# In[67]:


hsv[:,:,2].min()


# In[68]:


hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])


# In[69]:


eq_color_gorilla = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# In[70]:


display_img(eq_color_gorilla)

