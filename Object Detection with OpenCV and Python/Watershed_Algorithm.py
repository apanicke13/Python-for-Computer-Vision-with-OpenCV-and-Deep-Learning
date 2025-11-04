
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def display_img(img, cmap = 'gray'):
    fig = plt.figure(figsize = (12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, 'gray')


# In[3]:


sep_coins = cv2.imread('pennies.jpg')
display_img(sep_coins)


# In[4]:


# Median blur
blur = cv2.medianBlur(sep_coins, 25)

# Convert to gray scale

# binary Threshold

# Find Contours


# In[5]:


display_img(blur)


# In[6]:


gray_sep_coins = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
display_img(gray_sep_coins)


# In[7]:


ret, sep_thresh = cv2.threshold(gray_sep_coins, 160, 255, cv2.THRESH_BINARY_INV)


# In[8]:


display_img(sep_thresh)


# In[9]:


image, contours, heirarchy = cv2.findContours(sep_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


# In[11]:


for i in range(len(contours)):
    if heirarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins, contours, i, (255, 0, 0), 10)
        


# In[12]:


display_img(sep_coins)


# In[13]:


img = cv2.imread('pennies.jpg')
img = cv2.medianBlur(img, 35)
display_img(img)


# In[14]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[17]:


ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


# In[18]:


display_img(thresh)


# In[19]:


# Noise Removal
kernel = np.ones((3, 3), np.uint8)


# In[20]:


kernel


# In[22]:


opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)


# In[23]:


display_img(opening)


# In[24]:


dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)


# In[25]:


display_img(dist_transform)


# In[31]:


sure_bg = cv2.dilate(opening, kernel, iterations = 3)
display_img(sure_bg)


# In[26]:


ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)


# In[27]:


display_img(sure_fg)


# In[32]:


sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)


# In[33]:


display_img(unknown)


# In[34]:


ret, markers = cv2.connectedComponents(sure_fg)


# In[35]:


markers


# In[36]:


markers = markers + 1


# In[37]:


markers


# In[38]:


markers[unknown == 255] = 0


# In[39]:


display_img(markers)


# In[40]:


markers = cv2.watershed(img, markers)


# In[41]:


display_img(markers)


# In[43]:


image, contours, heirarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if heirarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins, contours, i, (255, 0, 0), 10)
        
display_img(sep_coins)

