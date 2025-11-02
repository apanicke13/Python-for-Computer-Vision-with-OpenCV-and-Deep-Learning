
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


full = cv2.imread('sammy.jpg')
full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)


# In[3]:


plt.imshow(full)


# In[4]:


face = cv2.imread('sammy_face.jpg')
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
plt.imshow(face)


# In[5]:


full.shape


# In[6]:


face.shape


# In[7]:


sum([1, 2, 3])


# In[8]:


mystring = 'sum'


# In[9]:


eval(mystring)


# In[10]:


myfunc = eval(mystring)


# In[12]:


myfunc([1, 2, 3])


# In[13]:


# All the 6 methods for comparison in a list
# Note how we are using strings, later on we'll use the eval() function to convert to function
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


# In[16]:


my_method = eval('cv2.TM_CCOEFF')
res = cv2.matchTemplate(full, face, my_method)
plt.imshow(res)


# In[18]:


for m in methods:
    # Create a copy of the image
    full_copy = full.copy()
    
    method = eval(m)
    
    # TEMPLATE MATCHING
    res = cv2.matchTemplate(full_copy, face, method)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        
    else:
        top_left = max_loc
        
    height, width, channels = face.shape
    
    bottom_right = (top_left[0] + width, top_left[1] + height)
    
    cv2.rectangle(full_copy, top_left, bottom_right, (255, 0, 0), 10)
    
    # PLOT AND SHOW IMAGES
    plt.subplot(121)
    plt.imshow(res)
    plt.title('HEATMAP OF TEMPLATE MATCHING')
    
    plt.subplot(122)
    plt.imshow(full_copy)
    plt.title('DETECTION OF TEMPLATE')
    # TITLE WITH THE METHOD NAME
    plt.suptitle(m)
    
    plt.show()
    
    print('\n')
    print('\n')

