
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def display(img, cmap = 'gray'):
    fig = plt.figure(figsize = (12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap = 'gray')


# In[3]:


reeses = cv2.imread('reeses_puffs.png', 0)
display(reeses)


# In[5]:


cereals = cv2.imread('many_cereals.jpg', 0)
display(cereals)


# In[6]:


orb = cv2.ORB_create()


# In[14]:


kp1, des1 = orb.detectAndCompute(reeses, None)
kp2, des2 = orb.detectAndCompute(cereals, None)


# In[15]:


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)


# In[16]:


matches = bf.match(des1, des2)


# In[17]:


single_match = matches[0]
single_match.distance


# In[19]:


matches = sorted(matches, key = lambda x:x.distance)


# In[20]:


reeses_matches = cv2.drawMatches(reeses, kp1, cereals, kp2, matches[:25], None, flags = 2)


# In[21]:


display(reeses_matches)


# In[22]:


sift = cv2.xfeatures2d.SIFT_create()


# In[23]:


kp1, des1 = sift.detectAndCompute(reeses, None)
kp2, des2 = sift.detectAndCompute(cereals, None)


# In[26]:


bf = cv2.BFMatcher()


# In[27]:


matches = bf.knnMatch(des1, des2, k = 2)


# In[28]:


matches


# In[29]:


good = []

for match1, match2 in matches:
    # IF MATCH1 DIST IS 75% LESS THAN MATCH2 DIST
    # RATIO TEST
    if match1.distance < 0.75 * match2.distance:
        good.append([match1])


# In[30]:


good


# In[31]:


len(good)


# In[32]:


len(matches)


# In[33]:


sift_matches = cv2.drawMatchesKnn(reeses, kp1, cereals, kp2, good, None, flags = 2)


# In[34]:


display(sift_matches)


# In[35]:


sift = cv2.xfeatures2d.SIFT_create()


# In[36]:


kp1, des1 = sift.detectAndCompute(reeses, None)
kp2, des2 = sift.detectAndCompute(cereals, None)


# In[37]:


# FLANN - FAST LIBRARY FOR APROX NEAREST NEIGHBOURS
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)


# In[38]:


flann = cv2.FlannBasedMatcher(index_params, search_params)


# In[39]:


matched = flann.knnMatch(des1, des2, k = 2)


# In[40]:


good = []

for match1, match2 in matches:
     if match1.distance < 0.7 * match2.distance:
        good.append([match1])


# In[41]:


flann_matches = cv2.drawMatchesKnn(reeses, kp1, cereals, kp2, good, None, flags = 0)


# In[42]:


display(flann_matches)


# In[43]:


flann_matches = cv2.drawMatchesKnn(reeses, kp1, cereals, kp2, good, None, flags = 2)
display(flann_matches)


# In[44]:


sift = cv2.xfeatures2d.SIFT_create()


# In[45]:


kp1, des1 = sift.detectAndCompute(reeses, None)
kp2, des2 = sift.detectAndCompute(cereals, None)


# In[46]:


# FLANN - FAST LIBRARY FOR APROX NEAREST NEIGHBOURS
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)


# In[47]:


flann = cv2.FlannBasedMatcher(index_params, search_params)


# In[48]:


matched = flann.knnMatch(des1, des2, k = 2)


# In[49]:


matchesMask = [[0, 0] for i in range(len(matches))]


# In[50]:


matchesMask


# In[51]:


good = []

for i, (match1, match2) in enumerate(matches):
     if match1.distance < 0.7 * match2.distance:
            matchesMask[i] = [1, 0]


# In[52]:


draw_params = dict(matchColor = (0, 255, 0), singlePointColor = (255, 0, 0), matchesMask = matchesMask, flags = 0)


# In[53]:


flann_matches = cv2.drawMatchesKnn(reeses, kp1, cereals, kp2, matches, None, **draw_params)
display(flann_matches)

