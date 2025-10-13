
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


mylist = [1,2,3]


# In[3]:


type(mylist)


# In[6]:


myarray = np.array(mylist)


# In[7]:


myarray


# In[8]:


type(myarray)


# In[14]:


np.arange(0,10,2) # Creates integer as default


# In[15]:


np.zeros(shape=(10,5)) # Creates float as default


# In[17]:


np.ones(shape=(2,4)) # (rows,col)


# In[18]:


np.random.seed(101) # random seed number 101
arr = np.random.randint(0,100,10) # between 0 and 100 and 10 random numbers


# In[19]:


arr


# In[20]:


arr2 = np.random.randint(0,100,10)


# In[21]:


arr2


# In[22]:


arr.max()


# In[24]:


arr.argmax() # index value


# In[25]:


arr.min()


# In[26]:


arr.argmin()


# In[27]:


arr.mean()


# In[28]:


arr.shape


# In[29]:


arr.reshape((2,5))


# In[31]:


mat = np.arange(0,100).reshape(10,10)


# In[32]:


mat


# In[35]:


row = 0
col = 1


# In[37]:


mat[row,col]


# In[38]:


mat[4,6]


# In[39]:


mat[:,1]


# In[40]:


mat


# In[44]:


mat[:,1].reshape(1,10)


# In[45]:


mat[:,1].reshape(10,1)


# In[46]:


mat[2,:]


# In[47]:


mat


# In[49]:


mat[0:3,0:4] = 0


# In[51]:


newmat = mat.copy()


# In[52]:


newmat


# In[53]:


newmat[0:6,0:6] = 999


# In[54]:


newmat

