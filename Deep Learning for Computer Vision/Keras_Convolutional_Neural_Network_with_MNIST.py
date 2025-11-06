
# coding: utf-8

# In[2]:


import numpy as np
from numpy import genfromtxt


# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical


# In[4]:


from keras.datasets import mnist


# In[5]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[6]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


x_train.shape


# In[8]:


single_img = x_train[0]


# In[11]:


plt.imshow(single_img, cmap = 'gray_r')


# In[12]:


y_train


# In[14]:


y_train.shape


# In[15]:


from keras.utils.np_utils import to_categorical


# In[16]:


y_cat_test = to_categorical(y_test, 10)
y_cat_train = to_categorical(y_train, 10)


# In[17]:


y_cat_test


# In[18]:


y_cat_train[0]


# In[19]:


single_img.max()


# In[20]:


x_train = x_train / x_train.max()


# In[21]:


x_test = x_test / x_test.max()


# In[22]:


scaled_image = x_train[0]


# In[23]:


scaled_image


# In[25]:


plt.imshow(scaled_image, cmap = 'gray_r')


# In[26]:


x_train.shape


# In[27]:


x_train = x_train.reshape(60000, 28, 28, 1)


# In[28]:


x_train.shape


# In[29]:


x_test = x_test.reshape(10000, 28, 28, 1)


# In[30]:


x_test.shape


# In[31]:


from keras.layers import Dense, Conv2D, MaxPool2D, Flatten


# In[32]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (4, 4), input_shape = (28, 28, 1), activation = 'relu'))

model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])


# In[33]:


model.summary()


# In[34]:


model.fit(x_train, y_cat_train, epochs = 2)


# In[35]:


model.metrics_names


# In[36]:


model.evaluate(x_test, y_cat_test)


# In[37]:


from sklearn.metrics import classification_report


# In[38]:


predictions = model.predict_classes(x_test)


# In[39]:


print(classification_report(y_test, predictions))

