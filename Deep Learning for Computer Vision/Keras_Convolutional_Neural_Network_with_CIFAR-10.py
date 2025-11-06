
# coding: utf-8

# In[2]:


from keras.datasets import cifar10


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[5]:


x_train.shape


# In[6]:


x_train[0].shape


# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


plt.imshow(x_train[12])


# In[10]:


x_train = x_train / 255
x_test = x_test / 255


# In[11]:


y_train


# In[12]:


y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)


# In[13]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (4, 4), input_shape = (32, 32, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Conv2D(filters = 32, kernel_size = (4, 4), input_shape = (32, 32, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])


# In[14]:


model.summary()


# In[15]:


model.fit(x_train, y_cat_train, verbose = 1, epochs = 10)


# In[16]:


model.evaluate(x_test, y_cat_test)


# In[17]:


predictions = model.predict_classes(x_test)


# In[18]:


print(classification_report(y_test, predictions))

