
# coding: utf-8

# In[2]:


import numpy as np
from numpy import genfromtxt


# In[3]:


data = genfromtxt('bank_note_data.txt', delimiter = ',')


# In[4]:


data


# In[5]:


labels = data[:, 4]


# In[7]:


labels


# In[8]:


features = data[:, 0 : 4]


# In[9]:


features


# In[10]:


X = features
y = labels


# In[11]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[14]:


X_train


# In[15]:


len(X_train)


# In[16]:


len(X)


# In[17]:


len(X_test)


# In[18]:


len(y_test)


# In[19]:


len(y_train)


# In[20]:


from sklearn.preprocessing import MinMaxScaler


# In[21]:


scaler = MinMaxScaler()


# In[22]:


scaler.fit(X_train)


# In[23]:


scaled_X_train = scaler.transform(X_train)


# In[24]:


scaled_X_test = scaler.transform(X_test)


# In[25]:


scaled_X_train.max()


# In[26]:


X_train


# In[27]:


scaled_X_train


# In[28]:


from keras.models import Sequential
from keras.layers import Dense


# In[29]:


model = Sequential()

model.add(Dense(4, input_dim = 4, activation = 'relu'))

model.add(Dense(8, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))


# In[30]:


model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[31]:


model.fit(scaled_X_train, y_train, epochs = 50, verbose = 2)


# In[32]:


model.predict_classes(scaled_X_test)


# In[33]:


model.metrics_names


# In[34]:


from sklearn.metrics import confusion_matrix, classification_report


# In[35]:


predictions = model.predict_classes(scaled_X_test)


# In[36]:


confusion_matrix(y_test, predictions)


# In[37]:


print(classification_report(y_test, predictions))


# In[38]:


model.save('mysupermodel.h5')


# In[39]:


from keras.models import load_model


# In[40]:


newmodel = load_model('mysupermodel.h5')


# In[41]:


newmodel.predict_classes(scaled_X_test)

