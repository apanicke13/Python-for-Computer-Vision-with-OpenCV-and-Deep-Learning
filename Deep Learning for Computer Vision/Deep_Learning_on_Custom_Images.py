
# coding: utf-8

# In[10]:


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
import cv2


# In[11]:


cat4 = cv2.imread('CATS_DOGS/CATS_DOGS/train/CAT/4.jpg')


# In[12]:


cat4 = cv2.cvtColor(cat4, cv2.COLOR_BGR2RGB)


# In[13]:


plt.imshow(cat4)


# In[14]:


cat4.shape


# In[15]:


dog = cv2.imread('CATS_DOGS/CATS_DOGS/train/DOG/2.jpg')


# In[16]:


dog = cv2.cvtColor(dog, cv2.COLOR_BGR2RGB)


# In[17]:


plt.imshow(dog)


# In[18]:


dog.shape


# In[19]:


from keras.preprocessing.image import ImageDataGenerator


# In[20]:


image_gen = ImageDataGenerator(rotation_range = 30,
                               width_shift_range = 0.1,
                               height_shift_range = 0.1,
                               rescale = 1 / 255,
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               horizontal_flip = True,
                               fill_mode = 'nearest'
                              )


# In[21]:


plt.imshow(image_gen.random_transform(dog))


# In[22]:


image_gen.flow_from_directory('CATS_DOGS/CATS_DOGS/train')


# In[23]:


from keras.models import Sequential
from keras.layers import Activation, Dropout


# In[24]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (150, 150, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Conv2D(filters = 64, kernel_size = (3, 3), input_shape = (150, 150, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Conv2D(filters = 64, kernel_size = (3, 3), input_shape = (150, 150, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[29]:


input_shape = (150, 150, 3)


# In[25]:


model.summary()


# In[30]:


batch_size = 16

train_image_gen = image_gen.flow_from_directory('CATS_DOGS/CATS_DOGS/train',
                                                target_size = input_shape[:2],
                                                batch_size = batch_size,
                                                class_mode = 'binary')


# In[31]:


test_image_gen = image_gen.flow_from_directory('CATS_DOGS/CATS_DOGS/test',
                                                target_size = input_shape[:2],
                                                batch_size = batch_size,
                                                class_mode = 'binary')


# In[32]:


train_image_gen.class_indices


# In[33]:


results = model.fit_generator(train_image_gen, epochs = 1,
                              steps_per_epoch = 150,
                              validation_data = test_image_gen,
                              validation_steps = 12)


# In[37]:


results.history['acc']


# In[36]:


from keras.models import load_model


# In[40]:


new_model = load_model('cat_dog_100epochs.h5')


# In[93]:


cat_file = 'CATS_DOGS/CATS_DOGS/test/CAT/10001.jpg'


# In[94]:


from keras.preprocessing import image


# In[95]:


cat_file = image.load_img(cat_file, target_size = (150, 150))


# In[103]:


cat_img = image.img_to_array(cat_file)


# In[104]:


cat_img.shape


# In[105]:


import numpy as np
cat_img = np.expand_dims(cat_img, axis = 0)


# In[106]:


cat_img.shape


# In[107]:


cat_img = cat_img / 255


# In[108]:


model.predict_classes(cat_img)


# In[109]:


model.predict(cat_img)

