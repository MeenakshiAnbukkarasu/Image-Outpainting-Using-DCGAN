


from keras.layers import Input, Conv2DTranspose
from keras.layers import ReLU, Dropout, Concatenate, BatchNormalization, Reshape
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers.convolutional import UpSampling2D
import os
import numpy as np
import PIL
import IPython.display
from generator import build_generator,g_build_deconv,g_build_conv
from utilities import save_model,load_model,save_image,save_log
from discriminator import dis_build_conv,build_discriminator
from datetime import datetime
from dataloader import Data
from mask_demask import masked_images,demask_images,mask_width


print("here")

# Initialize dataloader
data = Data()


# Saves Model in every N minutes
TIME_INTERVALS = 1
SHOW_SUMMARY = True

SHAPE = (256, 256, 3)
EPOCHS = 500
BATCH = 1

# ## Models

# ### Discriminator

# In[8]:

DISCRM_OPTIMIZER = Adam(0.0001, 0.5)


# Discriminator initialization
DISCRM = build_discriminator()
DISCRM.compile(loss='mse', optimizer=DISCRM_OPTIMIZER)
if SHOW_SUMMARY:
    DISCRM.summary()


# ### Generator Model

# In[13]:


# 25% i.e 64 width size will be mask from both side
MASK_PERCENTAGE = .25
gener_input_shape = (SHAPE[0], int(SHAPE[1] * (MASK_PERCENTAGE *2)), SHAPE[2])
GENER_OPTIMIZER = Adam(0.001, 0.5)


# Generator Initialization
GENER = build_generator()
GENER.compile(loss='mse', optimizer=GENER_OPTIMIZER)
if SHOW_SUMMARY:
    GENER.summary()


# ### Combined Model

# In[ ]:


IMAGE = Input(shape=gener_input_shape)
DISCRM.trainable = False
GENERATED_IMAGE = GENER(IMAGE)
CONF_GENERATED_IMAGE = DISCRM(GENERATED_IMAGE)

COMBINED = Model(IMAGE, [CONF_GENERATED_IMAGE, GENERATED_IMAGE])
COMBINED.compile(loss=['mse', 'mse'], optimizer=GENER_OPTIMIZER)


# ### Masking and De-Masking

x = data.get_data(1)

# a will be the input and b will be the output for the model
a, b = masked_images(x)
border = np.ones([x[0].shape[0], 10, 3]).astype(np.uint8)
print('After masking')
print('\tOriginal Image\t\t\t a \t\t b')
image = np.concatenate((border, x[0],border,a[0],border, b[0], border), axis=1)
IPython.display.display(PIL.Image.fromarray(image))

print("After desmasking: 'b/2' + a + 'b/2' ")
c = demask_images(a,b)
IPython.display.display(PIL.Image.fromarray(c[0]))



def train():
    start_time = datetime.now()
    saved_time = start_time
    
    global MIN_D_LOSS, MIN_G_LOSS, CURRENT_D_LOSS, CURRENT_G_LOSS
    for epoch in range(1, EPOCHS):
        steps = 1
        test = None
        while True:
            original = data.get_data(BATCH)
            if original is None:
                break
            batch_size = original.shape[0]

            mask_image, missing_image = masked_images(original)
            mask_image = mask_image / 127.5 - 1
            missing_image = missing_image / 127.5 - 1

            # Train Discriminator
            gener_missing = GENER.predict(mask_image)

            real = np.ones([batch_size, 1])
            fake = np.zeros([batch_size, 1])
            
            dis_loss_original = DISCRM.train_on_batch(missing_image, real)
            dis_loss_mask = DISCRM.train_on_batch(gener_missing, fake)
            dis_loss = 0.5 * np.add(dis_loss_original, dis_loss_mask)

            # Train Generator
            for i in range(2):
                gener_loss = COMBINED.train_on_batch(mask_image, [real, missing_image])
                    
            log = "steps: %d, epoch: %d, DISCRIMINATOR loss: %s, GENERATOR loss: %s, Identity loss: %s"    %(steps,epoch, str(dis_loss), str(gener_loss[0]), str(gener_loss[2]))
            print(log)
            save_log(log)
            steps += 1
            
            # Save model if time taken > TIME_INTERVALS
            current_time = datetime.now()
            difference_time = current_time - saved_time
            if difference_time.seconds >= (TIME_INTERVALS * 60):
                save_model()
                save_image(epoch, steps)
                saved_time = current_time

# In[ ]:


load_model()


train()


# ## Recursive paint

# In[14]:

load_model()


# In[15]:


def recursive_paint(image, factor=3):
    final_image = None
    gener_missing = None
    for i in range(factor):
        demask_image = None
        if i == 0:
            x, y = get_masked_images([image])
            gener_missing = GENER.predict(x)
            final_image = get_demask_images(x, gener_missing)[0]
        else:
            gen_missing = GENER.predict(gener_missing)
            final_image = get_demask_images([final_image], gener_missing)[0]
    return final_image
        


# In[18]:


images = data.get_data(1)

for i, image in enumerate(images):
    image = image / 127.5 - 1
    image = recursive_paint(image)
    image = (image + 1) * 127.5
    image = image.astype(np.uint8)
    path = 'recursive/'+str(i)+'.jpg'
    IPython.display.display(PIL.Image.fromarray(image))

