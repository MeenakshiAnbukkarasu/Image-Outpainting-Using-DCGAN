
# ### Utilities
# 1. Save Model
# 2. Load Model
# 3. Save Image
# 4. Save Log

import os
import numpy as np

import cv2
from keras.models import Model,model_from_json
from mask_demask import masked_images,demask_images,mask_width
from keras.optimizers import Adam
from keras.layers import Input
import PIL
import IPython.display
from dataloader import Data

check_point = "model_check/"
trained_images = "trained_images/"

data = Data()


DISCRM_OPTIMIZER = Adam(0.0001, 0.5)
GENER_OPTIMIZER = Adam(0.001, 0.5)
SHAPE = (256, 256, 3)
MASK_PERCENT = .25
gener_input_shape = (SHAPE[0], int(SHAPE[1] * (MASK_PERCENT *2)), SHAPE[2])


def save_model():
    
    global DCRM, GEN

    models = [DCRM, GEN]
    model_names = ['DCRM','GEN']

    for model, model_name in zip(models, model_names):
        model_path =  check_point + "%s.json" % model_name
        weights_path = check_point + "/%s.hdf5" % model_name
        options = {"file_arch": model_path, 
                    "file_weight": weights_path}
        json_string = model.to_json()
        open(options['file_arch'], 'w').write(json_string)
        model.save_weights(options['file_weight'])
    print("Saved Model")
    
    
def load_model():
    # Checking if all the model exists
    model_names = ['DCRM', 'GEN']
    files = os.listdir(check_point)
    for model_name in model_names:
        if model_name+".json" not in files or model_name+".hdf5" not in files:
            print("Models not Found")
            return
    global DCRM, GEN, COMBINED, IMAGE, GENERATED_IMAGE, CONF_GENERATED_IMAGE
    
    # load DCRM Model
    model_path = check_point + "%s.json" % 'DCRM'
    weight_path = check_point + "%s.hdf5" % 'DCRM'
    with open(model_path, 'r') as f:
        DCRM = model_from_json(f.read())
    DCRM.load_weights(weight_path)
    DCRM.compile(loss='mse', optimizer=DISCRM_OPTIMIZER)
    
    #load GEN Model
    model_path = check_point + "%s.json" % 'GEN'
    weight_path = check_point + "%s.hdf5" % 'GEN'
    with open(model_path, 'r') as f:
         GEN = model_from_json(f.read())
    GEN.load_weights(weight_path)
    
    # Combined Model
    DCRM.trainable = False
    IMAGE = Input(shape=gener_input_shape)
    GENERATED_IMAGE = GEN(IMAGE)
    CONF_GENERATED_IMAGE = DCRM(GENERATED_IMAGE)

    COMBINED = Model(IMAGE, [CONF_GENERATED_IMAGE, GENERATED_IMAGE])
    COMBINED.compile(loss=['mse', 'mse'], optimizer=GENER_OPTIMIZER)
    
    print("loaded model")
    
    
def save_image(epoch, steps):
    original = data.get_data(1)
    if original is None:
        original = data.get_data(1)
    
    mask_image_original , missing_image = masked_images(original)
    mask_image = mask_image_original.copy()
    mask_image = mask_image / 127.5 - 1
    missing_image = missing_image / 127.5 - 1
    gen_missing = GEN.predict(mask_image)
    gen_missing = (gen_missing + 1) * 127.5
    gen_missing = gen_missing.astype(np.uint8)
    demask_image = demask_images(mask_image_original, gen_missing)
    
    mask_image = (mask_image + 1) * 127.5
    mask_image = mask_image.astype(np.uint8)

    border = np.ones([original[0].shape[0], 10, 3]).astype(np.uint8)
    
    file_name = str(epoch) + "_" + str(steps) + ".jpg"
    final_image = np.concatenate((border, original[0],border,mask_image_original[0],border, demask_image[0], border), axis=1)
    cv2.imwrite(os.path.join(trained_images, file_name), final_image)
    print("\t1.Original image \t 2.Input \t\t 3. Output")
    IPython.display.display(PIL.Image.fromarray(final_image))
    print("image saved")


def save_log(log):
    with open('log.txt', 'a') as f:
        f.write("%s\n"%log)
