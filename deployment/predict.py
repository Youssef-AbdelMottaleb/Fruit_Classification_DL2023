from keras.preprocessing import image
import os
import numpy as np
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")
import PIL

# loading the model we saved.
model = load_model('model_CNN_saved')

# Creating a dictionary named fruitMap that maps the integer-encoded class labels used in the training dataset to their corresponding fruit names. =
# This is done by iterating over the class_indices attribute of the trainGenerator object, which is a dictionary that maps the string class names to their integer-encoded labels. 
#fruitMap = dict([(v, k) for k, v in trainGenerator.class_indices.items()])
#fruitMap

dic = {0: 'Apple Braeburn',
 1: 'Apple Granny Smith',
 2: 'Apricot',
 3: 'Avocado',
 4: 'Banana',
 5: 'Blueberry',
 6: 'Cactus fruit',
 7: 'Cantaloupe',
 8: 'Cherry',
 9: 'Clementine',
 10: 'Corn',
 11: 'Cucumber Ripe',
 12: 'Grape Blue',
 13: 'Kiwi',
 14: 'Lemon',
 15: 'Limes',
 16: 'Mango',
 17: 'Onion White',
 18: 'Orange',
 19: 'Papaya',
 20: 'Passion Fruit',
 21: 'Peach',
 22: 'Pear',
 23: 'Pepper Green',
 24: 'Pepper Red',
 25: 'Pineapple',
 26: 'Plum',
 27: 'Pomegranate',
 28: 'Potato Red',
 29: 'Raspberry',
 30: 'Strawberry',
 31: 'Tomato',
 32: 'Watermelon'}

# note that the keys and values of the dictionary are swapped

def predict_label(img_path):
    i=image.image_utils.load_img("static/images/" + img_path, target_size=(100,100))
    #i = image.load_img(img_path, target_size=(100,100))
    #i = image.img_to_array(i)/255.0
    i=image.image_utils.img_to_array(i)/255.0
    i = i.reshape(1, 100,100,3)
    p = model.predict(i)
    print(p)
    max_value = np.argmax(p)
    print(max_value)
    return dic[max_value]

#    i=image.image_utils.load_img(img_path,target_size=(230,230))
#   i=image.image_utils.img_to_array(i)
#  test_image = np.expand_dims(i,axis=0)
# test_image = test_image.reshape(1,230,230,3)
#predictions = np.argmax(model.predict(test_image), axis =1)
    #return dic[predictions]