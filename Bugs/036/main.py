from keras.applications import VGG16
from keras.layers import Input, Dense, Flatten
from keras.models import Model
import random

inp_img_h = random.randint(100, 200)
inp_img_w = random.randint(100, 200)

conv_base = VGG16(weights='imagenet',
              include_top=False,
              input_shape=(inp_img_h, inp_img_w, 3))

def create_functional_model():
   inp = Input(shape=(inp_img_h, inp_img_w, 3))
   model = conv_base(inp)
   model = Flatten()(model)
   model = Dense(256, activation='relu')(model)
   outp = Dense(1, activation='sigmoid')(model)
   return Model(inputs=inp, outputs=outp)

model = create_functional_model()
model.summary()