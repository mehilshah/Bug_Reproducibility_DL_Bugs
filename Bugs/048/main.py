import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_model():
    inp = Input(shape=(561,))
    x = Dense(units=1024,input_dim=561)(inp)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(units=512)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(units=256)(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(units=1, activation='sigmoid')(x)
    
    m = tf.convert_to_tensor(5) #creating a tensor of value = 5
    
    o = Multiply()([x, m]) #trying to multiply x with o. Doesn't work though!

    model = Model(inputs=[inp], outputs=[o])
    
    model.compile(loss='binary_crossentropy', optimizer = Adam(lr=0.0002, beta_1=0.5))
    
    return model

model = create_model()
model.summary()