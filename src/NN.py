#19086364d CAOQUN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *
####
import pandas as pd
import time
from datetime import datetime
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

#import data
test = pd.read_csv('./data/test.csv')
train = pd.read_csv('./data/train.csv')
test = test.drop(['id'], axis=1)
train_y = train['target']
train_x = pd.DataFrame(train, columns=[str(i) for i in range(300)])
num_classes = 2
train_one_hot = tf.keras.utils.to_categorical(train_y, num_classes)
x_train=pd.DataFrame.to_numpy(train_x)


#define the model
tf.keras.backend.clear_session()

inputs = keras.Input(shape = (300))
x = layers.Dense(512, activation='relu', kernel_regularizer= regularizers.l2(0.01))(inputs)
x = layers.Dense(256, activation='relu', kernel_regularizer= regularizers.l2(0.01))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(2, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
  loss = 'categorical_crossentropy',
  optimizer = 'SGD',
  metrics = ['accuracy']
)
model.summary()




# Defining the batch size and no. of epochs 
batch_size = 100
epoch = 1000 #of rounds you want to train


es = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0,
    patience=300,
    verbose=1,
    mode='auto',
    restore_best_weights=True
)

# Starting the training
hist = model.fit(x_train, train_one_hot, epochs=epoch, batch_size = batch_size,
                 validation_split=0.3, callbacks=[es])#validation set => validation set to top train

#make prediction
prediction2 = model.predict(test)
# get submission
sub = pd.read_csv('./data/sample_submission.csv')
sub['target'] = prediction2[:,1]
sub.to_csv('./out/submission-'+str(time.time())+'.csv',index=False) 