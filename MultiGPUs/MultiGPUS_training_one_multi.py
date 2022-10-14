import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from datetime import datetime

""" 
Synchronous distributed training on multiple GPUs on one machine (node)
"""

OUTPUT_CHANNELS = 3


def downsample(filters, kernel_size, apply_batchnorm=True, apply_dropout=False, pooling=False):

  """ 
  Downsampling function
  """
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv3D(filters, kernel_size=kernel_size, strides=(2,2,2), padding='same',
                             kernel_initializer=initializer, use_bias=False))


  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.3))

  result.add(tf.keras.layers.LeakyReLU())
  
  if pooling: # If pooling is applied, strides in Conv3D layers need to change to (1,1,1)
    result.add(tf.keras.layers.MaxPool3D(pool_size=(2,2,2), strides=(2,2,2), padding='same')) # Not recommend to use max-pooling 

  return result

def upsample(filters, kernel_size, apply_dropout=False, apply_batchnorm=True):

  """
  Upsampling function 
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv3DTranspose(filters, kernel_size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))


  if apply_batchnorm:
      result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.3))

  result.add(tf.keras.layers.LeakyReLU())


  return result

# Parameters for the two model we want to build
learning_rate1=0.001
learning_rate2=0.01

optimizer1 = tf.keras.optimizers.Adam(learning_rate=learning_rate1)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=learning_rate2)

batch_size1 = 8
batch_size2 = 8

kernel_size1 = 3
kernel_size2 = 3

filter_base1 = 16
filter_base2 = 32
filter_base3 = 64


self_defined_loss = False # To get the root mean square error

if self_defined_loss:

  def loss_func(y_true, y_pred): 
      
      # Root mean square loss
      squared_difference = tf.square(y_true - y_pred)
      mean_square_loss = tf.reduce_mean(squared_difference, axis=-1)
      root_mean_square_loss = tf.sqrt(mean_square_loss + 1e-20)
      
      return root_mean_square_loss

else:

  loss_func = 'mean_squared_error'


# Synchronous distributed training on multiple GPUs on one machine with data parallelism strategy.
mir_strategy = tf.distribute.MirroredStrategy()

with mir_strategy.scope():
  def architecture1(filter_base=filter_base3, kernel_size=kernel_size1):

    """ 
    Final architecture
    """

    down_stack = [
      downsample(filter_base, kernel_size), 
      downsample(filter_base*2, kernel_size),
      downsample(filter_base*4, kernel_size), 
      downsample(filter_base*8, kernel_size), 
      downsample(filter_base*16, kernel_size),
      downsample(filter_base*32, kernel_size), #change filters to suit the computation capability

    ]

    up_stack = [
      upsample(filter_base*16, kernel_size), 
      upsample(filter_base*8, kernel_size),
      upsample(filter_base*4, kernel_size),
      upsample(filter_base*2, kernel_size), 
      upsample(filter_base, kernel_size),

      # upsample(filter_base*32, kernel_size, apply_dropout=False), 
      # upsample(filter_base*16, kernel_size, apply_dropout=False),
      # upsample(filter_base*8, kernel_size, apply_dropout=False),
      # upsample(filter_base*4, kernel_size), 
      # upsample(filter_base*2, kernel_size),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    concat = tf.keras.layers.Concatenate()

    # Making three outputs
    # Ouput Axial component
    last1 = tf.keras.layers.Conv3DTranspose(OUTPUT_CHANNELS, kernel_size,
                                          strides=2,
                                          padding='same',
                                          kernel_initializer=initializer,
                                          activation=None) 
    

    inputs = tf.keras.layers.Input(shape=[None,None,None,3])
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
      x = down(x)
      skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      x = concat([x, skip])

    # Defining the outputs
    x = last1(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)


  model2 = architecture1()
  model2.compile(loss=loss_func, optimizer=optimizer1)



# Adjusting number of batch size
#batch_per_replicas = 16
global_batch_size_rate = mir_strategy.num_replicas_in_sync

# Loading datasets 
# To do: Change the folder containing the datasets as required

path_train = "/clusterdata/uqvngu19/scratch/Objective3Datasets/Combined_train" 
path_val = "/clusterdata/uqvngu19/scratch/Objective3Datasets/Combined_train_val"
path_test = "/clusterdata/uqvngu19/scratch/Objective3Datasets/Combined_test"
path_out = "/clusterdata/uqvngu19/scratch/Objective3Datasets/Cone_3Com_out_train"



# Get one element
def parse_tfr_element(element):
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = {
      
      'raw_image' : tf.io.FixedLenFeature([], tf.string),
      'raw_label' : tf.io.FixedLenFeature([], tf.string)

    }

  content = tf.io.parse_single_example(element, data)
 
  raw_image = content['raw_image']
  raw_label = content['raw_label']

  
  #get our 'feature'-- our image -- and reshape it appropriately
  feature = tf.io.parse_tensor(raw_image, out_type=tf.float64)
  #feature = tf.reshape(feature, shape=[64, 64, 64, 3])

  label = tf.io.parse_tensor(raw_label, out_type=tf.float64)
  #label = tf.reshape(label, shape=[64, 64, 64, 3])
  

  return (feature, label)

def parse_tfr_element_aug(element):
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = {
      
      'raw_image' : tf.io.FixedLenFeature([], tf.string),
      'raw_label' : tf.io.FixedLenFeature([], tf.string)

    }

  content = tf.io.parse_single_example(element, data)
 
  raw_image = content['raw_image']
  raw_label = content['raw_label']

  
  #get our 'feature'-- our image -- and reshape it appropriately
  feature = tf.io.parse_tensor(raw_image, out_type=tf.float64)
  #feature = tf.reshape(feature, shape=[64, 64, 64, 3])

  label = tf.io.parse_tensor(raw_label, out_type=tf.float64)
  #label = tf.reshape(label, shape=[64, 64, 64, 3])

  return (feature, label)

# Get batch of data

def get_dataset(filename, batch_size, count, data_aug=False):
  #create the dataset
  dataset = tf.data.TFRecordDataset(filename)

  #pass every single feature through our mapping function
  if data_aug:
    dataset = dataset.map(parse_tfr_element_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  else:
    dataset = dataset.map(parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  #dataset = dataset.snapshot(snap_path) #no need at the moment :)
  dataset = dataset.shuffle(buffer_size=round(3*count)+1)
  #dataset = dataset.shuffle(buffer_size=round(3*count)+1)
  #dataset = dataset.shuffle(buffer_size=count+1)

  dataset = dataset.batch(batch_size, drop_remainder=False)
  #dataset = dataset.repeat()
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
  return dataset


# Loading training data
if os.getcwd() != path_train:
  os.chdir(path_train)

with open('total_num', 'rb') as total:
  count = pickle.load(total)

# Part 1
filename_1 = path_train + "/train_images_scaled_3com.tfrecords_w_f" #"/train_images_scaled_3com.tfrecords_w_f" 
filename_2 = path_train + "/2_train_images_scaled_3com.tfrecords_w_f" 
filename_3 = path_train + "/3_train_images_scaled_3com.tfrecords_w_f" 

filename = [filename_1, filename_2, filename_3]

dataset_train = get_dataset(filename, batch_size=16*global_batch_size_rate, count=count, data_aug=False) # count is the total number of the objects; batch size should be 100 or more


# Loading validation dataset

if os.getcwd() != path_val:
  os.chdir(path_val)
  

# Loading data

with open('total_num', 'rb') as total:
  count = pickle.load(total)

filename = path_val + "/train_val_images_scaled_3com.tfrecords_w_f" # _4 for the training dataset and _1 for the validation

dataset_train_val = get_dataset(filename, batch_size=32*global_batch_size_rate, count=count) # count is the total number of the objects; batch_size should be 50 or more



#%load_ext tensorboard

now = datetime.now()

source_file = "/clusterdata/uqvngu19/scratch/Objective3Model_nor_64" # To do: Change the save model folder as required

source_file_model = os.path.join(source_file, 'saved_model_nograd')

if "saved_model_nograd" not in os.listdir(source_file):
  os.mkdir(source_file_model)

if os.getcwd != source_file_model:
  os.chdir(source_file_model)

filepath = os.path.join(os.getcwd(), 'V-Net-4') # 1 indicates dataset with 1

modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss', 
                                                     mode='min', verbose=1, save_best_only=True, save_weights_only=True) # change monitor to  val_loss; 
                                                     #current training monitor = 'loss'

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
verbose=1, mode='min', patience=15, min_delta=0.00005, restore_best_weights=False) #original patience = 20

new_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', verbose=1, patience=10, 
                                              factor=1/10, min_lr=1e-5)



callbacks = [modelcheckpoint, earlystopping, new_lr]

import_model = False

if import_model:
  model2.load_weights(filepath)

st = time.time()
#trial training without callbacks
history_1 = model2.fit(dataset_train, validation_data=dataset_train_val, 
                     epochs=10000, callbacks=callbacks)

used_time_1 = time.time() - st

st = time.time()

with open('history-V-Net-3com_1', 'wb') as filename:
  pickle.dump(history_1.history, filename)

with open('used_time-V-Net-3com_1', 'wb') as filename:
  pickle.dump(used_time_1, filename)
