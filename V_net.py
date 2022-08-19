import tensorflow as tf

OUTPUT_CHANNELS = 3



def downsample(filters, kernel_size, apply_batchnorm=True, pooling=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv3D(filters, kernel_size=kernel_size, strides=(2,2,2), padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if pooling:
    result.add(tf.keras.layers.MaxPool3D(pool_size=(2,2,2), strides=(2,2,2), padding='same')) # Not recommend to use max-pooling

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, kernel_size, apply_dropout=False, apply_batchnorm=True):
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
      result.add(tf.keras.layers.Dropout(0.5))

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

#loss1 = 'mean_squared_error'
#loss2 = 'mean_squared_error'

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



epochs_train = 300

save_period = epochs_train


def architecture1(filter_base=filter_base3, kernel_size=kernel_size1):
  down_stack = [
    downsample(filter_base, kernel_size), 
    downsample(filter_base*2, kernel_size),
    downsample(filter_base*4, kernel_size), 
    downsample(filter_base*8, kernel_size), 
    downsample(filter_base*16, kernel_size),
    downsample(filter_base*32, kernel_size), #change filters to suit the computation capability

  ]

  up_stack = [
    upsample(filter_base*16, kernel_size, apply_dropout=False), 
    upsample(filter_base*8, kernel_size, apply_dropout=False),
    upsample(filter_base*4, kernel_size, apply_dropout=False),
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

# model1 = architecture1()
# model1.compile(loss=loss_func, optimizer=optimizer1)
#model1.summary()

# model2 = architecture1()
# model2.compile(loss=loss_func, optimizer=optimizer1)