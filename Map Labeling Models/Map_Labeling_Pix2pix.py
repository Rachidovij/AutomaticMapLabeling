from cgi import test
import tensorflow as tf

import os
import pathlib
import time
import datetime

from tensorflow.keras import backend
from matplotlib import pyplot as plt
from tensorflow.keras import layers as ly



PATH = "Path to the project folder"
BUFFER_SIZE = 5000
BATCH_SIZE = 1
IMG_WIDTH = 512
IMG_HEIGHT = 512

#Plot image sample
sample_image = tf.io.read_file(str(PATH+'/Data/Background_Images/2/R60C60.jpg'))
sample_image = tf.io.decode_jpeg(sample_image)
print(sample_image.shape)
plt.figure()
plt.imshow(sample_image)
plt.show()


# Definition of some util functions, such as normalization of the images between [-1, 1] and random crop. Random jitter can be also tested
def normalize(input_image):
    input_image = (input_image / 127.5) - 1

    return input_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


# Read the image files as uint8 tensor, then convert them to float32 tensors
def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)
    input_image = tf.cast(image, tf.float32)

    return input_image

# inp, re = load(str(PATH / 'train/100.jpg'))
# # Casting to int for matplotlib to display the images
# plt.figure()
# plt.imshow(inp / 255.0)
# plt.figure()
# plt.imshow(re / 255.0)


#Load and Normalize the train images
def load_image_train(image_file):
    input_image = load(image_file)
    input_image = normalize(input_image)
    return input_image

#Load and Normalize the test images
def load_image_test(image_file):
    input_image = load(image_file)
    input_image = normalize(input_image)
    return input_image



inp = load(str(PATH+'/Data/Background_Images/2/R60C60.jpg'))
re = load(str(PATH+'/Data/Ground_Truth/2/R60C60.jpg'))
# Casting to int for matplotlib to display the images
plt.figure()
plt.imshow(inp / 255.0)
plt.figure()
plt.imshow(re / 255.0)

def data_train_pipline1():
    train_dataset_labels = tf.data.Dataset.list_files(str(PATH+'/Data/Ground_Truth/2/*.jpg'), shuffle=False)
    # for j, i in enumerate(train_dataset_labels):
    #     print(i)
    #     if j==20:
    #         break
    train_dataset_labels = train_dataset_labels.map(load_image_train,
                                    num_parallel_calls=tf.data.AUTOTUNE)

    
    return train_dataset_labels

def data_train_pipline():
    train_dataset_input = tf.data.Dataset.list_files(str(PATH+'/Data/Background_Images/2/*.jpg'), shuffle=False)

    train_dataset_input = train_dataset_input.map(load_image_train,
                                    num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset_labels = tf.data.Dataset.list_files(str(PATH+'/Data/Ground_Truth/2/*.jpg'), shuffle=False)

    train_dataset_labels = train_dataset_labels.map(load_image_train,
                                    num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = tf.data.Dataset.zip((train_dataset_input, train_dataset_labels))  #.map(lambda x, y: tf.concat((x, y), axis=0))


    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    return train_dataset

train_dataset = data_train_pipline()

#example_input, example_target = next(iter(train_dataset.take(1)))

#i = 0
#for element in train_dataset:
#    i = i + 1
#    print(i)

#print(example_input, example_target)

def data_test_pipline():
    test_dataset_input = tf.data.Dataset.list_files(str(PATH +'/Data/Background_Images/2/*.jpg'), shuffle=False)
    test_dataset_labels = tf.data.Dataset.list_files(str(PATH +'/Data/Ground_Truth/2/*.jpg'), shuffle=False)
    test_dataset_input = test_dataset_input.map(load_image_test)
    test_dataset_labels = test_dataset_labels.map(load_image_test)

    test_dataset = tf.data.Dataset.zip((test_dataset_input, test_dataset_labels))   #.map(lambda x, y: tf.concat((x, y), axis=0))
    test_dataset = test_dataset.batch(BATCH_SIZE)
    return test_dataset


test_dataset = data_test_pipline()


OUTPUT_CHANNELS = 3


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    ly.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(ly.BatchNormalization())

    result.add(ly.LeakyReLU())

    return result

down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print (down_result.shape)

def trans_attention(channels, attention_input):
    batch_size, height, width, channels_num = attention_input.get_shape().as_list()
    f = ly.Conv2D(channels // 8, kernel_size=1, strides=1, )(attention_input)
    g = ly.Conv2D(channels // 8, kernel_size=1, strides=1, )(attention_input)
    h = ly.Conv2D(channels // 2, kernel_size=1, strides=1, )(attention_input)

    dk = tf.cast(tf.shape(g)[-1], tf.float32)

    s = tf.matmul(ly.Reshape(( -1, x.shape[-2]* x.shape[-1]))(g), ly.Reshape(( -1, x.shape[-2]* x.shape[-1]))(f), transpose_b=True)/tf.math.sqrt(dk)
    att = tf.nn.softmax(s) 

    o = tf.matmul(att, ly.Reshape(( -1, x.shape[-2]* x.shape[-1]))(h), transpose_a=True) 
    gamma = 0.002 
    o = ly.Reshape((height, width, channels_num//2))(o) 
    o = ly.Conv2D(channels, kernel_size=1, strides=1, activation='relu')(o)
    out = tf.keras.ly.Dropout(0.1)(o)
    out = tf.keras.ly.LayerNormalization(epsilon=1e-6)(gamma * out + x) 
    return out

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    ly.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(ly.BatchNormalization())

    if apply_dropout:
        result.add(ly.Dropout(0.5))

    result.add(ly.ReLU())

    return result


up_model = upsample(3, 4)
up_result = up_model(down_result)
print (up_result.shape)


def Generator():
    inputs = ly.Input(shape=[512, 512, 3])

    down_stack = [
    downsample(64, 4, apply_batchnorm=False),  
    downsample(128, 4),  
    downsample(512, 4),  
    downsample(512, 4),  
    downsample(512, 4),  
    downsample(512, 4),  
    downsample(512, 4),  
    downsample(512, 4),  
    ]

    up_stack = [
    upsample(512, 4, apply_dropout=True),  
    upsample(512, 4, apply_dropout=True),  
    upsample(512, 4, apply_dropout=True),  
    upsample(512, 4),  
    upsample(512, 4),  
    upsample(128, 4),  
    upsample(64, 4),  
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = ly.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh') 

    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    x = trans_attention(x.get_shape().as_list()[0],x)

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = ly.Concatenate()([x, skip])
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


generator = Generator()


LAMBDA = 100
loss_fct = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_fct(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss




def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = ly.Input(shape=[512, 512, 3], name='input_image')
    tar = ly.Input(shape=[512, 512, 3], name='target_image')

    x = ly.concatenate([inp, tar])  

    out_1 = downsample(64, 4, False)(x)  
    out_2 = downsample(128, 4)(out_1)  
    out_3 = downsample(512, 4)(out_2)  

    pad_1 = ly.ZeroPadding2D()(out_3)  
    out_conv = ly.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(pad_1)  

    batchnorm_1 = ly.BatchNormalization()(out_conv)

    out_relu = ly.LeakyReLU()(batchnorm_1)

    pad_2 = ly.ZeroPadding2D()(out_relu) 

    output = ly.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) 

    return tf.keras.Model(inputs=[inp, tar], outputs=output)


discriminator = Discriminator()


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_fct(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_fct(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return tf.keras.backend.mean(y_true * y_pred)


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

#The RMSprop can be tested. In addition, wasserstein_loss can also be used. 
# opt = tf.keras.optimizers.RMSprop(lr=0.00005)
# tf.keras.optimizers.model.compile(loss=wasserstein_loss, optimizer=opt)

#Determine the location whete to save the training checkpoints of the model
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# Restore the latest checkpoint saved in the checkpoint directory
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    fig = plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
            
        plt.close()
    fig.savefig(PATH+"/Predictions/img_1.png")
    # plt.show()     #can be uncommented to plot the predictions


#The metric values such as losses can be saved in log files
log_dir="logs/"
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
    
    if (step+1) % 10000 == 0:
        old_values = tf.io.read_file("ResultsPix.txt")
        one_string = tf.strings.format("{}   {}  {}   {}  {}  ", (old_values,gen_total_loss,gen_gan_loss,gen_l1_loss,disc_loss))
        tf.io.write_file("ResultsPix.txt", one_string)
        
        print("total_gen_g_loss", gen_total_loss)
        print(" total_gen_f_loss", gen_gan_loss )
        print(" disc_x_loss", gen_l1_loss )
        print(" disc_y_loss", disc_loss )

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


def fit(train_ds, test_ds, steps):
    #example_input, example_target = next(iter(test_ds.take(1)))
    #start = time.time()

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 30000 == 0:
            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()

        train_step(input_image, target, step)

        # Training step
        if (step+1) % 4000 == 0:
            print('.', end='', flush=True)


        # Save the model checkpoint every 100k steps
        if (step+1) % 100000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


# %load_ext tensorboard
# %tensorboard --logdir {log_dir}

#Train the model for 2M steps. To train the model on a specific number of epochs (200), train method can be used. 
fit(train_dataset, test_dataset, steps=2000000) 
#train(train_dataset, 200)
