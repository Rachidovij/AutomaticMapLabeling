import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt

print(tf.__version__)


AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
PATH = "Path to the project"


# Definition of some util functions, such as normalization of the images between [-1, 1] and random crop. Random jitter can be also tested
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image


def preprocess_image_test(image, label):
    image = normalize(image)
    return image


#Check the access to data
sample_image = tf.io.read_file(str(PATH+'/Data/Background_Images/2/R60C60.jpg'))
sample_image = tf.io.decode_jpeg(sample_image)
print(sample_image.shape)


# Read the image files as uint8 tensor, then convert them to float32 tensors
def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)
    # 
    input_image = tf.cast(image, tf.float32)

    return input_image


#Load and Normalize the train images
def load_image_train(image_file):
    input_image = load(image_file)
    input_image = normalize(input_image)

    return input_image

#Load and normalize the test images
def load_image_test(image_file):
    input_image = load(image_file)
    input_image = normalize(input_image)

    return input_image

#Plot some normalized samples
input = load(str(PATH+'/Data/Background_Images/2/R60C60.jpg'))
real_output = load(str(PATH+'/Data/Ground_Truth/2/R60C60.jpg'))


def data_train_pipline():
    train_dataset_input = tf.data.Dataset.list_files(str(PATH+'/Data/Background_Images/2/*.jpg'), shuffle=False)

    train_dataset_input = train_dataset_input.map(load_image_train,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(BUFFER_SIZE,seed = 42).batch(BATCH_SIZE)
    train_dataset_labels = tf.data.Dataset.list_files(str(PATH+'/Data/Ground_Truth/2/*.jpg'), shuffle=False)
    train_dataset_labels = train_dataset_labels.map(load_image_train,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(BUFFER_SIZE,seed = 42).batch(BATCH_SIZE)
    
    return train_dataset_input, train_dataset_labels

train_dataset_input, train_dataset_labels =  data_train_pipline()

# train_dataset =  (train_dataset_input, train_dataset_labels)
# train_dataset = tf.data.Dataset.zip((train_dataset_input, train_dataset_labels)).shuffle(
#                                         BUFFER_SIZE)
# train_dataset_input, train_dataset_labels = zip(*train_dataset)

def data_test_pipline():
    try:
        test_dataset = tf.data.Dataset.list_files(str(PATH+'/Data/Background_Images/2/*.jpg'))
    except tf.errors.InvalidArgumentError:
        test_dataset = tf.data.Dataset.list_files(str(PATH+'/Data/Background_Images/2/*.jpg'))
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return test_dataset

#test_dataset = data_test_pipline()


sample_map = next(iter(train_dataset_input))
sample_background = next(iter(train_dataset_labels))

#Plot the data to ensure that the model input is as intended
def plot_confirmation():
    sample_map1 = next(iter(train_dataset_input))
    sample_background1 = next(iter(train_dataset_labels))

    plt.figure(figsize=(8, 8))
    contrast = 8

    imgs = [sample_map, sample_background, sample_map1, sample_background1]
    title = ['labeled map', 'To labeled map', 'background', 'To background map']

    for i in range(len(imgs)):
        plt.subplot(2, 2, i+1)
        plt.title(title[i])
        if i % 2 == 0:
            plt.imshow(imgs[i][0] * 0.5 + 0.5)
        else:
            plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
    plt.show()



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


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_fct(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    x = ly.Input(shape=[512, 512, 3], name='input_image') 

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

    return tf.keras.Model(inputs=inp, outputs=output)



OUTPUT_CHANNELS = 3

generator_g = Generator()
generator_f = Generator()

discriminator_x = discriminator()
discriminator_y = discriminator()



LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_image, generated_image):
    real_loss = loss_obj(tf.ones_like(real_image), real_image)

    generated_loss = loss_obj(tf.zeros_like(generated_image), generated_image)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_loss(generated_image):
    return loss_obj(tf.ones_like(generated_image), generated_image)



def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)



checkpoint_path = PATH+"/checkpoints"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')


EPOCHS = 61


def generate_images(model, test_input):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    # plt.show()

#@tf.function
def train_step(real_x, real_y, n):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))
    
    
    if n % 600 == 0:
        print(n)
        old_values = tf.io.read_file("Results.txt")
        one_string = tf.strings.format("{}   {}  {}   {}  {}  ", (old_values,total_gen_g_loss,total_gen_f_loss,disc_x_loss,disc_y_loss))
        tf.io.write_file("Results.txt", one_string)
        
        print("total_gen_g_loss", total_gen_g_loss)
        print(" total_gen_f_loss", total_gen_f_loss )
        print(" disc_x_loss", disc_x_loss )
        print(" disc_y_loss", disc_y_loss )



for epoch in range(EPOCHS):
    print("Training started...")
    tf.io.write_file("Results.txt", "Training started")
    start = time.time()

    n = 0
    for image_x, image_y in tf.data.Dataset.zip((train_dataset_input, train_dataset_labels)):
        train_step(image_x, image_y,n)
        if n % 200 == 0:
            print ('.', end='')
        n += 1

    # save the model each 5 epochs
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                        ckpt_save_path))

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                    time.time()-start))















