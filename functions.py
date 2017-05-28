
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import math
import random

def generator(input, image_depth=3, alpha=0.2, reuse=False, dropout_rate=0.5):
    """Function builds generator model for image generation.
    
    Starts with a one dimensional vector and returns an output equal to the shape of 
    input image to the discriminator
    Args:
        input: numpy array input  to generator
        image_depth: depth of output images (default=3)
        alpha: leaky relu multiplier(default=0.2)
        reuse: boolean value to enable or disable variable reuse in generator
        dropout_rate: drop out rate for dropout layers in generator(default=0.5)
     Returns:
         numpy array: array of 64x64ximage_depth
     
    """
    #make all variables here start with name space 'ns_generator' like ns_generator/weight1
    with tf.variable_scope('ns_generator', reuse=reuse):
        #First connected layer  layer
        con_layer1 = tf.layers.dense(input, 4096)
        dropout1 = tf.layers.dropout(con_layer1,rate=dropout_rate)
        batch_con1 = tf.layers.batch_normalization(dropout1, training=True)
        leaky_con1 = tf.maximum(alpha * batch_con1, batch_con1)
        
        #second connected layer
        con_layer2 = tf.layers.dense(leaky_con1, 8192)
        dropout2 = tf.layers.dropout(con_layer2,rate=dropout_rate)
        batch_con2 = tf.layers.batch_normalization(dropout2, training=True)
        leaky_con2 = tf.maximum(alpha * batch_con2, batch_con2)
        
        #apply reshaping to make it compartible with convolutions transpose
        reshaped_layer = tf.reshape(leaky_con2, (-1, 4, 4, 512))
        
        #input=(4x4x512)
        conv1 = tf.layers.conv2d_transpose(reshaped_layer, filters=256, kernel_size=(5,5), strides=(2,2), padding='same')
        conv_batch = tf.layers.batch_normalization(conv1, training=True)
        conv_leaky_relu1 = tf.maximum(alpha * conv_batch, conv_batch)
        #output  size = 8x8x256 now
        
    
        conv2 = tf.layers.conv2d_transpose(conv_leaky_relu1, filters=128, kernel_size=(5,5), strides=(2,2), padding='same')
        conv_batch2 = tf.layers.batch_normalization(conv2, training=True)
        conv_leaky_relu2 = tf.maximum(alpha * conv_batch2, conv_batch2)
        #output  size = 16x16x128 
        
        #final layer
        conv3 = tf.layers.conv2d_transpose(conv_leaky_relu2, filters=64, kernel_size=(5,5), strides=(2,2), padding='same')
        conv_batch3 = tf.layers.batch_normalization(conv3, training=True)
        conv_leaky_relu3 = tf.maximum(alpha * conv_batch3, conv_batch3)
        #output size= 32x32x64
        
          #final layer
        conv4 = tf.layers.conv2d_transpose(conv_leaky_relu3, filters=32, kernel_size=(5,5), strides=(2,2), padding='same')
        conv_batch4 = tf.layers.batch_normalization(conv4, training=True)
        conv_leaky_relu4 = tf.maximum(alpha * conv_batch4, conv_batch4)
        #output size= 64x64x3
        
        conv5 = tf.layers.conv2d_transpose(conv_leaky_relu4, filters=image_depth, kernel_size=(5,5), strides=(2,2), padding='same')
        output = tf.tanh(conv5)
        #output size= 128x128x3
        return output
    
    
    
    
def discriminator(input, reuse=False, alpha=0.3, dropout_rate=0.5):
    """ Defines a discriminator for the model architecture.
    
    Args:
         input: normalised (-1,1)input array of images with dimenstion(batch_size,64,64,3)
         reuse: boolean to enable or disable parameter reuse
         alpha: leaky relu parameter
         dropout_rate:   drop out rate for dropout layers in generator(default=0.5)
    Returns:
        numpy array of logits: numpy array of dimension(batch_size,1)
    """
    #all variables in this scope will be name "ns_discriminator"
    with tf.variable_scope('ns_discriminator', reuse=reuse):
        # input shape will depend on image shapes supplied
        
        conv1 = tf.layers.conv2d(inputs=input,filters=32,kernel_size=(5,5), strides=(1,1), padding='same')
        leaky_relu1 = tf.maximum(alpha * conv1, conv1)
        
        #want to keep as many features as possible while limiting parameter size
        conv2 = tf.layers.conv2d(inputs=leaky_relu1,filters=64,kernel_size=(5,5), strides=(2,2), padding='same')
        leaky_relu2 = tf.maximum(alpha * conv2, conv2)
        
        conv3 = tf.layers.conv2d(inputs=leaky_relu2,filters=128,kernel_size=(5,5), strides=(2,2), padding='same')
        batch_norm3 = tf.layers.batch_normalization(conv3, training=True)
        leaky_relu3 = tf.maximum(alpha * batch_norm3, batch_norm3)
        
        conv4 = tf.layers.conv2d(inputs=leaky_relu3,filters=256,kernel_size=(5,5), strides=(2,2), padding='same')
        batch_norm4 = tf.layers.batch_normalization(conv4, training=True)
        leaky_relu4 = tf.maximum(alpha * batch_norm4, batch_norm4)
        
        #flatten image data for each image
        flatten_layer = tf.contrib.layers.flatten(leaky_relu4)
        #connected layers
        connected1 = tf.layers.dense(flatten_layer, 1000,name="dens1")
        dropout1 = tf.layers.dropout(connected1,rate=dropout_rate)
        batch_con1 = tf.layers.batch_normalization(dropout1, training=True)
        leaky_con1 = tf.maximum(alpha * batch_con1, batch_con1)
        
        
        connected2 = tf.layers.dense(leaky_con1, 500,name="dens2")
        dropout2 = tf.layers.dropout(connected2,rate=dropout_rate)
        batch_con2 = tf.layers.batch_normalization(dropout2, training=True)
        leaky_con2 = tf.maximum(alpha * batch_con2, batch_con2)
        
        
        connected3 = tf.layers.dense(leaky_con2, 200, name="dens3")
        dropout3 = tf.layers.dropout(connected3,rate=dropout_rate)
        batch_con3 = tf.layers.batch_normalization(dropout3, training=True)
        leaky_con3 = tf.maximum(alpha * batch_con3, batch_con3)
        
        #return a logit of the prediction
        logits = tf.layers.dense(flatten_layer, 1, name="dens4")
        return logits
    
    
    
def loss_and_optimization(real_input, fake_input, image_depth=3, alpha=0.2, beta1=0.5, learning_rate=0.0001, dropout_rate=0.5):
    """Computes the loss and optimization for generator and discriminator. 
    
    Args:
        real_input: normalised(1,-1) tensor   of good input images size(batch_size,64,64,image_depth)
        fake_input: normalised(1,-1)  tensor of fake data input to generator size (batch_size,?)
        image_depth: depth of input images(default=3)
        alpha: leaky relu parameter(default=0.2)
        beta1: adamoptimizer variable(default=0.5)
        learning_rate: learning rate for optimizer(default=0.0001)
        
    Returns:
        discriminator_loss: loss from real input and fake inputs to discriminator
        generator_loss: loss of generator input
        discriminator_optimizer: optimizer for discriminator variables
        generator_optimizer: optimizer for generator variables
        
    
    """
    generator_model = generator(fake_input, image_depth, alpha=alpha, dropout_rate=dropout_rate)
    discrimator_real_logits = discriminator(real_input, alpha=alpha, dropout_rate = dropout_rate)
    
    #reuse discriminator parameters and get loss from generator output
    discrimator_fake_logits = discriminator(generator_model, reuse=True, alpha=alpha)

    discriminator_real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=discrimator_real_logits, labels=tf.ones_like(discrimator_real_logits)))
    
    discriminator_fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=discrimator_fake_logits, labels=tf.zeros_like(discrimator_fake_logits)))
    
    generator_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=discrimator_fake_logits, labels=tf.ones_like(discrimator_fake_logits)))

    #total discriminator loss
    discriminator_loss = discriminator_real_loss + discriminator_fake_loss
    
    # extract the discriminator and generator trainable variables
    all_variables = tf.trainable_variables()
    discriminator_vars = [var for var in all_variables if var.name.startswith('ns_discriminator')]
    generator_vars = [var for var in all_variables if var.name.startswith('ns_generator')]

    # optimize trainable variables
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(discriminator_loss, var_list=discriminator_vars)

    generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(generator_loss, var_list=generator_vars)
    
    return generator_loss, discriminator_loss, generator_optimizer, discriminator_optimizer



def load_image(image_path, output_size):
    """Returns an image of rank 4 given image path
    Args:
        image_path: path of input image
        output_size: tuple of desired image output size 
    
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb,(output_size[0],output_size[1]))
    return_image = np.reshape(image_resized,(1,*output_size))
    return return_image
    
def load_images(image_paths,output_size):
    
    """Returns images of rank 4 given image paths
    Args:
        image_paths: list  of input image paths
        output_size: tuple of desired image output size 
    
    """
    length=len(image_paths)
    return_array = np.empty((length,*output_size))
    for i in range(length):
        return_array[i]=load_image(image_paths[i],output_size)
    return return_array


def image_generator(all_image_path, output_size, batch_size=100, min=-1, max=1):
    """Obtains a batch of  images from a list of image paths
    Args: 
        all_image_path: path to all input images
        batch_size: batch size to generate
        min: minimum normalisaiton value
        max: maximum normalisation value
        output_size: tuple of desired image output size 
        
    """
    num_samples = len(all_image_path)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(all_image_path)
        for offset in range(0, num_samples, batch_size):
            batch_images = load_images(all_image_path[offset:offset+batch_size],output_size=output_size).astype('float32')
            #scale images to zero and one
            #check out http://scikit
            #learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler.fit_transform
            batch_images_std = (batch_images- batch_images.min(axis=0)) / (batch_images.max(axis=0) - batch_images.min(axis=0))
            batch_images_scaled = batch_images_std * (max - min) + min
            yield batch_images_scaled

            
def show_images(img,num_cols):
    """Plot images in columns of four
    Args:
        img: input images to plot
        num_cols: desired number of display columns
    """
    #convert images to RGB-255 for viewing
    img = ((img - img.min())*255 / (img.max() - img.min())).astype(np.uint8)
    #calculate number of rows
    n_rows = math.floor(len(img)/num_cols)
    #plot images in columns of num_cols
    plt.figure(figsize=(num_cols,n_rows))
    for i in range(n_rows):
        for j in range(num_cols):
            image_pos = i*num_cols+j
            plt.subplot(n_rows,num_cols,image_pos+1).imshow(img[image_pos],interpolation='nearest')
            plt.axis('off')
        
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
 