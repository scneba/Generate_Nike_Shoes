
# coding: utf-8

#relevant imports
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from glob import glob
from functions import *
import cv2
import argparse



def run(imagepath,batch_size,epochs,print_after, show_after):
    print("Image path: {}\n".format(imagepath))
    print("Batch size: {}\n".format(batch_size))
    print("Number of Epochs: {}\n".format(epochs))
    print("Print loss after: {}\n".format(print_after))
    print("Show generated Images after: {}\n".format(show_after))
    image_size = (128,128,3)
    fake_size = 32*32
    learning_rate = 0.0002
    batch_size = batch_size
    epochs = epochs
    alpha = 0.2
    beta1 = 0.5
    output_dim=3
    dropout_rate = 0.5

    #defined placeholders  for real and face inputs
    real_input = tf.placeholder(tf.float32, (None, *image_size), name='real_input')
    fake_input = tf.placeholder(tf.float32, (None, fake_size), name='fake_input')

    #run loss and optimization
    g_loss, d_loss, g_train_opt, d_train_opt = loss_and_optimization(real_input=real_input,
    fake_input=fake_input, learning_rate=learning_rate, dropout_rate=dropout_rate, alpha=alpha,beta1=beta1)

    #load images from path
    input_images_path =glob("{}*".format(imagepath))

    #image genrator definition
    image_gen = image_generator(all_image_path=input_images_path, output_size = image_size,batch_size=batch_size)

    saver = tf.train.Saver()
    fake_samples = np.random.uniform(-1, 1, size=(20, fake_size))

    count=0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for i in range(int(len(input_images_path)/batch_size)):
                image_batch = next(image_gen)
                count += 1
                print("step number: {}".format(count))

                # Sample random noise for G
                fake_batch = np.random.uniform(-1, 1, size=(batch_size, fake_size))

                # Run optimizers
                sess.run(d_train_opt, feed_dict={real_input: image_batch,fake_input: fake_batch})
                sess.run(g_train_opt, feed_dict={fake_input: fake_batch})

                if count % print_after == 0:
                    #print loss for every 5 counts
                    train_loss_d = d_loss.eval({real_input: image_batch,fake_input: fake_batch})
                    train_loss_g = g_loss.eval({fake_input: fake_batch})

                    print("Epoch {}/{}...".format(epoch+1, epochs),
                          "Discriminator Loss: {:.5f}...".format(train_loss_d),
                          "Generator Loss: {:.5f}".format(train_loss_g))
                if count % show_after== 0:
                    #generate some images after 20 counts
                    samples_generated = sess.run( generator(fake_input, reuse=True,dropout_rate=dropout_rate,alpha=alpha)
                                          ,feed_dict={fake_input: fake_samples})
                    show_images(samples_generated,5)
                    plt.show()
        saver.save(sess, './generator.ckpt')




    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        test_samples = sess.run( generator(fake_input, reuse=True,dropout_rate=dropout_rate,alpha=alpha)
                               ,feed_dict={fake_input: fake_samples})
        show_images(test_samples,5)
        



def main():
    parser = argparse.ArgumentParser(description='Run a GAN model to generate images of size 128x128x3.')
    parser.add_argument(
        'images_path',
        type=str,
        default='',
        help='Path to folder containing images. The model will be ran on the images in this folder. Make sure all images have same incremental name. eg image1.jpg,image2.jpg, image3.jpg so they can be compartible with globe'
    )
    parser.add_argument(
        '--b',
        type=int,
        default=20,
        help='Batch size: default=20.')
    parser.add_argument(
        '--e',
        type=int,
        default=100,
        help='Number of Epochs: default=50.')
    
    parser.add_argument(
        '--s',
        type=int,
        default=5,
        help='Number of Steps to run before showing Losses for discriminator and generator: default=3')
        
        
    parser.add_argument(
        '--d',
        type=int,
        default=20,
        help='Number of Steps to run before generating samples from generator: default=10')
        
        
    args = parser.parse_args()

    #run algorithm
    run(args.images_path,args.b, args.e, args.s,args.d)


if __name__ == '__main__':
    main()


