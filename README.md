
##  Image generation using Deep convolutional generative adversary networks example.

The **Deep Convolutional Generative Adversary Network(DCGAN)** is based on the idea of  **Generative Adversary Network(GAN)** first used at University of Montreal by  **Ian J. Goodfellow** and his colleaques. [Here](https://arxiv.org/abs/1406.2661) is the original paper for GANs published in 2014. This repo presents an example of using DCGAN to generate Nike shoes from 100 Nike shoes downloaded from the internet.

### Original Images used
![Example of Images Used](./Capture.jpg)

### File Organisation
- functions.py - This file contains all the functions used in this repo.
- app.ipynb - This is the Ipython notebook originally used to train the network.
- app.ppy - This is the python file which is  an alternative to the Ipython Notebook that can be ran from the command line.
- Shoes - Folder containing shoes used as training data for this model.

#### Usage
- This project can be ran directly from the notebook and this line `input_images_path =glob("Shoes/*")` changed to the path images that will be used to train the network. The `generator` generates images of size `(128x128x3)` which is also the input to the  `discriminator`. In order to change the image size, the `generator` has to be adapted to output the image size and the `image_size = (128,128,3)` in the second cell of the Notebook changed to the corresponding size. Please, endevour to use a squared iamge size to improve performance of model.
- The model can also be ran from the command line. Just type `python app.py -h` to get help on how to run model:
#### python app.py -h
![Example of Images Used](help.jpg)

#### Examples
- **python app.py Shoes/** 
 - Run model on images in path **Shoes/**
 
- **python app.py Shoes/ --b 20 --e 100 --s 5 --d 20** 
 - Path to images **Shoes/**.
 - Batch size: **20**
 - Number of Epochs: **100**
 - Show loss after a count of **5**
 - Display generated images after a count of **20**
 
 
 #### Results
 These are the results obtained using just **100** downloaded Nike shoes. These shoes are included in the repo [here](./Shoes/).
 
![Results1](./results1.jpg)
![Results2](./result2.jpg)





### License
Anybody is free to download and make changes to this project as long as they reference this work.



