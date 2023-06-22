# AutoEncoder based Image Retrieval System

### Dataset Preperation

The CIFAR-10 dataset contains 60000 images 32 x 32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
We used keras.datasets library to load the images. Since we are dealing with color images, each image will have 3 channels (Red,Green & Blue) and each channel will have pixel values in the range of [0 - 255]. To speed up the training and for faster convergence we convert all these values in the image array to be in the range of [0-1]. To achieve this we divide each value in the image array with 255.0. Since in its initial form the values stored in the image array are of type uint8, we have changed it to float32 type before performing the division. Otherwise, the division will result in erroneous results.
