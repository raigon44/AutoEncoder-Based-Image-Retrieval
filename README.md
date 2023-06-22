# AutoEncoder based Image Retrieval System

### Dataset Preperation

The CIFAR-10 dataset contains 60000 images 32 x 32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
We used keras.datasets library to load the images. Since we are dealing with color images, each image will have 3 channels (Red,Green & Blue) and each channel will have pixel values in the range of [0 - 255]. To speed up the training and for faster convergence we convert all these values in the image array to be in the range of [0-1]. To achieve this we divide each value in the image array with 255.0. Since in its initial form the values stored in the image array are of type uint8, we have changed it to float32 type before performing the division. Otherwise, the division will result in erroneous results.

Run the **DataPreperation.py** file in this repository to perform the data preperation.

### Building AutoEncoder

#### AutoEncoder

AutoEncoders uses an unsupervised approach for learning a lower-dimensional feature representation from unlabeled training data. The encoder block learns the mapping from the input data to a low-dimensional latent space z. Having a low dimensional latent space allows to compress the data into small latent vector which learns a very compact enrich feature representation. Decoder learns mapping back from latent space z, to a reconstruct the original data.

Autoencoders are trained using the reconstruction loss, where we compare the difference between the orginal image and the reconstructed image from the decoder. The reconstruction loss forces the latent representation to capture as much "information" about the data as possible.

<img src="AE.PNG" alt="AuotEncoder">

We tried different architectures for both the encoder and the decoder before selecting the architecture which we felt gave us the best result. Initially, we tried with just one hidden dense layer. Then we iteratively kept adding more conv2D layers followed by max-pooling layers and kept on checking both the accuracy scores and the results of the sanity check plots. Some architecture gave us slightly better accuracy, but the sanity check results were not good. For instance, we tried an architecture where we stacked 2 conv2D layers and then added a max pooling layer. We added 3 more of these stacked layers in our encoder and retrained the model. Even though this stacked architecture (trained with 25 epochs) gave a better accuracy but the projected latent values had less number of clusters for images belonging to the same data class and image data was more widely distributed across the space. And even when we extracted similar images using the encoder model, the results were not satisfactory. Hence we decided to abandon this architecture. Similarly we tried some different architectures before fixing on the below one(figure 2 and figure 3). Additionally we also used the python library ‘hyperopt’ to fine tune the hyper parameter like number of hidden units in each layer. Since the library was giving the best results for hyperparameters solely based on the metric like accuracy or loss, after a while we stopped using it. This was because we had to also consider the visualization of different sanity checks to choose the best model that captures the most important features in the 10 dimensional latent space. We used the hyper parameters we got while tuning with hyperopt as the base and changed it manually and retrained to see how the change is impacting the results during sanity check.  

The intuition for adding conv2D layers was to capture the local features like edges from the input image. And the intuition for adding the max pooling layer was to downsample the input image so as to get the most important global features from the feature map created by the conv2D layers. It also helped to reduce the size of the input to the next layer.

During the training process, we used different optimizers and loss functions to see which gave us the best result. The ‘adam’ optimizer and the ‘mse’ loss function were giving us the best results. Hence we decided to use them for the final model as well.

We also tried using different activation functions like ‘relu’ and ‘sigmoid’ during the training process. We were getting the best results with relu. And while changing the activation functions during training, we closely monitored the pair plot which plots the distribution of latent space. We got some good distributions while using relu as the activation function for our autoencoder model.

During the training process we also observed that, for most of the models after around 12 epochs, the accuracy reached around 60 and from there the improvement in accuracy was very less.

**BuildingAutoencoder.py** contains the final code for our building and saving the autoencoder model.

#### Training AutoEncoder

Once we had finalized the hyperparameters during the iterative training process, we ran the final training for 100 epochs. Additionally, the two callbacks that we introduced during the training process assisted us in logging and saving the best model. For saving the best model, we monitored the ‘val_loss’ metric with the value of mode as ‘min’. This was because, in our epoch vs loss plot, we observed that the validation loss was still decreasing with more epochs compared to other metrics. We used tensorboard to visualize the epoch vs accuracy & loss plots generated from the logs saved using the TensorBoard callback.

While observing the epoch vs accuracy plot we saw that after around 80 epoch the was decreasing little while the training accuracy was increasing. But since the difference of accuracy values were training and the validation was not very high, we concluded it was not a case of overfitting. The highest difference between them was at 100 epoch. Training accuracy was 63 and validation accuracy was 61. The difference was only 2. The epoch vs loss plot also had the same trend after 80 epochs.

Below figures have the epoch vs accuracy plot and epoch vs loss plot.



