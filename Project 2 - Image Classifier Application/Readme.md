# Deep Learning 

### Project: Image Classifier Project 


### Data

The data for this project is quite large - in fact, it is so large you cannot upload it onto Github. You will be training using 102 different types of flowers, where there ~20 images per flower to train on.  Then you will use your trained classifier to see if you can predict the type for new images of the flowers (Quoted from Udacity).

### Jupyter Notebook

This notebook implements the inception model in the jupyter notebook format. Most of the functions are static. To view the notebook, go to the following link

[Project Notebook: Image Classifier](http://nbviewer.jupyter.org/github/chenbowen184/Udacity_Data_Science_Projects/blob/master/Project%202%20-%20Image%20Classifier%20Application/Image%20Classifier%20Project.ipynb?flush_cache=true)

### Application

The notebook is then converted into a command line application

Specifications

The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. 

Train a new network on a data set with train.py

Basic usage: python train.py data_directory
* Prints out training loss, validation loss, and validation accuracy as the network trains

Options:
* Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
* Choose architecture: python train.py data_dir --arch "vgg13"
* Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
* Use GPU for training: python train.py data_dir --gpu

Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image * /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
* Return top K most likely classes: python predict.py input checkpoint --top_k 3
* Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
* Use GPU for inference: python predict.py input checkpoint --gpu
