# Image Classification Application - Command Line

### Structure

The whole interface contains 5 files and 1 folder,


#### **train.py** - training interface

Basic usage: python train.py data_directory

Prints out training loss, validation loss, and validation accuracy as the network trains
Options:

* Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
* Choose architecture: python train.py data_dir --arch "vgg13"
* Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
* Use GPU for training: python train.py data_dir --gpu


#### **predict.py** - predicting interface


Basic usage: python predict.py /path/to/image checkpoint Options:

* Return top K most likely classes: python predict.py input checkpoint --top_k 3
* Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
* Use GPU for inference: python predict.py input checkpoint --gpu


#### **model_spec.py** - utitity functions

Provides all helper functions that **train.py** and **predict.py** uses. The main train and predict functions are implemented here

#### **cat_to_json.json** - categories name mapping

Provides the encoding of 102 flower classes

#### **workspace_utils.py** - active session

Keeps the training session from being timed out

#### **checkpoints** - saves different model checkpoints

Directory where the trained model will be saved to. The pretrained weights could not be uploaded due to file size limits
