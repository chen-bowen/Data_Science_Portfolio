# Capstone Project - Sparkify Music Service Analytics

### Motivation

As many music streaming services that opreates on dataset with seconds of timeframe, the size of the data grows expotentially large in a short while. The size of the data would quickly outgrown the memory limit of most personal computers. To ensure the analytical activities are still performed as normal, we need to utilize the power of spark to perform a distributed data analytics task to obtain insights for from the data. In particular, we would like to understand the factors that contributes to users churning behaviors.

### Data Descriptions

The data provided is the user log of the service, having demographic info, user activities, timestamps and etc. The data contains the user information logs that includes 

* Add Friend
* Add to Playlist
* Cancel/Cancel Confirmation
* Submit Upgrade/Upgrade
* Submit Downgrade/Downgrade
* Error
* Help
* Home
* Logout
* Nextsong
* Roll Advert
* Save Settings
* Thumbs Up / Down

### Project Objective

Using the user information logs, we will attempt to predict the possiblities of a user churning using machine learning models. We will also attempt to understand contributing factors of the churning behaviors.

### Required Packages

* Pandas
* pyspark
* matplot
* numpy

### Model Refinement

The presented model represents the best model I have constructed so far. Originally I only used the all the activities in *page* as features, which yielded 0.69 F1 score on the small test set we have. After I engineered 6 other features as noted in the project, I was able to obtain an F1 score of 0.80 (0.88 after scale up to the large dataset).

### Gallery


[Project Notebook: Spark - Subset Analytics](https://nbviewer.jupyter.org/github/chenbowen184/Data_Science_Portfolio/blob/master/Capstone%20Project/Spark%20-%20Subset%20Analytics.ipynb)

[Project Notebook: Spark - Full Dataset Analytics](https://nbviewer.jupyter.org/github/chenbowen184/Data_Science_Portfolio/blob/master/Capstone%20Project/Spark%20-%20Full%20Dataset.ipynb)

[Blog Post: Understanding Customer Churning with Big Data Analytics](https://medium.com/@bowenchen/understanding-customer-churning-with-big-data-analytics-70ce4eb17669)

