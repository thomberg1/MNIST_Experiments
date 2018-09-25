MNIST (Modified National Institute of Standards and Technology) is a dataset of handwritten digits that has been and is extensively used to experiment with neural networks. The goal is to train a classifier to predict a digit (0-9) represented by a 28x28 image. The dataset was designed by Yann LeCun, Corinna Cortes, Christopher J.C. Burges <a href=http://yann.lecun.com/exdb/mnist/> (The MNIST Database of handwritten digits)</a>

<b>data_analysis_MNIST</b> - notebook that contains visualizations and statistics about the dataset.

![Alt text](images/dataanalysis.jpg?raw=true "")
<br/>

<b>experiments_*</b> - notebooks that train several different models with and without augmentation. To get to consistent results that allow a comparison of models hyperparameter selection, model training, and model evaluation are fully automated. 

The scripts use a brute force parameter grid evaluation using sklearn <a href=http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html>ParameterGrid</a>. The function takes a parameter grid and produces a list of dictionaries of parameters.

For each parameter dictionary the script spawns a sub process that trains the selected model architecture with the hyperparameters given both by the dictionary. Once all parameter combinations have been evaluated the script selects the trained model with the highest validation score and performs an evaluation.

![Alt text](images/parametereval.jpg?raw=true "")
<br/>
