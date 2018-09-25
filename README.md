MNIST (Modified National Institute of Standards and Technology) is a dataset of handwritten digits that has been and is extensively used to experiment with neural networks. The goal is to train a classifier to predict a digit (0-9) represented by a 28x28 image. The dataset was designed by Yann LeCun, Corinna Cortes, Christopher J.C. Burges <a href=http://yann.lecun.com/exdb/mnist/> (The MNIST Database of handwritten digits)</a>

<b>data_analysis_MNIST</b> - notebook that contains visualizations and statistics about the dataset.

![Alt text](images/dataanalysis.jpg?raw=true "")
<br/>

<b>experiments_*</b> - notebooks that train several different models with and without augmentation. To get to consistent results that allow a comparison of models hyperparameter selection, model training, and model evaluation are fully automated. 

The scripts use a brute force parameter grid evaluation using sklearn <a href=http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html>ParameterGrid</a>. The function takes a parameter grid and produces a list of dictionaries of parameters.

For each parameter dictionary the script spawns a sub process that trains the selected model architecture with the hyperparameters given both by the dictionary. Once all parameter combinations have been evaluated the script selects the trained model with the highest validation score and performs an evaluation.

![Alt text](images/parametereval.jpg?raw=true "")
<br/>

<b>LeNet 5 3</b> – uses Yann LeCun LeNet 5 architecture but with a 3x3 receptive field. Note, that the parameters of the trained model are used to initialize the LeNet 5 3 model used in all the experiments below. 
<br/><br/>
<b>LeNet 5 3 augmented</b> – uses the LeNet 5 3 model combined with data augmentation such as elastic distortion, rotation, zoom etc.
<br/><br/>
<b>MNIST STN</b> - uses the LeNet 5 3 with a Spatial Transformer Network with bilinear interpolation. The implementation is based on work by Fei Xia <a href=” https://github.com/fxia22/stn.pytorch”> PyTorch version of spatial transformer network</a>. Pytorch has a build-in implementation - affine_grid and grid_sample -  based on the NVIDIA cuDNN library. However the implementation is one of the cuDNN functions that is non-deterministic (and can’t be changed by the “deterministic” flag). This causes results to skew over different runs and makes the evaluation of experiments hard.
<br/><br/>
<b>MNIST TPN</b> - uses the LeNet 5 3 and a STN with Thin Plate Spline (TPS). The implementation is based on WarBean’s repository <a href=https://github.com/WarBean/tps_stn_pytorch> PyTorch implementation of Spatial Transformer Network (STN) with Thin Plate Spline (TPS)</a>

The results of above experimentation scripts are summarized in the table below:
<br/>

![Alt text](images/results.jpg?raw=true "")

<br/>
<b>visualization_*</b> - notebooks that provide several interesting ways to explore the MNIST datasets using the trained neural networks.
<br/>
<b>Data Augmentation</b> - notebook to explore data transformations interactively. The script is used to define the parameters for the different augmentation functions such as rotate, shear, elastic transformation etc.
<br/>
![Alt text](images/transforms.jpg?raw=true "")
<br/>
<b>MNIST CAM</b> - notebook to interactively explore activation maps in the final layer before the fully connected layer, i.e. to visualize what the neural network (its convolution layers) are focusing on.
<br/>
![Alt text](images/cam.jpg?raw=true "")
<br/>
<b> MNIST Recognizer GUI</b> - provides a GUI to draw a digit and to classify the digit using the LeNet 5 3 model.
<br/>
![Alt text](images/recognizer.jpg?raw=true "")


![Alt text](images/recognizer.jpg?raw=true "")

