# Deformable Template Estimation Implementation
***By:*** *Peeranat (ToTo) Tokaeo*

This is an implementation of the deformable template model as described in Allassonnière et al. 2007.
The paper is freely available [here](http://www.cis.jhu.edu/~allasson/em2.6.3.pdf).

This program is an artifact produced as part of 
my final year capstone project at Yale-NUS in 2021.

# Quick start

- Install dependencies using `python -m pip install -r requirements.txt --user`. 
  This assumes that your computer has a GPU that supports at least CUDA 10.1.
  
- Run `complete_workflow/main.py` with Python.
  

# How to use `complete_workflow/main.py` file

There are some options to configure at the top of `main.py`. 
Most of them should be self-explanatory. 

Note that if the `COINS` variable is set to `True`, the inputs used will be in
`complete_workflow/input_coins`. Else, `complete_workflow/input_data` is used.
Currently, the data in `input_data` are images of digits from the MNIST dataset.

# Data
Feel free to look through the ancient coins dataset in 
`complete_workflow/input_coins`. Only the `*.png` are used in training and classification.

# Adding data
Currently, the data is organized in 
the following structure.

<pre>
├───complete_workflow
│   ├───input_coins
│   │   ├───template1
│   │   │   ├───test
│   │   │   └───train
│   │   ├───template4
│   │   │   ├───test
│   │   │   └───train
│   │   ├───...
│   ├───input_data
│   │   ├───template0
│   │   │   ├───test
│   │   │   └───train
│   │   ├───template1
│   │   │   ├───test
│   │   │   └───train
│   │   ├───....
│   ├───...
├───....
</pre>
The images to be used are in `*.png` format and are divided
into the `test` and `train` for cross-validation.
Images that are pre-classified into the same class should
be in the same `template*` folder.

To use your own data, simply overwrite the images
in the  `input_data` folder and run the program
with `COINS` variable set to `FALSE`. Note that they should
still be a similar format to the default data, i.e.
are  in `*.png`, grayscale, etc.


## Coin data info
Alexander tetradrachms from Damascus (Glenn 2018),

Glenn, S. (2018). Exploring localities. A die study of Alexanders from Damascus. In: Alexander the
Great. A Linked Open World (Bordeaux), 91-126.

Thanks to Prof. Ernst Emanuel Mayer for providing the images.

## Sample output data info
A sample output is available at `complete_workflow/sample_train_output`.
Note that this is deliberately incomplete. I removed the saved covariance matrices
since they are too large for a Git repo. 

Outputs from running the actual script will have these covariance matrices.