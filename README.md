# Deformable Template Estimation Implementation

This is an implementation of the deformable template estimation based on Allassonnière et al. 2007
The paper is freely available [here](http://galton.uchicago.edu/~amit/Papers/em.pdf).

# Quick start

- Install dependencies using `python -m pip install -r requirements.txt --user` 
  This assumes that your computer has a GPU that supports at least CUDA 10.1.
  
- Run complete_workflow/main.py
  

# How to use `complete_workflow/main.py` file

There are some options to configure at the top of `main.py`. 
Most of them should be self-explanatory. 
Note that if the `COINS` variable is set to `True`, the inputs used will be in
`complete_workflow/input_coins`. Else, `complete_workflow/input_data` is used.
Currently, there are some images of digits in there from the MNIST dataset.

# Data
Feel free to look through the ancient coins dataset in 
`complete_workflow/input_coins`.

## Input data info
Alexander tetradrachms from Damascus (Glenn 2018),

Glenn, S. (2018). Exploring localities. A die study of Alexanders from Damascus. In: Alexander the
Great. A Linked Open World (Bordeaux), 91-126.

Thanks to Prof. Ernst Emanuel Mayer for providing the images.

## Output data info
A sample output is available at `complete_workflow/sample_train_output`.
Note that this is deliberately incomplete. I removed the saved covariance matrices
since they are too large for a Git repo. 

Outputs from running the actual script will have these covariance matrices.