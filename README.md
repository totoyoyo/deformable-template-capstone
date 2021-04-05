# Deformable Template Estimation Implementation

This is an implementation of the deformable template estimation based on Allassonnière et al. 2007
The paper is freely available [here](http://galton.uchicago.edu/~amit/Papers/em.pdf)

# Quick Start

- Install dependencies using `python -m pip install -r requirements.txt --user` 
  This assumes that your computer has a GPU that supports at least CUDA 10.1.
  
- Run complete_workflow/main.py
  

# How to Use `complete_workflow/main.py` file

There are some options to configure at the top of `main.py`. 
Most of them should be self-explanatory. 
Note that if the `COINS` variable is set to `True`, the inputs used will be in
`complete_workflow/input_coins`. Else, `complete_workflow/input_data` is used.
Currently, there are some images of digits in there from the MNIST dataset.
