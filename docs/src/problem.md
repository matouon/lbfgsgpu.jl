# Problem

My core motivation (can show some alternative GPU use) for kernelizing L-BFGS is a real life HEP analysis problem at CERN. As described below, there is a need for an efficient and fast equation solver. The problem is rather difficult and to my current best knowledge it is not open for public yet. However you will have to trust me on this, but it can be reformulated as finding a set of input variables 

```math 
x_1, x_2, ..., x_n 
``` 

for which a function reaches a given height (user-specified):

```math 
f(\vec{x}) = \text{given\_height} 
``` 

which translates into:

```math 
f(\vec{x}) - \text{given\_height} = 0 
``` 

We formulate the objective function as:

```math 
f_{\text{opt}} = \text{SSD}(f(\vec{x}), \text{given\_height}) 
``` 

where SSD is the sum of squared differences, making L-BFGS an ideal solver for such problem.
