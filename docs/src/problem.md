# Problem

My core motivation (can show some alternative GPU use) for kernelizing L-BFGS is a real life HEP analysis problem at CERN. The problem is rather difficult to explain and to my current best knowledge it is not formalized as an official assignment in some CERN documentation yet. However you will have to trust me on this, but it can be reformulated as finding a set of input variables 

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
