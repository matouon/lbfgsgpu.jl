################
The goal of this package is to solve a physics inspired problem described in documentation using a GPU implementation of L-BFGS solver from Optim.jl .

The base L-BFGS is directly used from https://github.com/JuliaNLSolvers/Optim.jl/blob/master/src/multivariate/solvers/first_order/l_bfgs.jl


# Problem statement

The base problem for the kernelization of L-BFGS is physics inspired requierement for efficient and fast equation solver. Without any difficult description, the problem can be reformulated as finding set of input variables (x_1, x_2, ..., x_n), for which a function is at a given height (user specified). We hence try to solve f(\vec{x}) = given_height, which can be translated into  f(\vec{x}) - given_height = 0. 

# Solution
The solution to this problem can be formulated as sum of squared differences between each element from vector x and a scalar value in this manner -> f_opt = SSD(f(\vec{x}), given_height), where f_opt is objective function, which we want to minimize. Because we can formulate the problem as minimization of some SSD, we can directly use L-BFGS as this is the ideal function to solve such problem. 

The implemented function that solved the CUDA incompability is in /src/Optim_dispatch.jl

Apart from the obvious change of type Array - > CuArray,
the only issue was scalar indexing, which is not very good for GPU as it basically cancels the parallelism. The fix was correct and logical use of broadcasting. 

# Benchmarking
To obtain some basic benchmarking, run /scripts/gpu_bench.jl. After running the benchmarking there should be a file called benchmark_results.csv with computed benchmark data. 


# Tests

You can test this package with provided testsets.

# Implementation of the solver with CUDA
The functiom implementing the CPU -> GPU transition is
```
function FiniteDiff.finite_difference_gradient!(
    df::StridedVector{<:Number},
    f,
    x::CuArray,
    cache::FiniteDiff.GradientCache{T1,T2,T3,T4,fdtype,returntype,inplace};
    relstep=FiniteDiff.default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    dir=true) where {T1,T2,T3,T4,fdtype,returntype,inplace}
``` 
from the pkg FiniteDiff. It is used in multiple dispatch as we dispatch through the vector of initial solutions x. In the original script there is x::Array, but to enable GPU use we use x::CuArray

# Brief description of /scripts

### lbfgs_test.jl -> Main example, which can be used to test the individual functions
### draw_sols.jl -> A function, which can be used to draw the solution plots 
### gpu_bench.jl -> Benchmarking script
### plot_bench.jl -> Helper script to plot the benchmarking results into a line plot


# Example: Using LBFGS for Optimization

This example demonstrates how to use `LBFGS()` for optimization in Julia. A random initial solution is generated and a simple quadratic function is then optimized. 

## Setup

To run this example, ensure you have the required packages installed:

```r
using Pkg
Pkg.add(["Optim", "CUDA", "Random"])
Pkg.add(url="https://github.com/matouon/lbfgsgpu.jl")

```

Example code below on minimizing a SSD of 3 quadratic functions (M=3).
```r
using Optim
using CUDA
using lbfgsgpu
using Random

function gaussian(x::AbstractVector, mu::Number, std::Number)
    (1 / (std * sqrt(2 * pi))) * exp.(-((x .- mu) .^ 2) / (2 * std^2))
end

function gaussian(x::Number, mu::Number, std::Number)
    (1 / (std * sqrt(2 * pi))) * exp.(-((x - mu) ^ 2) / (2 * std^2))
end
# Randomly initialize solution
function random_init(M::Int, min_r::T, max_r::T) where {T<:Number}
    return rand(M) .* (max_r - min_r) .+ min_r
end

# Compute solution using LBFGS
function compute_and_print(f::Function, x0::AbstractVector)
    res = optimize(f, x0, LBFGS())
    println("Minimum f(x): ", Optim.minimum(res))
    return Optim.minimizer(res)
end

#Gaussian gaussian with squared input
function f_gaus_sq(x::AbstractVector{T}, given_height::F, gaus_mu::G, gaus_std::H) where {T<:Number,F<:Number,G<:Number,H<:Number}
    # Compute the sum of the squared differences for all elements
    return sum((gaussian(x .^ 2, gaus_mu, gaus_std) .- given_height).^2)
end

# Initialize parameters
M, min_r, max_r = 300, -10, 10
gaus_mu = 1
gaus_std = 1

x0 = random_init(M, min_r, max_r)

given_height = 0.25  # given_height value for function
f = x -> f_gaus_sq(x, given_height, gaus_mu, gaus_std)

# Run optimization
compute_and_print(f, x0)
compute_and_print(f, CuArray(x0))
```