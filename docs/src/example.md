# Example: Using LBFGS for Optimization

This example demonstrates how to use `LBFGS()` for optimization in Julia. A random initial solution is generated and a simple quadratic function is then optimized. 

## Setup

To run this example, ensure you have the required packages installed:

```r
using Pkg
Pkg.add(["Optim", "CUDA", "lbfgsgpu", "Random"])
```

Example code below on minimizing a SSD of 3 quadratic functions (M=3).
```r
using Optim
using CUDA
using lbfgsgpu
using Random

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

# Example function: Quadratic
function f_q(x::AbstractVector, given_height::Number)
    return sum((x .^ 2 .- given_height) .^ 2)
end

# Initialize parameters
M, min_r, max_r = 3, -10, 10
Random.seed!(69420)
x0 = random_init(M, min_r, max_r)

given_height = 0.5  # given_height value for function
f = x -> f_q(x, given_height)

# Run optimization
compute_and_print(f, x0)
compute_and_print(f, CuArray(x0))
```