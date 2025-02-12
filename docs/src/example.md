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
M, min_r, max_r = 50, -10, 10
gaus_mu = 1
gaus_std = 1

x0 = random_init(M, min_r, max_r)

given_height = 0.25  # given_height value for function
f = x -> f_gaus_sq(x, given_height, gaus_mu, gaus_std)

# Run optimization
compute_and_print(f, x0)
compute_and_print(f, CuArray(x0))
```