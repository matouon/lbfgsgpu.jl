# Here is a very brief description of all usable functions in package


#### functions : Standard definition for functions
##### Gaussian, Gaussian with squared input, Quadratic function
Here x is the input, mu is the mean value for gaussian and std is standard deviation for gaussian.
These functions return function values for respective x.
```
function gaussian(x::AbstractVector, mu::Number, std::Number)
function gaussian(x::Number, mu::Number, std::Number)
function gaussian_sq(x::AbstractVector, mu::Number, std::Number)
function gaussian_sq(x::Number, mu::Number, std::Number)
function quad(x::AbstractVector)
function quad(x::Number)
```

#### These are the SSD metrics for respective functions
Here x is the solution, given_height is the target function value for the optimization and gaus_mu, gaus_std are mu and std for gaussians. These functions return the objective value for some input vector x.
```
function f_gaus(x::AbstractVector{T}, given_height::F, gaus_mu::G, gaus_std::H) where {T<:Number,F<:Number,G<:Number,H<:Number}
function f_gaus_sq(x::AbstractVector{T}, given_height::F, gaus_mu::G, gaus_std::H) where {T<:Number,F<:Number,G<:Number,H<:Number}
function f_q(x::AbstractVector{T}, given_height::F) where {T<:Number,F<:Number}
```

#### A helper structure for function selection
```
struct fun_sel
    gauss::Bool
    gauss_sq::Bool
    quad::Bool
end
```

#### Helper Wrapper function for selection of the optimization function
Here f_str can be either `q`, `g` or `gs` for respective function selection of : Quadratic, Gaussian or Gaussian with squared input.
This function returns `struct fun_sel` with user specified settings.
```
function set_selection_struct(f_str::String)
```

#### Helper function to select a function using string input from user
Here f_str can be either `q`, `g` or `gs` for respective function selection of : Quadratic, Gaussian or Gaussian with squared input.
The g, gs and q inputs are the anonymous functions defined in 
 `define_functions(given_height, gaus_mu, gaus_std)`.
This function returns f, h, where f is the anonymous SSD function and h is the original actual function of interest.
```
function sel_opt_fun(f_str::String, g, gs, q)
```

#### Wrapper function to define anonymous helper functions 
Here given_height is the target function value for the optimization and gaus_mu, gaus_std are mu and std for gaussians.
This function returns g, gs, q, which are the anonymous SSD objective functions.
```
function define_functions(given_height, gaus_mu, gaus_std)
```

#### Simple constructor for diff input number for gaussians/quadratic
This function in this context is used with x_vals being some range to allow drawing of the function selected by f_str. Sols are the minimizers of our optimization, gaus_mu, gaus_std are mean and std deviation, respectively and f_str is the selection string.
This function returns sol_vals, y_vals, where sol_vals are the original function values for sols and y_vals are function values for x_vals. 

```
function eval_sols(x_vals::T, sols::F, gaus_mu::G, gaus_std::H, f_str::String) where {T <: AbstractArray, F <: AbstractArray, G <: Number, H <: Number}
```

#### Multiple dispatch for gaussian
Same return type as original `eval_sols`
```
function eval_sols_(x_vals::T, sols::F, gaus_mu::G, gaus_std::H) 
where {T <: AbstractArray, F <: AbstractArray, G <: Number, H <: Number}   
```

#### Multiple dispatch for quadratic
Same return type as original `eval_sols`

```
function eval_sols_(x_vals::T, sols::F) where {T <: AbstractArray, F <: AbstractArray}
```

##### Simple constructor for diff input number for gaussians/quadratic
All these are just the previous eval_sols functions, 
but modified very slightly to allow testing.
During tests sols is used as the minimizer, gaus_mu,gaus_std are mu and std for gaussian respectively, f_str is the selection string and orig_fun is the function we want to use in optimization (NOT THE SSD objective function).

This function returns sol_vals, which are the original function values for sols.
```
function eval_sols_test(sols::F, gaus_mu::G, gaus_std::H, f_str::String, orig_fun) where {F <: AbstractArray, G <: Number, H <: Number}
```

##### Multiple dispatch for quadratic
Same return type as for original `eval_sols_test` function
```
function eval_sols_test_(sols::F, gaus_mu::G, gaus_std::H, orig_fun) where {F <: AbstractArray, G <: Number, H <: Number}
```

##### Multiple dispatch for quadratic
Same return type as for original `eval_sols_test` function

```
function eval_sols_test_(sols::F, orig_fun) where {F <: AbstractArray}
```

#### Random initial solution generator
Given number of initial solutions M, 
minimal and maximal ranges for the random generatin min_r, max_r,
this generates a vector of random floats given desired properties.
Returns generated Vector{Float64} of size 1xM 
```
function random_init(M::Int, min_r::T, max_r::F) where {T<:Number,F<:Number}
```


#### Computes the solution to a given function f with initial solution x0
This is a helper function to be used during the optimization, where f is the objective function (in our case SSD), x0 the initial solution.
Returns the  minimizer, and the minimum for given SSD objective function.

```
function compute_and_print(f::Function, x0::AbstractVector, verbose::Bool)
```

#### Simple constructor for verbose input
Same returns as the original `compute_and_print` function as this is a constructor for it. 
```
compute_and_print(f, x0; verbose=false) = compute_and_print(f, x0, verbose)
```

#### filtering function to filter correct solutions to the HEP problem (upto some tolerance)
This function filters solution function values sol_vals based on some given_height up to some tolerance.
This function returns `filtered_sols`, `filtered_vals`, which are
the filtered solutions and the values of these solutions respectively.

```
function filtering(sol_vals::AbstractArray, given_height tolerance::Number)
```

#### Base overload for nice outputs (works for quadratic and gaussian and is not currently used)
```
function Base.show(io::IO, f_sel::fun_sel)
```

#### The main function, which is correctly described 
#### up to the x::CuArray in 
#### [Optim.jl](https://github.com/JuliaDiff/FiniteDiff.jl/blob/a7eca2d4b73c4de12140d89df7621fcc90d29190/src/gradients.jl#L165)
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