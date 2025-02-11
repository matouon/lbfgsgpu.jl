using Random
using Optim
include("../src/lbfgsgpu.jl")

#M is single solution size, N is number of solutions wanted
function random_init(M::Int, min_r::T, max_r::F) where {T<:Number,F<:Number}
    return rand(M).*(max_r.-min_r) .+ min_r
end

# CPU Variant
function compute_and_print(f::Function, x0::Array,verbose::Bool)
    res_cpu = optimize(f, x0, LBFGS())
    minimizer = Optim.minimizer(res_cpu)
    minimum = Optim.minimum(res_cpu)
    if verbose
        # println("Optimal x without CUDA: ", minimizer)
        println("Minimum f(x) without CUDA: ", minimum) 
    end
    return minimizer, minimum
end

# GPU Variant


function compute_and_print(f::Function, x0::CuArray, verbose::Bool)
    d = OnceDifferentiable(f, x0)
    method = LBFGS()
    res_gpu = optimize(d, x0, method)
    minimizer = Optim.minimizer(res_gpu)
    minimum = Optim.minimum(res_gpu)
    if verbose
        # println("Optimal x with CUDA: ", Optim.minimizer(res_gpu))
        println("Minimum f(x) with CUDA: ", minimum) 
    end
    return minimizer, minimum

end


function gaussian(x::AbstractVector, mu::Number, std::Number)
    (1/(std*sqrt(2*pi))).*exp.(-((x.-mu).^2)./(2*std^2))
end

compute_and_print(f, x0; verbose=false) = compute_and_print(f, x0, verbose)

#Define a function to optimize
function f(x::CUDA.CuArray{T}, given_height::F, gaus_mu::G, gaus_std::H) where {T<:Number,F<:Number,G<:Number,H<:Number}
    # Compute the sum of the absolute differences for all elements
    return sum((gaussian(x .^ 2, gaus_mu, gaus_std) .- given_height).^2)
end

function f(x::Array{T}, given_height::F, gaus_mu::G, gaus_std::H) where {T<:Number,F<:Number,G<:Number,H<:Number}
    # Compute the sum of the absolute differences for all elements
    return sum((gaussian(x .^ 2, gaus_mu, gaus_std) .- given_height).^2)
end



#quadratic functions -> not working? TODO 
function f_q(x::CUDA.CuArray{T}, given_height::F) where {T<:Number,F<:Number}
    # Compute the sum of the absolute differences for all elements
    return sum((x .^ 2 .- given_height).^2)
end

function f_q(x::Array{T}, given_height::F) where {T<:Number,F<:Number}
    # Compute the sum of the absolute differences for all elements
    return sum((x .^ 2 .- given_height).^2)
end

struct fun_sel
    gauss::Bool
end

function Base.show(io::IO, f_sel::fun_sel)
    if f_sel.gauss
        print(io, "Aeˣ")
    else
        print(io, "x²")
    end
end


########################################################################################################
#Run the base bfgs on this function
gauss = false #If gauss = true, obj. function is gauss. If gauss= false : objective func is quadratic.   (sum of squared diff*)
#This is for graphical purposes only
f_sel = fun_sel(gauss)
Random.seed!(69420)

M = 3 #solution size 
min_r, max_r = -10, 10 # min_range, max_range for the random initialization
given_height = 1/2

gaus_mu = 2
gaus_std = 1

#Sum of Gaussians
g = x -> f(x, given_height, gaus_mu, gaus_std)

#Sum of Quadratics
q = x -> f_q(x, given_height)


x0 = random_init(M, min_r, max_r) # Initial guess
compute_and_print(g, x0, verbose=true) 

res_gpu = optimize(g, CuArray(x0), LBFGS())


# x0 = random_init(M, min_r, max_r) # Initial guess
# compute_and_print(g, CuArray(x0), verbose=true)
