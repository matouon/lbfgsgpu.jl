using Random
using Optim
using CUDA
using lbfgsgpu

#Randomly initializes a init solution, where M is solution size, min_r, max_r are minimal and maximal range of random numbers
function random_init(M::Int, min_r::T, max_r::F) where {T<:Number,F<:Number}
    return rand(M).*(max_r.-min_r) .+ min_r
end

#Computes the solution to a given function f with initial solution x0
function compute_and_print(f::Function, x0::AbstractVector, verbose::Bool)
    res_gpu = optimize(f, x0, LBFGS())
    minimizer = Optim.minimizer(res_gpu)
    minimum = Optim.minimum(res_gpu)
    if verbose
        if x0 isa CuArray
            # println("Optimal x with CUDA: ", Optim.minimizer(res_gpu))
            println("Minimum f(x) with CUDA: ", minimum) 
        else    
            # println("Optimal x with CUDA: ", Optim.minimizer(res_gpu))
            println("Minimum f(x) without CUDA: ", minimum) 
        end
    end
    return minimizer, minimum
end

function gaussian(x::AbstractVector, mu::Number, std::Number)
    (1/(std*sqrt(2*pi))).*exp.(-((x.-mu).^2)./(2*std^2))
end

#Simple constructor for verbose input
compute_and_print(f, x0; verbose=false) = compute_and_print(f, x0, verbose)


#Gaussian function
function f_gaus(x::AbstractVector{T}, given_height::F, gaus_mu::G, gaus_std::H) where {T<:Number,F<:Number,G<:Number,H<:Number}
    # Compute the sum of the squared differences for all elements
    return sum((gaussian(x, gaus_mu, gaus_std) .- given_height).^2)
end

#Gaussian function with squared input
function f_gaus_sq(x::AbstractVector{T}, given_height::F, gaus_mu::G, gaus_std::H) where {T<:Number,F<:Number,G<:Number,H<:Number}
    # Compute the sum of the squared differences for all elements
    return sum((gaussian(x .^ 2, gaus_mu, gaus_std) .- given_height).^2)
end

#Quadratic function
function f_q(x::AbstractVector{T}, given_height::F) where {T<:Number,F<:Number}
    # Compute the sum of the squared differences for all elements
    return sum((x .^ 2 .- given_height).^2)
end


#Setting default function to identity, in case user specifies incorrect function name 
def_fun = x -> x

#A helper structure for function selection
struct fun_sel
    gauss::Bool
    gauss_sq::Bool
    quad::Bool
end

#Helper Wrapper function for selection of the optimization function
function set_selection_struct(f_str::String)
    if f_str == "g"
        f_sel = fun_sel(true, false, false)
    elseif f_str == "gs"
        f_sel = fun_sel(false, true, false)
    elseif f_str == "q"
        f_sel = fun_sel(false, false, true)
    else
        #Unknown function -> sets every decision flag to false
        f_sel = fun_sel(false, false, false)
    end
    f_sel
end

#Helper function to select a function using string input from user
function sel_opt_fun(f_str::String)
    f_sel = set_selection_struct(f_str)
    if f_sel.gauss
        f = g
    elseif f_sel.gauss_sq
        f = gs
    elseif f_sel.quad
        f = q
    else
        f = Nothing
    end
    @assert f !== Nothing "Incorrect function selected, enter a valid function name"
    f
end

#Usable functions definition
function define_functions(given_height, gaus_mu, gaus_std)
    #Sum of Gaussians
    g = x -> f_gaus(x, given_height, gaus_mu, gaus_std)
    #Sum of gaussians with squared input
    gs = x -> f_gaus_sq(x, given_height, gaus_mu, gaus_std)
    #Sum of Quadratics
    q = x -> f_q(x, given_height)
    g, gs, q
end


########################################################################################################
#Run the base bfgs on this function
M = 3 #solution size 
min_r, max_r = -10, 10 # min_range, max_range for the random initialization
given_height = 1/2 #The height at which we want the value of selected function to be
gaus_mu = 2
gaus_std = 1
f_str = "q" #either g, gs, q -> gaussian, gaussian with sq. input, quadratic

f = sel_opt_fun(f_str)

g, gs, q = define_functions(given_height, gaus_mu, gaus_std)

Random.seed!(69420)
x0 = random_init(M, min_r, max_r) # Initial guess
compute_and_print(f, x0, verbose=true) 

Random.seed!(69420)
x0 = random_init(M, min_r, max_r) # Initial guess
compute_and_print(f, CuArray(x0), verbose=true)
