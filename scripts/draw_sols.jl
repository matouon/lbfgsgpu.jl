using Random
using CUDA
using lbfgsgpu.Optim
using lbfgsgpu
using Plots

function gaussian(x::AbstractVector, mu::Number, std::Number)
    (1 / (std * sqrt(2 * pi))) * exp.(-((x .- mu) .^ 2) / (2 * std^2))
end

function gaussian(x::Number, mu::Number, std::Number)
    (1 / (std * sqrt(2 * pi))) * exp.(-((x - mu) ^ 2) / (2 * std^2))
end

function gaussian_sq(x::AbstractVector, mu::Number, std::Number)
    (1 / (std * sqrt(2 * pi))) * exp.(-((x .^ 2 .- mu) .^ 2) / (2 * std^2))
end

function gaussian_sq(x::Number, mu::Number, std::Number)
    (1 / (std * sqrt(2 * pi))) * exp.(-((x ^ 2 - mu) ^ 2) / (2 * std^2))
end


function quad(x::AbstractVector)
    x.^2
end

function quad(x::Number)
    x.^2
end



#Gaussian function
function f_gaus(x::AbstractVector{T}, given_height::F, gaus_mu::G, gaus_std::H) where {T<:Number,F<:Number,G<:Number,H<:Number}
    # Compute the sum of the squared differences for all elements
    return sum((gaussian(x, gaus_mu, gaus_std) .- given_height).^2)
end

#Gaussian function with squared input
function f_gaus_sq(x::AbstractVector{T}, given_height::F, gaus_mu::G, gaus_std::H) where {T<:Number,F<:Number,G<:Number,H<:Number}
    # Compute the sum of the squared differences for all elements
    return sum((gaussian(x, gaus_mu, gaus_std) .- given_height).^2)
end

#Quadratic function
function f_q(x::AbstractVector{T}, given_height::F) where {T<:Number,F<:Number}
    # Compute the sum of the squared differences for all elements
    return sum((x .^ 2 .- given_height).^2)
end

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
function sel_opt_fun(f_str::String, g, gs, q)
    f_sel = set_selection_struct(f_str)
    if f_sel.gauss
        f = g
        h = gaussian
    elseif f_sel.gauss_sq
        f = gs
        h = gaussian_sq
    elseif f_sel.quad
        f = q
        h = quad
    else
        f = Nothing
        h = Nothing
    end
    @assert f !== Nothing "Incorrect function selected, enter a valid function name"
    f, h
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

#Simple constructor for diff input number for gaussian/quadratic
function eval_sols(x_vals::T, sols::F, gaus_mu::G, gaus_std::H, f_str::String) where {T <: AbstractArray, F <: AbstractArray, G <: Number, H <: Number}
    if f_str == "g" || f_str == "gs"
        sol_vals, y_vals = eval_sols_(x_vals, sols, gaus_mu, gaus_std)
    else
        sol_vals, y_vals = eval_sols_(x_vals, sols)
    end
    sol_vals, y_vals
end

#Multiple dispatch for gaussian
function eval_sols_(x_vals::T, sols::F, gaus_mu::G, gaus_std::H) where {T <: AbstractArray, F <: AbstractArray, G <: Number, H <: Number}
    sol_vals = orig_fun.([sols], gaus_mu, gaus_std)
    y_vals = orig_fun.(collect(x for x in x_vals), gaus_mu, gaus_std)
    sol_vals, y_vals
end

#Multiple dispatch for quadratic
function eval_sols_(x_vals::T, sols::F) where {T <: AbstractArray, F <: AbstractArray}
    sol_vals = orig_fun.([sols])
    y_vals = orig_fun.(collect(x for x in x_vals))
    sol_vals, y_vals
end


#M is single solution size
function random_init(M::Int, min_r::T, max_r::F) where {T<:Number,F<:Number}
    return rand(M).*(max_r.-min_r) .+ min_r
end


#Simple constructor for verbose input
compute_and_print(f, x0; verbose=false) = compute_and_print(f, x0, verbose)

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

function filtering(sol_vals::AbstractArray, given_height::Number, tolerance::Number)
    sol_vals_cpu_flat = vcat(sol_vals...)  # Flatten nested arrays

    filtered_indices = findall(abs.(sol_vals_cpu_flat .- given_height) .< tolerance)
    filtered_sols = sols[filtered_indices]
    filtered_vals = sol_vals_cpu_flat[filtered_indices]

    filtered_sols, filtered_vals
end
########################################################################################################
f_str = "q" #either g, gs, q -> gaussian, gaussian with sq. input, quadratic

M = 500 #solution size 
min_r, max_r = -10, 10 # min_range, max_range for the random initialization
given_height = 0.25 #The height at which we want the value of selected function to be
gaus_mu = 1
gaus_std = 1

g, gs, q = define_functions(given_height, gaus_mu, gaus_std)
obj_fun, orig_fun = sel_opt_fun(f_str, g, gs, q)


Random.seed!(69420)

#CPU
x0 = random_init(M, min_r, max_r) # Initial guess
sols, min = compute_and_print(obj_fun, x0, verbose=true)

#GPU
#Here min is not the actual function value, but the value of the OBJECTIVE function
x0_cuda = CuArray(x0)
sols, min = compute_and_print(obj_fun, x0_cuda, verbose=true)


num_of_x_vals = 200
#Get solution values and y values of respective solutions
x_vals = Array(range(min_r, max_r, length=num_of_x_vals))

sol_vals, y_vals = eval_sols(x_vals, sols, gaus_mu, gaus_std, f_str)

p = plot(x_vals, y_vals, label="g(x)", linewidth=2, legend=:topright)


#post processing of solution!!!! Very important step to check whether solutions converged correctly!!
tolerance = 0.01

filtered_sols, filtered_vals = filtering(sol_vals, given_height, tolerance)

# filtered_sols, filtered_vals
scatter!(Array(filtered_sols), Array(filtered_vals), label="Solutions", markersize=5, color=:red)

savefig(p, "q.png")

