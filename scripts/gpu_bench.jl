using Random
using Optim
using BenchmarkTools
using CUDA
using DataFrames
using CSV
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


#Base overload for nice outputs
function Base.show(io::IO, f_sel::fun_sel)
    if f_sel.gauss
        print(io, "Aeˣ")
    else
        print(io, "x²")
    end
end


########################################################################################################
f_str = "g" #either g, gs, q -> gaussian, gaussian with sq. input, quadratic

min_r, max_r = -10, 10 # min_range, max_range for the random initialization
given_height = 1/2
gaus_mu = 2
gaus_std = 1


g, gs, q = define_functions(given_height, gaus_mu, gaus_std)

fun = sel_opt_fun(f_str, g, gs, q)

# Initialize DataFrame to store results
results = DataFrame(
    Num_Variables = Int[],
    CUDA = Bool[],
    Mean_t = Float64[],
    Min_t = Float64[],
    Min_Value = Float64[]
)


ns_to_s = 1e-9 #Conversio nmade because benchmark returns time in ns
num_of_points = Int64[1e2, 2e2, 2e3, 3e3, 5e3, 7e3, 1e4]

for m in num_of_points
    print("Testing for: ", m, " variables --------> ")
    print(" Without CUDA -------> ")
    Random.seed!(69420)
    x0 = random_init(m, min_r, max_r) # Initial guess
    #TODO replace with benchmarking to get min and mean time and not only one time!!!
    stats = @benchmark compute_and_print(fun, $x0) 

    min_time = minimum(stats.times)  # Minimum cycle time
    mean_time = mean(stats.times) # Mean cycle time
    min_sol, min_value = compute_and_print(fun, x0)  # Compute min value (assuming it returns one)
    push!(results, (m, false, mean_time*ns_to_s, min_time*ns_to_s, min_value))

    print(" With CUDA\n")
    Random.seed!(69420)
    x0_cuda = CuArray(random_init(m, min_r, max_r)) # Initial guess
    stats_cuda = @benchmark compute_and_print(fun, $x0_cuda)
    min_time_cuda = minimum(stats_cuda.times) # Minimum cycle time
    mean_time_cuda = mean(stats_cuda.times) # Mean cycle time
    min_sol_cuda, min_value_cuda = compute_and_print(fun, x0_cuda)
    push!(results, (m, true, mean_time_cuda*ns_to_s, min_time_cuda*ns_to_s, min_value_cuda))
end
# Display results
println(results)
CSV.write("benchmark_results.csv", results)






