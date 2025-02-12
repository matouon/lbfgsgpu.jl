using Test
using CUDA
using lbfgsgpu.Optim
using lbfgsgpu
using Random
using Statistics

#M is single solution size
function random_init(M::Int, min_r::T, max_r::F) where {T<:Number,F<:Number}
    return rand(M).*(max_r.-min_r) .+ min_r
end


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
    return sum((gaussian(x .^ 2, gaus_mu, gaus_std) .- given_height).^2)
end

#Quadratic function
function f_q(x::AbstractVector{T}, given_height::F) where {T<:Number,F<:Number}
    # Compute the sum of the squared differences for all elements
    return sum((x .^ 2 .- given_height).^2)
end

function compute_test(f, x0::AbstractVector)
    res = optimize(f, x0, LBFGS())
    minimizer = Optim.minimizer(res)
    Array(minimizer)
end

function filtering(sol_vals::AbstractArray, tolerance::Number, given_height::Number)
    sol_vals_cpu_flat = vcat(sol_vals...)  # Flatten nested arrays

    filtered_indices = findall(abs.(sol_vals_cpu_flat .- given_height) .< tolerance)
    filtered_vals = sol_vals_cpu_flat[filtered_indices]

    filtered_vals
end


#Simple constructor for diff input number for gaussian/quadratic
function eval_sols(sols::F, gaus_mu::G, gaus_std::H, f_str::String, orig_fun) where {F <: AbstractArray, G <: Number, H <: Number}
    if f_str == "g" || f_str == "gs"
        sol_vals = eval_sols_( sols, gaus_mu, gaus_std, orig_fun)
    else
        sol_vals = eval_sols_(sols, orig_fun)
    end
    sol_vals
end

#Multiple dispatch for gaussian
function eval_sols_(sols::F, gaus_mu::G, gaus_std::H, orig_fun) where {F <: AbstractArray, G <: Number, H <: Number}
    sol_vals = orig_fun.([sols], gaus_mu, gaus_std)
    sol_vals
end

#Multiple dispatch for quadratic
function eval_sols_(sols::F, orig_fun) where {F <: AbstractArray}
    sol_vals = orig_fun.([sols])
    sol_vals
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

function init_test(var_n, giv_h, gaus_mu, gaus_std, f, f_str, tolerance)
        g, gs, q = define_functions(giv_h, gaus_mu, gaus_std)
        obj_fun, orig_fun = sel_opt_fun(f_str, g, gs, q)


        Random.seed!(69420)
        x0 = random_init(var_n, -10, 10) # Initial guess
        x0_cuda = CuArray(copy(x0))

        minimizer = compute_test(f, x0)
        minimizer_cuda = compute_test(f, x0_cuda)
        sol_vals = eval_sols(minimizer, gaus_mu, gaus_std, f_str, orig_fun)
        sol_vals_cuda = eval_sols(minimizer_cuda, gaus_mu, gaus_std, f_str, orig_fun)

        filtered_vals = filtering(sol_vals, tolerance, giv_h)
        filtered_vals_cuda = filtering(sol_vals_cuda, tolerance, giv_h)
        m_val = mean(filtered_vals)
        m_val_c = mean(filtered_vals_cuda)
        m_val, m_val_c
end


@testset "function optimization tests" begin
    @testset "Gauss_vals" begin
        giv_h = 1/4
        gaus_mu = 1
        gaus_std = 1
        f = x -> f_gaus(x, giv_h, gaus_mu, gaus_std)
        f_str = "g"
        var_n = 10 #Number of variables
 
        Random.seed!(69420)
        x0 = random_init(var_n, -10, 10) # Initial guess
        x0_cuda = CuArray(copy(x0))

        minimizer = compute_test(f, x0)
        minimizer_cuda = compute_test(f, x0_cuda)

        #Round solutions to 4 decimal places.
        @test round.(minimizer, digits=4) == round.(minimizer_cuda, digits=4)
    end

    @testset "Quadratic_vals" begin
        giv_h = 30

        f = x -> f_q(x, giv_h)
        f_str = "q"
        var_n = 10 #Number of variables
        
        Random.seed!(69420)       
        x0 = random_init(var_n, -10, 10) # Initial guess
        x0_cuda = CuArray(copy(x0))

        minimizer = compute_test(f, x0)
        minimizer_cuda = compute_test(f, x0_cuda)

        @test round.(minimizer, digits=4) == round.(minimizer_cuda, digits=4)
    end

    @testset "Gauss_sq_inp_vals" begin
        giv_h = 1/4
        gaus_mu = 1
        gaus_std = 1
        f = x -> f_gaus(x, giv_h, gaus_mu, gaus_std)
        f_str = "gs"
        var_n = 10 #Number of variables
        
        Random.seed!(69420)       
        x0 = random_init(var_n, -10, 10) # Initial guess
        x0_cuda = CuArray(copy(x0))

        minimizer = compute_test(f, x0)
        minimizer_cuda = compute_test(f, x0_cuda)

        @test round.(minimizer, digits=4) == round.(minimizer_cuda, digits=4)
    end
    @testset "Gauss_mean" begin
        giv_h = 1/4
        gaus_mu = 1
        gaus_std = 1
        f = x -> f_gaus(x, giv_h, gaus_mu, gaus_std)
        tolerance = 0.01
        f_str = "g"
        
        var_n = 10
        m_val, m_val_c = init_test(var_n, giv_h, gaus_mu, gaus_std, f, f_str, tolerance)
        @test m_val ≈ m_val_c atol = tolerance
        
        var_n = 20
        m_val, m_val_c = init_test(var_n, giv_h, gaus_mu, gaus_std, f, f_str, tolerance)
        @test m_val ≈ m_val_c atol = tolerance
        
        var_n = 40
        m_val, m_val_c = init_test(var_n, giv_h, gaus_mu, gaus_std, f, f_str, tolerance)
        @test m_val ≈ m_val_c atol = tolerance
    end

    @testset "Gauss_sq_inp_mean" begin
        giv_h = 1/4
        gaus_mu = 1
        gaus_std = 1
        f = x -> f_gaus_sq(x, giv_h, gaus_mu, gaus_std)
        tolerance = 0.01
        f_str = "gs"
        
        var_n = 10
        m_val, m_val_c = init_test(var_n, giv_h, gaus_mu, gaus_std, f, f_str, tolerance)
        @test m_val ≈ m_val_c atol = tolerance
        
        var_n = 20
        m_val, m_val_c = init_test(var_n, giv_h, gaus_mu, gaus_std, f, f_str, tolerance)
        @test m_val ≈ m_val_c atol = tolerance
        
        var_n = 40
        m_val, m_val_c = init_test(var_n, giv_h, gaus_mu, gaus_std, f, f_str, tolerance)
        @test m_val ≈ m_val_c atol = tolerance
    end

end