using Test
using CUDA
using Optim
using lbfgsgpu

#M is single solution size
function random_init(M::Int, min_r::T, max_r::F) where {T<:Number,F<:Number}
    return rand(M).*(max_r.-min_r) .+ min_r
end


function gaussian(x::AbstractVector, mu::Number, std::Number)
    (1/(std*sqrt(2*pi))).*exp.(-((x.-mu).^2)./(2*std^2))
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
    minimum = Optim.minimum(res)
    minimum
end
@testset "function optimization tests" begin

    @testset "Quadratic" begin
        f = x -> f_q(x, 1/4)

        var_n = 100 #Number of variables
        x0 = random_init(var_n, -10, 10) # Initial guess
        x0_cuda = CuArray(copy(x0))
        # @test compute_test(f, x0) == compute_test(f, x0_cuda)
        @test compute_test(f, x0) ≈ compute_test(f, x0_cuda) atol=1e-2

        var_n = 500 #Number of variables
        x0 = random_init(var_n, -10, 10) # Initial guess
        x0_cuda = CuArray(copy(x0))
        # @test compute_test(f, x0) == compute_test(f, x0_cuda)
        @test compute_test(f, x0) ≈ compute_test(f, x0_cuda) atol=1e-2

    end

    @testset "Gauss" begin
        f = x -> f_gaus(x, 1/4, 1, 1)

        var_n = 100 #Number of variables
        x0 = random_init(var_n, -10, 10) # Initial guess
        x0_cuda = CuArray(copy(x0))
        @test compute_test(f, x0) ≈ compute_test(f, x0_cuda) atol=1e-2

        var_n = 500 #Number of variables
        x0 = random_init(var_n, -10, 10) # Initial guess
        x0_cuda = CuArray(copy(x0))
        @test compute_test(f, x0) ≈ compute_test(f, x0_cuda) atol=1e-2
    end

    @testset "Gauss_sq_inp" begin
        f = x -> f_gaus_sq(x, 1/4, 1, 1)

        var_n = 100 #Number of variables
        x0 = random_init(var_n, -10, 10) # Initial guess
        x0_cuda = CuArray(copy(x0))
        @test compute_test(f, x0) ≈ compute_test(f, x0_cuda) atol=1e-2

        var_n = 500 #Number of variables
        x0 = random_init(var_n, -10, 10) # Initial guess
        x0_cuda = CuArray(copy(x0))
        @test compute_test(f, x0) ≈ compute_test(f, x0_cuda) atol=1e-2
    end
end