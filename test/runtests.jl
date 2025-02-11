using Test
using CUDA

#M is single solution size
function random_init(M::Int, min_r::T, max_r::F) where {T<:Number,F<:Number}
    return rand(M).*(max_r.-min_r) .+ min_r
end

@testset "Quadratic" begin
    f_q = sel_opt_fun("q")

    var_n = 100 #Number of variables
    x0 = random_init(var_n, -10, 10) # Initial guess
    x0_cuda = CuArray(copy(x0))
    @test compute_and_print(f_q, x0) == compute_and_print(f_q, x0_cuda)

    var_n = 500 #Number of variables
    x0 = random_init(var_n, -10, 10) # Initial guess
    x0_cuda = CuArray(copy(x0))
    @test compute_and_print(f_q, x0) == compute_and_print(f_q, x0_cuda)
end

@testset "Gauss" begin
    f_q = sel_opt_fun("g")

    var_n = 100 #Number of variables
    x0 = random_init(var_n, -10, 10) # Initial guess
    x0_cuda = CuArray(copy(x0))
    @test compute_and_print(f_q, x0) == compute_and_print(f_q, x0_cuda)

    var_n = 500 #Number of variables
    x0 = random_init(var_n, -10, 10) # Initial guess
    x0_cuda = CuArray(copy(x0))
    @test compute_and_print(f_q, x0) == compute_and_print(f_q, x0_cuda)
end

@testset "Gauss_sq_inp" begin
    f_q = sel_opt_fun("gs")

    var_n = 100 #Number of variables
    x0 = random_init(var_n, -10, 10) # Initial guess
    x0_cuda = CuArray(copy(x0))
    @test compute_and_print(f_q, x0) == compute_and_print(f_q, x0_cuda)

    var_n = 500 #Number of variables
    x0 = random_init(var_n, -10, 10) # Initial guess
    x0_cuda = CuArray(copy(x0))
    @test compute_and_print(f_q, x0) == compute_and_print(f_q, x0_cuda)
end