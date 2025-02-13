using CUDA
using LinearAlgebra

function FiniteDiff.finite_difference_gradient!(
    df::StridedVector{<:Number},
    f,
    x::CuArray,
    cache::FiniteDiff.GradientCache{T1,T2,T3,T4,fdtype,returntype,inplace};
    relstep=FiniteDiff.default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    dir=true) where {T1,T2,T3,T4,fdtype,returntype,inplace}

    fx, c1, c2, c3 = cache.fx, cache.c1, cache.c2, cache.c3
    # df = CUDA.zeros(length(c3))  # Preallocate output array

    if fdtype != Val(:complex)
        if eltype(df) <: Complex && !(eltype(x) <: Complex)
            copyto!(c1, x)
        end
    end
    copyto!(c3, x)
    #x = [-5.143633339334399, -3.719855687175146, 1.7641842547134914] so far so good
    if fdtype == Val(:forward)
        epsilons = FiniteDiff.compute_epsilon.(fdtype, x, relstep, absstep, dir)
        if typeof(fx) != Nothing
            c3 .= x .+ epsilons
            dfi = (f(c3) - fx) ./ epsilons
            c3 .= x
        else
            fx0 = f.(x)
            c3 .= x .+ epsilons
            dfi = (f(c3) - fx0) ./ epsilons
            c3 .= x
        end
        df .= real.(dfi)

        if eltype(df) <: Complex
            if eltype(x) <: Complex
                c3 .+= im .* epsilons
                if typeof(fx) != Nothing
                    dfi = (f(c3) - fx) ./ (im .* epsilons)
                else
                    dfi = (f(c3) - fx0) ./ (im .* epsilons)
                end
                c3 .= x
            else
                c1 .= x .+ im .* epsilons
                if typeof(fx) != Nothing
                    dfi = (f(c1) - fx) ./ (im .* epsilons)
                else
                    dfi = (f(c1) - fx0) ./ (im .* epsilons)
                end
                c1 .= x
            end
            df .-= im .* imag.(dfi)
            
        end
        
    elseif fdtype == Val(:central)
        epsilons = similar(x)
        epsilons .= FiniteDiff.compute_epsilon.(fdtype, x, relstep, absstep, dir)
    
        c3i = x .- epsilons

        c_st = repeat(c3', size(c3i, 1), 1)

        B = CUDA.ones(Float64, size(c_st))
        B[diagind(B)] .= 0.0
        
        dd = CUDA.zeros(Float64, size(c_st))
        dd[diagind(dd)] .= c3 .+ epsilons
        c_st_i = c_st.*B .+ dd

        dfi = CUDA.zeros(Float64, size(c_st, 1), 1)
        dfi .= map(f, eachrow(c_st_i)) |> CuArray

        dd .= 0.0
        dd[diagind(dd)] .= c3 .- epsilons
        c_st .= c_st.*B .+ dd

        dfi .= dfi .- (map(f, eachrow(c_st)) |> CuArray)

        df .= real.(dfi ./ ( 2 .*epsilons))#vec

        if eltype(df) <: Complex
            if eltype(x) <: Complex
                c3 .+= im .* epsilons
                dfi .= f.(c3)
                c3 .= x .- im .* epsilons
                dfi .-= f.(c3)
                c3 .= x
            else
                c1 .+= im .* epsilons
                dfi .= f.(c1)
                c1 .= x .- im .* epsilons
                dfi .-= f.(c1)
                c1 .= x
            end
            df .-= im .* imag.(dfi ./ (2 .* im .* epsilons))
        end
    
    elseif fdtype == Val(:complex) && returntype <: Real && eltype(df) <: Real && eltype(x) <: Real
        copyto!(c1, x)
        epsilon_complex = eps(real(eltype(x)))
        c1_old = c1
        c1 .+= im * epsilon_complex
        df .= imag.(f.(c1)) ./ epsilon_complex
        c1 .= c1_old
    else
        fdtype_error(returntype)
    end

    df
end



# function FiniteDiff.finite_difference_gradient!(
#     df::StridedVector{<:Number},
#     f,
#     x::CuArray,
#     cache::FiniteDiff.GradientCache{T1,T2,T3,T4,fdtype,returntype,inplace};
#     relstep=FiniteDiff.default_relstep(fdtype, eltype(x)),
#     absstep=relstep,
#     dir=true) where {T1,T2,T3,T4,fdtype,returntype,inplace}

#     fx, c1, c2, c3 = cache.fx, cache.c1, cache.c2, cache.c3
#     if fdtype != Val(:complex) && eltype(df) <: Complex && !(eltype(x) <: Complex)
#         copyto!(c1, x)
#     end
#     copyto!(c3, x)
#     if fdtype == Val(:forward)
#         epsilons = FiniteDiff.compute_epsilon.(fdtype, x, relstep, absstep, dir)
#         x_old = copy(x)
#         if typeof(fx) != Nothing
#             c3 .= x .+ epsilons
#             dfi = (f(c3) - fx) ./ epsilons
#             c3 .= x_old
#         else
#             fx0 = f.(x)
#             c3 .= x .+ epsilons
#             dfi = (f(c3) - fx0) ./ epsilons
#             c3 .= x_old
#         end
#         df .= real.(dfi)

#         if eltype(df) <: Complex
#             if eltype(x) <: Complex
#                 c3 .+= im .* epsilons
                
#                 dfi = (f(c3) - fx) ./ (im .* epsilons)
#             else
#                 c1 .= x .+ im .* epsilons
#                 dfi = (f(c1) - fx) ./ (im .* epsilons)
#             end
#             df .-= im .* imag.(dfi)
            
#         end

#     elseif fdtype == Val(:central)
#         epsilons = FiniteDiff.compute_epsilon.(fdtype, x, relstep, absstep, dir)
#         x_old = copy(x)
#         c3 .+= epsilons
#         dfi = f(c3)
#         c3 .= x_old .- epsilons
#         dfi -= f(c3)
#         c3 .= x_old
#         df .= real.(dfi ./ (2 .* epsilons))
        
#         if eltype(df) <: Complex
#             if eltype(x) <: Complex
#                 c3 .+= im .* epsilons
#                 dfi = f(c3)
#                 c3 .= x_old .- im .* epsilons
#                 dfi .-= f(c3)
#                 c3 .= x_old
#             else
#                 c1 .+= im .* epsilons
#                 dfi = f(c1)
#                 c1 .= x_old .- im .* epsilons
#                 dfi .-= f(c1)
#                 c1 .= x_old
#             end
#             df .-= im .* imag.(dfi ./ (2 .* im .* epsilons))
#         end
#     else
#         fdtype_error(returntype)
#     end   
#     df
# end


# function FiniteDiff.finite_difference_gradient!(
#     df::StridedVector{<:Complex},
#     f,
#     x::CuArray{<:Complex},
#     cache::FiniteDiff.GradientCache{T1,T2,T3,T4,fdtype,returntype,inplace};
#     relstep=FiniteDiff.default_relstep(fdtype, eltype(x)),
#     absstep=relstep,
#     dir=true) where {T1,T2,T3,T4,fdtype<:Val{:complex}, returntype<:Real,inplace} 

#     fx, c1, c2, c3 = cache.fx, cache.c1, cache.c2, cache.c3
#     copyto!(c3, x)
#     copyto!(c1, x)
#     epsilon_complex = eps(real(eltype(x)))
#     c1_old = copy(c1)
#     c1 .+= im * epsilon_complex
#     df .= imag.(f(c1)) ./ epsilon_complex
#     c1 .= c1_old
#     df
#     end