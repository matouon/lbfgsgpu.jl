# using CUDA
# function gpu_finite_diff!(df::CuArray, f::Function, c3::CuArray, x::CuArray, epsilons::CuArray)
#     @cuda threads=length(c3) kernel_finite_diff!(df, f, c3, x, epsilons)
#     return
# end

# function kernel_finite_diff!(df, f, c3, x, epsilons)
#     i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
#     if i <= length(c3)
#         x_old = c3[i]  # Save original value

#         # Compute finite difference approximation
#         c3[i] = x_old + epsilons[i]
#         dfi1 = f(c3)  # Ensure f(c3) returns a scalar

#         c3[i] = x_old - epsilons[i]
#         dfi2 = f(c3)  # Ensure f(c3) returns a scalar

#         c3[i] = x_old  # Restore original value

#         df[i] = (dfi1 - dfi2) / (2 * epsilons[i])  # Compute finite difference
#     end
#     return
# end


function FiniteDiff.finite_difference_gradient!(
    df::StridedVector{<:Number},
    f,
    x::CuArray,
    cache::FiniteDiff.GradientCache{T1,T2,T3,T4,fdtype,returntype,inplace};
    relstep=FiniteDiff.default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    dir=true) where {T1,T2,T3,T4,fdtype,returntype,inplace}

    fx, c1, c2, c3 = cache.fx, cache.c1, cache.c2, cache.c3
    # [-5.143633339334399, -3.719855687175146, 1.7641842547134914]nothingnothingnothing[0.0, 0.0, 0.0]
    # print(x, fx, c1, c2, c3)
    # error()

    if fdtype != Val(:complex)
        if eltype(df) <: Complex && !(eltype(x) <: Complex)

            copyto!(c1, x)
        end
    end
    copyto!(c3, x)
    #x = [-5.143633339334399, -3.719855687175146, 1.7641842547134914] so far so good
    if fdtype == Val(:forward)
        epsilons = FiniteDiff.compute_epsilon.(fdtype, x, relstep, absstep, dir)
        x_old = x
        if typeof(fx) != Nothing
            c3 .= x .+ epsilons
            dfi = (f(c3) - fx) ./ epsilons
            c3 .= x_old
        else
            fx0 = f.(x)
            c3 .= x .+ epsilons
            dfi = (f(c3) - fx0) ./ epsilons
            c3 .= x_old
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
                c3 .= x_old
            else
                c1 .= x .+ im .* epsilons
                if typeof(fx) != Nothing
                    dfi = (f(c1) - fx) ./ (im .* epsilons)
                else
                    dfi = (f(c1) - fx0) ./ (im .* epsilons)
                end
                c1 .= x_old
            end
            df .-= im .* imag.(dfi)
            
        end
        
    elseif fdtype == Val(:central)
        #inbounds???
        # print("INBB")
        # print(x, relstep, absstep)
        #[-5.143633339334399, -3.719855687175146, 1.7641842547134914]
        #6.0554544523933395e-66.0554544523933395e-6
        # err()
        epsilons = FiniteDiff.compute_epsilon.(fdtype, x, relstep, absstep, dir)
        # print(epsilons)
        # error()
        #epsilons are the same

        #x, relstep, absstep, epsilons
        #[-5.143633339334399, -3.719855687175146, 1.7641842547134914] ok
        # 6.0554544523933395e-6 6.0554544523933395e-6 ok
        # [3.114703740615131e-5, 2.2525416683165425e-5, 1.0682937400047037e-5]
        # print(x, relstep, absstep, epsilons)
        # err()
        df = CUDA.zeros(length(c3))  # Preallocate output array
        # gpu_finite_diff!(df, f, c3, x, epsilons)  # Run GPU kernel
        x_old = x
        c3 .+= epsilons
        # # print("C3",c3)
        # # err()
        # #-5.143602192296993, -3.719833161758463, 1.7641949376508914 ok
        dfi = f(c3)
        c3 .= x_old .- epsilons
        # print(c3, "  dfi  ", dfi)
        # # [-5.143664486371805, -3.719878212591829, 1.7641735717760914]
        # #   dfi  858.4516950019134 NOT OK
        dfi -= sum(f(c3))
        c3 .= x_old
        # df .= real.(dfi ./ (2 .* epsilons))
        # [-671.2491449325007, -928.1702762766218, -1957.0855321091392]
        # [-5.143633339334399, -3.719855687175146, 1.7641842547134914]
        # [-0.04181484445211936, -0.04181484445211936, -0.04181484445211936]
        # print("TADY") 
        #FIRST ENTRY TO FUNCTION
        # print(df, c3, dfi)
        # err()
        if eltype(df) <: Complex
            if eltype(x) <: Complex
                c3 .+= im .* epsilons
                dfi .= f.(c3)
                c3 .= x_old .- im .* epsilons
                dfi .-= f.(c3)
                c3 .= x_old
            else
                c1 .+= im .* epsilons
                dfi .= f.(c1)
                c1 .= x_old .- im .* epsilons
                dfi .-= f.(c1)
                c1 .= x_old
            end
            df .-= im .* imag.(dfi ./ (2 .* im .* epsilons))
        end
    
    elseif fdtype == Val(:complex) && returntype <: Real && eltype(df) <: Real && eltype(x) <: Real
        # print("AAAAAAAAAA")
        # err()
        copyto!(c1, x)
        epsilon_complex = eps(real(eltype(x)))
        c1_old = c1
        c1 .+= im * epsilon_complex
        df .= imag.(f.(c1)) ./ epsilon_complex
        c1 .= c1_old
    else
        fdtype_error(returntype)
    end
    #[-671.2491449325007, -928.1702762766218, -1957.0855321091392]
    # print(df, "HERE")
    # error()
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