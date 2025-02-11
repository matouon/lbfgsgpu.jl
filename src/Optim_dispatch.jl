#https://github.com/JuliaDiff/FiniteDiff.jl/blob/a7eca2d4b73c4de12140d89df7621fcc90d29190/src/gradients.jl#L183
function FiniteDiff.finite_difference_gradient!(
    df,
    f,
    x::CuArray,
    cache::FiniteDiff.GradientCache{T1,T2,T3,T4,fdtype,returntype,inplace};
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    dir=true) where {T1,T2,T3,T4,fdtype,returntype,inplace}

    fx, c1, c2, c3 = cache.fx, cache.c1, cache.c2, cache.c3
    if fdtype != Val(:complex) && eltype(df) <: Complex && !(eltype(x) <: Complex)
        copyto!(c1, x)
    end
    copyto!(c3, x)

    if fdtype == Val(:forward)
        epsilons = compute_epsilon.(fdtype, x, relstep, absstep, dir)
        x_old = copy(x)
        c3 .= x .+ epsilons
        dfi = (f(c3) .- fx) ./ epsilons
        df .= real.(dfi)
        c3 .= x_old
        
        if eltype(df) <: Complex
            if eltype(x) <: Complex
                c3 .+= im .* epsilons
                dfi = (f(c3) .- fx) ./ (im .* epsilons)
            else
                c1 .= x .+ im .* epsilons
                dfi = (f(c1) .- fx) ./ (im .* epsilons)
            end
            df .-= im .* imag.(dfi)
        end
    
    elseif fdtype == Val(:central)
        epsilons = compute_epsilon.(fdtype, x, relstep, absstep, dir)
        x_old = copy(x)
        c3 .+= epsilons
        dfi = f(c3)
        c3 .= x_old .- epsilons
        dfi -= f(c3)
        c3 .= x_old
        df .= real.(dfi ./ (2 .* epsilons))
        
        if eltype(df) <: Complex
            if eltype(x) <: Complex
                c3 .+= im .* epsilons
                dfi = f(c3)
                c3 .= x_old .- im .* epsilons
                dfi .-= f(c3)
                c3 .= x_old
            else
                c1 .+= im .* epsilons
                dfi = f(c1)
                c1 .= x_old .- im .* epsilons
                dfi .-= f(c1)
                c1 .= x_old
            end
            df .-= im .* imag.(dfi ./ (2 .* im .* epsilons))
        end
    
    elseif fdtype == Val(:complex) && returntype <: Real && eltype(df) <: Real && eltype(x) <: Real
        copyto!(c1, x)
        epsilon_complex = eps(real(eltype(x)))
        c1_old = copy(c1)
        c1 .+= im * epsilon_complex
        df .= imag.(f(c1)) ./ epsilon_complex
        c1 .= c1_old
    else
        fdtype_error(returntype)
    end
    
    df
end
