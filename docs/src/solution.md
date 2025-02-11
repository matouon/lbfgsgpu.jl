# Solution

The implementation that solves the CUDA incompatibility is in `/src/Optim_dispatch.jl`.

Key modifications:
- **Array → CuArray conversion**
- **Avoiding scalar indexing** (as it reduces GPU parallelism)
- **Correctly using broadcasting (`.`) instead of loops**

# Example fix:
x .= x .+ epsilon  # Instead of `for` loops
