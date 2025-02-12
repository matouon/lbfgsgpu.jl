# L-BFGS GPU Documentation

Welcome to the **L-BFGS GPU** documentation. This package provides a GPU implementation of the L-BFGS solver from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).

!!! info
    For installation and usage, see the sections below.

## Installation

Since `L-BFGS GPU` is not a registered Julia package, you need to install it directly from its GitHub repository. To do this, open the Julia REPL and run:

```julia
using Pkg
Pkg.add(url="https://github.com/matouon/lbfgsgpu.jl")
```

Alternatively, you can activate the package in an environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.add(url="https://github.com/matouon/lbfgsgpu.jl")
```

## Usage

To use `L-BFGS GPU`, first load the package:

```julia
using lbfgsgpu
```

Then, you can call the solver in the same manner as `L-BFGS`, but with CuArray instead of Array type of the initial solution.
An example use can be seen `Example` slidebar or in the actual package in /scripts/lbfgs_test.jl

## Dependencies

`L-BFGS GPU` requires among others CUDA.jl for GPU acceleration. Ensure you have CUDA installed and properly configured for Julia:

```julia
using Pkg
Pkg.add("CUDA")
using CUDA
CUDA.version()
```

## References

- [Optim.jl Documentation](https://julianlsolvers.github.io/Optim.jl/stable/)
- [CUDA.jl Documentation](https://cuda.juliagpu.org/stable/)

For further assistance, open an issue on the [GitHub repository](https://github.com/matouon/lbfgsgpu.jl/issues).
