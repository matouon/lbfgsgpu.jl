################
The goal of this package is to provide a GPU implementation of L-BFGS solver from Optim.jl

The base L-BFGS is directly used from https://github.com/JuliaNLSolvers/Optim.jl/blob/master/src/multivariate/solvers/first_order/l_bfgs.jl


# Problem statement

The base problem for the kernelization of L-BFGS is physics inspired requierement for efficient and fast equation solver. Without any difficult description, the problem can be reformulated as finding set of input variables (x_1, x_2, ..., x_n), for which a function is at a given height (user specified). We hence try to solve f(\vec{x}) = given_height, which can be translated into  f(\vec{x}) - given_height = 0. 

# Solution
The solution to this problem can be formulated as sum of squared differences between each element from vector x and a scalar value in this manner -> f_opt = SSD(f(\vec{x}), given_height), where f_opt is objective function, which we want to minimize. Because we can formulate the problem as minimization of some SSD, we can directly use L-BFGS as this is the ideal function to solve such problem. 

The implemented function that solved the CUDA incompability is in /src/Optim_dispatch.jl

Apart from the obvious change of type Array - > CuArray,
the only issue was scalar indexing, which is not very good for GPU as it basically cancels the parallelism. The fix was correct and logical use of broadcasting. 

# Benchmarking
To obtain some basic benchmarking, run /scripts/gpu_bench.jl


# Tests

You can test this package with provided testsets.
