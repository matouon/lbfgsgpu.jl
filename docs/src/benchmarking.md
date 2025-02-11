# Benchmarking L-BFGS Optimization with and without CUDA Acceleration

The package evaluates the performance of the L-BFGS optimization method for various functions, both with and without CUDA acceleration. The functions being tested are:

- **Gaussian Function** 
- **Gaussian with Squared Input**
- **Quadratic Function** 

Each function is tested with increasing solution sizes, and the execution time (mean and minimum) for both CPU and GPU implementations is recorded.

The performance is measured using BenchmarkTools, with benchmarking being run on both CPU and GPU (using CUDA). Results, including computation times and minimum values of the objective functions, are stored in a DataFrame format for analysis. The optimization function is selected based on a user string input (f_str), allowing for flexible function testing.