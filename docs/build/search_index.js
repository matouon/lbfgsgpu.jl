var documenterSearchIndex = {"docs":
[{"location":"tests.html#Tests","page":"Tests","title":"Tests","text":"","category":"section"},{"location":"tests.html","page":"Tests","title":"Tests","text":"This package is tested via direct comparison between the outputs of original L-BFGS and the GPU accelerated L-BFGS.","category":"page"},{"location":"tests.html#Test-Cases","page":"Tests","title":"Test Cases","text":"","category":"section"},{"location":"tests.html","page":"Tests","title":"Tests","text":"In runtests.jl 6 tests are provided. The functionality is tested by optimizing sum of squared differences between the function outputs and a given height, which is provided as user input. The minimal value obtained is filtered to remove any outliers as convergence issues can be common in both CPU and GPU implementations. The mean value of these correctly converged solutions is then compared between the original L-BFGS and the L-BFGS_GPU implementation. The tests are done firstly for 50 variables, then for 500 variables and in the end for 1000 variables. This should ensure enough diversity for general function solution testing.","category":"page"},{"location":"tests.html","page":"Tests","title":"Tests","text":"Gaussian SSD tests","category":"page"},{"location":"tests.html","page":"Tests","title":"Tests","text":"beginequation\ntextSSD_textgauss = sum_i=1^n left( A expleft( -frac(x_i - mu)^22 sigma^2 right) - y_i right)^2\nendequation","category":"page"},{"location":"tests.html","page":"Tests","title":"Tests","text":"Gaussian with squared input SSD tests","category":"page"},{"location":"tests.html","page":"Tests","title":"Tests","text":"beginequation\ntextSSD_textgauss-sq = sum_i=1^n left( A expleft( -frac(x_i^2 - mu)^22 sigma^2 right) - y_i right)^2\nendequation","category":"page"},{"location":"tests.html","page":"Tests","title":"Tests","text":"You can run test cases as usual with command : test","category":"page"},{"location":"benchmarking.html#Benchmarking-L-BFGS-Optimization-with-and-without-CUDA-Acceleration","page":"Benchmarking","title":"Benchmarking L-BFGS Optimization with and without CUDA Acceleration","text":"","category":"section"},{"location":"benchmarking.html","page":"Benchmarking","title":"Benchmarking","text":"The package can be used to evaluate the performance of the L-BFGS optimization method for various functions, both with and without CUDA acceleration. The functions are:","category":"page"},{"location":"benchmarking.html","page":"Benchmarking","title":"Benchmarking","text":"Gaussian Function \nGaussian with Squared Input\nQuadratic Function ","category":"page"},{"location":"benchmarking.html","page":"Benchmarking","title":"Benchmarking","text":"A single benchmarking test is performed with increasing solution sizes, and the execution time (mean and minimum) for both CPU and GPU implementations is recorded.","category":"page"},{"location":"benchmarking.html","page":"Benchmarking","title":"Benchmarking","text":"The performance is measured using BenchmarkTools, with benchmarking being run on both CPU and GPU (using CUDA). Results, including computation times and minimum values of the objective functions, can be stored in a DataFrame format for further analysis. The optimization function is selected based on a user string input (f_str), allowing for flexible function testing.","category":"page"},{"location":"benchmarking.html","page":"Benchmarking","title":"Benchmarking","text":"I will show the plot only for quadratic function as even for 100 variables the simple gaussian took both solvers about 30 minutes to benchmark and if I was to somehow approximate the relationship between number of variables used in the optimization and the time which it takes to finish the @benchmark test as a linear function, it would take cca 308064=24e4 seconds, which is about 3 days of continuous load for a single benchmarking test, which my parents would probably not be happy about :o).","category":"page"},{"location":"benchmarking.html","page":"Benchmarking","title":"Benchmarking","text":"The plot can be recreated using simple plotting script in scripts/plot_bench.jl","category":"page"},{"location":"benchmarking.html#Quadratic-function-for-height-40","page":"Benchmarking","title":"Quadratic function for height = 40","text":"","category":"section"},{"location":"benchmarking.html","page":"Benchmarking","title":"Benchmarking","text":"(Image: My Image)","category":"page"},{"location":"benchmarking.html","page":"Benchmarking","title":"Benchmarking","text":"The plot has logarithmic y-axis, hence the difference in time is bigger. The original values together with the measured data in table are shown below.","category":"page"},{"location":"benchmarking.html#Measured-data-for-Quadratic-function","page":"Benchmarking","title":"Measured data for Quadratic function","text":"","category":"section"},{"location":"benchmarking.html","page":"Benchmarking","title":"Benchmarking","text":"The data were rounded to 4 decimal points","category":"page"},{"location":"benchmarking.html","page":"Benchmarking","title":"Benchmarking","text":"Num_Variables CUDA Mean_t Min_t Min_Value\n100 false 0.0032 0.0009 0.0000\n100 true 0.0132 0.0117 0.0002\n200 false 0.0108 0.0037 0.0000\n200 true 0.1221 0.1097 0.0003\n2000 false 0.8175 0.7480 0.0000\n2000 true 0.0170 0.0133 0.0037\n3000 false 2.7774 1.8775 0.0000\n3000 true 0.8006 0.7481 0.0057\n5000 false 6.5454 6.5454 0.0000\n5000 true 0.0180 0.0146 0.0095\n7000 false 11.9569 11.9569 0.0000\n7000 true 0.0926 0.0822 0.0136\n10000 false 24.0851 24.0851 0.0000\n10000 true 0.0237 0.0196 0.0195","category":"page"},{"location":"benchmarking.html","page":"Benchmarking","title":"Benchmarking","text":"It is clear that the more variables we use, the more CUDA enabled solution dominates. The most extreme values tell us that even though we have on 10000 variable optimization problem,  the best (smallest) sum of squared errors was 0.0195 (which by itself is very small -> 0.0195/1e4=...), we got speedup of around 24/0.02=1200!! This means that if we optimized 10000 variable quadratic function using CUDA, it would run in average 600 times faster than on CPU.","category":"page"},{"location":"problem.html#Problem","page":"Problem","title":"Problem","text":"","category":"section"},{"location":"problem.html","page":"Problem","title":"Problem","text":"My core motivation (can show some alternative GPU use) for kernelizing L-BFGS is a real life HEP analysis problem at CERN. As described below, there is a need for an efficient and fast equation solver. The problem is rather difficult and to my current best knowledge it is not formalized as an official assignment yet. However you will have to trust me on this, but it can be reformulated as finding a set of input variables ","category":"page"},{"location":"problem.html","page":"Problem","title":"Problem","text":"x_1 x_2  x_n ","category":"page"},{"location":"problem.html","page":"Problem","title":"Problem","text":"for which a function reaches a given height (user-specified):","category":"page"},{"location":"problem.html","page":"Problem","title":"Problem","text":"f(vecx) = textgiven_height ","category":"page"},{"location":"problem.html","page":"Problem","title":"Problem","text":"which translates into:","category":"page"},{"location":"problem.html","page":"Problem","title":"Problem","text":"f(vecx) - textgiven_height = 0 ","category":"page"},{"location":"problem.html","page":"Problem","title":"Problem","text":"We formulate the objective function as:","category":"page"},{"location":"problem.html","page":"Problem","title":"Problem","text":"f_textopt = textSSD(f(vecx) textgiven_height) ","category":"page"},{"location":"problem.html","page":"Problem","title":"Problem","text":"where SSD is the sum of squared differences, making L-BFGS an ideal solver for such problem.","category":"page"},{"location":"solution.html#Solution-to-L-BFGS-kernelization","page":"Solution","title":"Solution to L-BFGS kernelization","text":"","category":"section"},{"location":"solution.html","page":"Solution","title":"Solution","text":"The implementation that solves the CUDA incompatibility is in /src/Optim_dispatch.jl.","category":"page"},{"location":"solution.html","page":"Solution","title":"Solution","text":"Key modifications:","category":"page"},{"location":"solution.html","page":"Solution","title":"Solution","text":"Array → CuArray conversion\nAvoiding scalar indexing (as it reduces GPU parallelism)\nCorrectly using broadcasting (.) instead of loops","category":"page"},{"location":"solution.html#Example-fix:","page":"Solution","title":"Example fix:","text":"","category":"section"},{"location":"solution.html","page":"Solution","title":"Solution","text":"x .= x .+ epsilon  # Instead of for loops","category":"page"},{"location":"solution.html#Example-solutions-to-underlying-HEP-problem","page":"Solution","title":"Example solutions to underlying HEP problem","text":"","category":"section"},{"location":"solution.html","page":"Solution","title":"Solution","text":"As the solutions to higher dimensional problem tend to be hard to vizualize, I have selected 3 nonlinear one dimensional functions to serve as proof of concept. As we optimize at the same time over 500 variables, it can be seen as highly-multinomial optimization. The tested functions are listed below.","category":"page"},{"location":"solution.html","page":"Solution","title":"Solution","text":"Gaussian function\nGaussian function with squared input","category":"page"},{"location":"solution.html","page":"Solution","title":"Solution","text":"500 solutions were computed using L-BFGS with CUDA and filtered so that only converged solutions were visible. The convergence issue has nothing to do with GPU as this happens also with the base L-BFGS. This is the reason why we need very fast L-BFGS, as we need more solutions than required to filter out outliers.","category":"page"},{"location":"solution.html","page":"Solution","title":"Solution","text":"This can be seen in \\scripts\\draw_sols.jl, where user can interactively choose drawn function and parameters of the functions.","category":"page"},{"location":"solution.html","page":"Solution","title":"Solution","text":"After 1 run of L-BFGS with CUDA the solutions are shown here for each respective function.","category":"page"},{"location":"solution.html","page":"Solution","title":"Solution","text":"As the underlying problem does not require precise solution, but rather an interval in which we should do additional computations, this filtering method is quite suitable for this specific problem.","category":"page"},{"location":"solution.html#Gaussian-for-height-0.25","page":"Solution","title":"Gaussian for height = 0.25","text":"","category":"section"},{"location":"solution.html","page":"Solution","title":"Solution","text":"(Image: My Image)","category":"page"},{"location":"solution.html#Gaussian-with-squared-input-for-height-0.25","page":"Solution","title":"Gaussian with squared input for height = 0.25","text":"","category":"section"},{"location":"solution.html","page":"Solution","title":"Solution","text":"(Image: My Image)","category":"page"},{"location":"ack.html","page":"Acknowledgements","title":"Acknowledgements","text":"I would like to thank my cat Luděk for his undying support in the last few days. He did not have it easy as I often programmed instead of eating and forgor to give him food as well. He will be given a complementary snack while you are reading this documentation.","category":"page"},{"location":"index.html#L-BFGS-GPU-Documentation","page":"Home","title":"L-BFGS GPU Documentation","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Welcome to the L-BFGS GPU documentation. This package provides a GPU implementation of the L-BFGS solver from Optim.jl.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"info: Info\nFor installation and usage, see the sections below.","category":"page"},{"location":"index.html#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Since L-BFGS GPU is not a registered Julia package, you need to install it directly from its GitHub repository. To do this, open the Julia REPL and run:","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"using Pkg\nPkg.add(url=\"https://github.com/matouon/lbfgsgpu.jl\")","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Alternatively, you can activate the package in an environment:","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"using Pkg\nPkg.activate(\".\")\nPkg.add(url=\"https://github.com/matouon/lbfgsgpu.jl\")","category":"page"},{"location":"index.html#Usage","page":"Home","title":"Usage","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"To use L-BFGS GPU, first load the package:","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"using lbfgsgpu","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Then, you can call the solver in the same manner as L-BFGS, but with CuArray instead of Array type of the initial solution. An example use can be seen Example slidebar or in the actual package in /scripts/lbfgs_test.jl","category":"page"},{"location":"index.html#Dependencies","page":"Home","title":"Dependencies","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"L-BFGS GPU The dependencies to run an example as described in the Example section are listed below","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"using Pkg\nPkg.add([\"Optim\", \"CUDA\", \"Random\"])\nPkg.add(url=\"https://github.com/matouon/lbfgsgpu.jl\")\n","category":"page"},{"location":"index.html#References","page":"Home","title":"References","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Optim.jl Documentation\nCUDA.jl Documentation","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"For further assistance, open an issue on the GitHub repository.","category":"page"},{"location":"example.html#Example:-Using-LBFGS-for-Optimization","page":"Example","title":"Example: Using LBFGS for Optimization","text":"","category":"section"},{"location":"example.html","page":"Example","title":"Example","text":"This example demonstrates how to use LBFGS() for optimization in Julia. A random initial solution is generated and a simple quadratic function is then optimized. ","category":"page"},{"location":"example.html#Setup","page":"Example","title":"Setup","text":"","category":"section"},{"location":"example.html","page":"Example","title":"Example","text":"To run this example, ensure you have the required packages installed:","category":"page"},{"location":"example.html","page":"Example","title":"Example","text":"using Pkg\nPkg.add([\"Optim\", \"CUDA\", \"Random\"])\nPkg.add(url=\"https://github.com/matouon/lbfgsgpu.jl\")\n","category":"page"},{"location":"example.html","page":"Example","title":"Example","text":"Example code below on minimizing a SSD of 3 quadratic functions (M=3).","category":"page"},{"location":"example.html","page":"Example","title":"Example","text":"using Optim\nusing CUDA\nusing lbfgsgpu\nusing Random\n\nfunction gaussian(x::AbstractVector, mu::Number, std::Number)\n    (1 / (std * sqrt(2 * pi))) * exp.(-((x .- mu) .^ 2) / (2 * std^2))\nend\n\nfunction gaussian(x::Number, mu::Number, std::Number)\n    (1 / (std * sqrt(2 * pi))) * exp.(-((x - mu) ^ 2) / (2 * std^2))\nend\n# Randomly initialize solution\nfunction random_init(M::Int, min_r::T, max_r::T) where {T<:Number}\n    return rand(M) .* (max_r - min_r) .+ min_r\nend\n\n# Compute solution using LBFGS\nfunction compute_and_print(f::Function, x0::AbstractVector)\n    res = optimize(f, x0, LBFGS())\n    println(\"Minimum f(x): \", Optim.minimum(res))\n    return Optim.minimizer(res)\nend\n\n#Gaussian gaussian with squared input\nfunction f_gaus_sq(x::AbstractVector{T}, given_height::F, gaus_mu::G, gaus_std::H) where {T<:Number,F<:Number,G<:Number,H<:Number}\n    # Compute the sum of the squared differences for all elements\n    return sum((gaussian(x .^ 2, gaus_mu, gaus_std) .- given_height).^2)\nend\n\n# Initialize parameters\nM, min_r, max_r = 300, -10, 10\ngaus_mu = 1\ngaus_std = 1\n\nx0 = random_init(M, min_r, max_r)\n\ngiven_height = 0.25  # given_height value for function\nf = x -> f_gaus_sq(x, given_height, gaus_mu, gaus_std)\n\n# Run optimization\ncompute_and_print(f, x0)\ncompute_and_print(f, CuArray(x0))","category":"page"}]
}
