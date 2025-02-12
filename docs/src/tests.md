# Tests

This package is tested via direct comparison between the outputs of original `L-BFGS` and the GPU accelerated `L-BFGS`.

# Test Cases

In runtests.jl 6 tests are provided. The functionality is tested by optimizing sum of squared differences between the function outputs and a given height, which is provided as user input. The minimal value obtained is filtered to remove any outliers as convergence issues can be common in both CPU and GPU implementations. The mean value of these correctly converged solutions is then compared between the original `L-BFGS` and the `L-BFGS_GPU` implementation. The tests are done firstly for 50 variables, then for 500 variables and in the end for 1000 variables. This should ensure enough diversity for general function solution testing.

- **Gaussian SSD tests**
```math
\begin{equation}
\text{SSD}_{\text{gauss}} = \sum_{i=1}^{n} \left( A \exp\left( -\frac{(x_i - \mu)^2}{2 \sigma^2} \right) - y_i \right)^2
\end{equation}
```

- **Gaussian with squared input SSD tests**
```math
\begin{equation}
\text{SSD}_{\text{gauss-sq}} = \sum_{i=1}^{n} \left( A \exp\left( -\frac{(x_i^2 - \mu)^2}{2 \sigma^2} \right) - y_i \right)^2
\end{equation}
```

You can run test cases as usual with command : test

