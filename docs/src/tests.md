# Tests

This package is tested via direct comparison between the outputs of original `L-BFGS` and the GPU accelerated `L-BFGS`.

# Test Cases

In runtests.jl three test cases are provided. The functionality is tested by optimizing sum of squared differences between the function outputs and a given height, which is provided as user input. Each of the tests is done twice. Firstly for 100 variables and then for 500 variables. This should ensure enough diversity for general function testing.

- **Quadratic SSD tests**
```math
\begin{equation}
\text{SSD}_{\text{quad}} = \sum_{i=1}^{n} (x_i^2 - y_i)^2
\end{equation}
```

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

