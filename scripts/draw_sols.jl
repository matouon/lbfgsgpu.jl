using Random



#Define a function to optimize
function f(x::CUDA.CuArray{T}, given_height::F, gaus_mu::G, gaus_std::H) where {T<:Number,F<:Number,G<:Number,H<:Number}
    # Compute the sum of the absolute differences for all elements
    return sum((gaussian(x .^ 2, gaus_mu, gaus_std) .- given_height).^2)
end

function f(x::Array{T}, given_height::F, gaus_mu::G, gaus_std::H) where {T<:Number,F<:Number,G<:Number,H<:Number}
    # Compute the sum of the absolute differences for all elements
    return sum((gaussian(x .^ 2, gaus_mu, gaus_std) .- given_height).^2)
end



#quadratic functions -> not working? TODO 
function f_q(x::CUDA.CuArray{T}, given_height::F) where {T<:Number,F<:Number}
    # Compute the sum of the absolute differences for all elements
    return sum((x .^ 2 .- given_height).^2)
end

function f_q(x::Array{T}, given_height::F) where {T<:Number,F<:Number}
    # Compute the sum of the absolute differences for all elements
    return sum((x .^ 2 .- given_height).^2)
end
########################################################################################################
gauss = false #If gauss = true, obj. function is gauss. If gauss= false : objective func is quadratic.   (sum of squared diff*)
#This is for graphical purposes only
f_sel = fun_sel(gauss)

Random.seed!(69420)

M = 1 #solution size 
min_r, max_r = -10, 10 # min_range, max_range for the random initialization
given_height = 68

gaus_mu = 2
gaus_std = 1

#Sum of Gaussians
g = x -> f(x, given_height, gaus_mu, gaus_std)
#Sum of Quadratics
q = x -> f_q(x, given_height)

x0 = random_init(M, min_r, max_r) # Initial guess

sols, min = compute_and_print(g, x0, verbose=true)

sol_vals = q.([sols])

num_of_x_vals = 200
x_vals = Array(range(min_r, max_r, length=num_of_x_vals))
# y_vals = g([x_vals])  # Compute corresponding function values
y_vals = q.([x] for x in x_vals)

p = plot(x_vals, y_vals, label="g(x)", linewidth=2, legend=:topright)

# print(typeof(sols), typeof(sol_vals))

scatter!(Array(sols), sol_vals, label="Solutions", markersize=5, color=:red)

savefig(p, "plot.png")

#TODO proc to nefunguje? Display Plot sem zapnul tak wtf
# display(plot!)
# readline()


# #testcases can be done just by comparing LBFGS and cuda LBFGS results.

#TODO add to tests 2-3 functions and test on known results.


#run N times BFGS on random init solutions x0 and draw where they end. It will be feasible to visualise in 2d.
#Demo -> This TODO should be added to scripts/demo_draw.jl
