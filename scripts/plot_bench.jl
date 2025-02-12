using CSV, DataFrames, Plots

# Read the CSV file
df = CSV.read("benchmark_results.csv", DataFrame)

# Extract unique Num_Variables values
num_variables = unique(df.Num_Variables)

# Extract mean execution times
mean_time_cuda = [mean(df[df.Num_Variables .== n, :Mean_t][df[df.Num_Variables .== n, :CUDA] .== true]) for n in num_variables]
mean_time_no_cuda = [mean(df[df.Num_Variables .== n, :Mean_t][df[df.Num_Variables .== n, :CUDA] .== false]) for n in num_variables]

# Line plot
p = plot(
    num_variables, mean_time_no_cuda, 
    label="No CUDA", marker=:o, linestyle=:dash, lw=2, color=:blue
)
plot!(
    num_variables, mean_time_cuda, 
    label="CUDA", marker=:s, linestyle=:solid, lw=2, color=:red
)

# Labels and formatting
xlabel!("Number of Variables")
ylabel!("Execution Time (s)")
title!("CUDA vs Non-CUDA Execution Time")
yaxis!(:log10)  # Log scale for better visibility

# Save and show the plot
savefig("cuda_vs_no_cuda_lineplot.png")
display(p)
