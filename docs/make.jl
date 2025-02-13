using Documenter, lbfgsgpu


makedocs(
    sitename = "L-BFGS GPU",
    format = Documenter.HTML(prettyurls=false),
    authors = "Ondřej Matoušek",
    # modules = [lbfgsgpu],  # Specify your module name if applicable
    pages = [
        "Home" => "index.md",
        "Function list" => "fun_list.md",
        "Problem" => "problem.md",
        "Solution" => "solution.md",
        "Example" => "example.md",
        "Benchmarking" => "benchmarking.md",
        "Tests" => "tests.md",
        "Acknowledgements" => "ack.md"
    ]
)

deploydocs(
    repo = "github.com/matouon/lbfgsgpu.jl.git",
    branch = "gh-pages",
    devbranch = "main",  # Change this if your default branch is not "main"
    versions = ["stable" => "v^", "dev" => "main"]
)