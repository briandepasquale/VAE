push!(LOAD_PATH,"../src/")

using Documenter, VAE

makedocs(sitename="VAE.jl",modules=[VAE], doctest=false)
 
deploydocs(deps   = Deps.pip("mkdocs", "python-markdown-math"),
           repo = "github.com/briandepasquale/VAE.jl.git")
