push!(LOAD_PATH,"../src/")

using Documenter, VAE

makedocs(sitename="VAE", modules=[VAE], doctest=false,
        pages = Any[
        "Home" => "index.md",
        "Tutorials" => Any[
          "tutorial/page1.md",
          "tutorial/page2.md"
         ],
         "Section2" => Any[
           "sec2/page1.md",
           "sec2/page2.md"
         ]
         ])
 
deploydocs(deps   = Deps.pip("mkdocs", "python-markdown-math"),
           repo = "github.com/briandepasquale/VAE.git",
           versions = ["stable" => "v^", "v#.#"]
          )
