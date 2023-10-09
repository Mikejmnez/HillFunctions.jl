using HillFunctions
using Documenter

DocMeta.setdocmeta!(HillFunctions, :DocTestSetup, :(using HillFunctions); recursive=true)

makedocs(;
    modules=[HillFunctions],
    authors="Miguel A. Jimenez-Urias",
    repo="https://github.com/Mikejmnez/HillFunctions.jl/blob/{commit}{path}#{line}",
    sitename="HillFunctions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Mikejmnez.github.io/HillFunctions.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Mikejmnez/HillFunctions.jl",
    devbranch="main",
)
