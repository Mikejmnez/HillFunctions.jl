#!/usr/bin/env julia

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using JuliaFormatter

format(".", verbose = false)
