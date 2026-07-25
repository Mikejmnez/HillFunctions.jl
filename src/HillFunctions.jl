module HillFunctions

using SparseArrays
using LinearAlgebra
using GenericSchur

include("core.jl")
include("normalize.jl")
include("eigensolvers.jl")
include("io_api.jl")
include("sweep.jl")

export even_matrix,
    odd_matrix,
    even_eigvals,
    odd_eigvals,
    even_eigen,
    odd_eigen,
    adaptive_eigen,
    even_eigenfunctions,
    odd_eigenfunctions,
    Even,
    Odd,
    sweep_eigen,
    collect_sweep_eigen,
    collect_sweep_eigen_dense,
    sweep_even_eigenfunctions,
    sweep_odd_eigenfunctions
AbstractSweepWriter, write_step!, close_writer

end
