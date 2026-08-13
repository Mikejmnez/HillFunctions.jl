# --------------------------
# Boundary conditions
# --------------------------

abstract type BoundaryCondition end
struct Neumann <: BoundaryCondition end
struct Dirichlet <: BoundaryCondition end
struct Robin{T} <: BoundaryCondition
    α::T
    β::T
end

"""
    HillSolution{T}

Container for the result of `solve`.

- `an` : even eigenvalues (empty `Vector{T}` if the even problem was not solved)
- `bn` : odd eigenvalues (empty `Vector{T}` if the odd problem was not solved)
- `Φe` : even eigenfunctions evaluated on the `y` grid, size `(length(y), length(an))`
         (empty `Matrix{T}` if the even problem was not solved)
- `Φo` : odd eigenfunctions evaluated on the `y` grid, size `(length(y), length(bn))`
         (empty `Matrix{T}` if the odd problem was not solved)
"""
struct HillSolution{T}
    an::Vector{T}
    bn::Vector{T}
    Φe::Matrix{T}
    Φo::Matrix{T}
end

# Convenience constructor for an all-empty solution of element type T.
HillSolution(::Type{T}) where {T} =
    HillSolution{T}(T[], T[], Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0))
