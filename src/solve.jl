"""
    Solve(q,y,bc,Fu)

Container for the result of `solve`.
"""
function solve(
    q::Union{Real,ComplexF64,Complex{BigFloat}},
    y::AbstractVector{T},
    bc::BoundaryCondition,
    alphas::AbstractVector,
) where {T<:Real}

    N = length(y)÷2
    if bc isa Neumann
        an, Ak = even_eigen(q, N, alphas)
        Φe = even_eigenfunctions(Ak, y)
        CT = eltype(an)
        bn, Φo = CT[], Matrix{CT}(undef, 0, 0)
    elseif bc isa Dirichlet
        bn, Bk = odd_eigen(q, N, alphas)
        Φo = odd_eigenfunctions(Bk, y)
        CT = eltype(bn)
        an, Φe = CT[], Matrix{CT}(undef, 0, 0)
    else
        throw(ArgumentError("Unsupported boundary condition"))
    end


    return HillSolution{CT}(an, bn, Φe, Φo)
end
