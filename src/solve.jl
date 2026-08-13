"""
    Solve(q,y,bc,Fu)

Container for the result of `solve`.
"""
function solve(
    q::Union{Real,ComplexF64,Complex{BigFloat}},
    y::AbstractVector{T},
    bc::BoundaryCondition,
    fy::AbstractVector,
) where {T<:Real}

    _, α, β = _normal_mode_decomposition(fy, y)
    even_extension = any(x -> abs(x) > 1e-3, β) && bc isa Neumann


    if any(x -> abs(x) > 1e-3, β) && bc isa Neumann
        fy = vcat(fy, fy[end:-1:1])
        y = vcat(y, y[2] + y[end] .+ y)
        _, α, β = _normal_mode_decomposition(fy, y)
    end

    N = length(y)÷2

    if bc isa Neumann
        if even_extension
            an, Ak = even_eigen(q, N÷2, α)
            Φe = even_eigenfunctions(Ak, y)[1:N, :]
        else
            an, Ak = even_eigen(q, N, α)
            Φe = even_eigenfunctions(Ak, y)
        end
        CT = eltype(an)
        bn, Φo = CT[], Matrix{CT}(undef, 0, 0)

    elseif bc isa Dirichlet
        if any(x -> abs(x) > 1e-3, α)
            throw(ArgumentError("Dirichlet boundary condition requires odd `fy`"))
        end
        bn, Bk = odd_eigen(q, N, β)
        Φo = odd_eigenfunctions(Bk, y)
        CT = eltype(bn)
        an, Φe = CT[], Matrix{CT}(undef, 0, 0)
    else
        throw(ArgumentError("Unsupported boundary condition"))
    end


    return HillSolution{CT}(an, bn, Φe, Φo)
end
