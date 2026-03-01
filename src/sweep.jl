"""
    sweep_eigen(::Type{S}, qs, N, alphas; prec_bits=nothing, writer=nothing, callback=nothing)

Iterate over qs, computing eigenpairs for each q using the same logic as `even_eigen` / `odd_eigen`.

- If `writer` is provided, calls `write_step!(writer, iq, q, λ, V)` each iteration.
- If `callback` is provided, calls `callback(iq, q, λ, V)` each iteration.

Returns `nothing` (streaming workflow).
"""
function sweep_eigen(
    ::Type{S},
    qs,
    N::Integer,
    alphas;
    prec_bits::Union{Nothing,Int} = nothing,
    writer::Union{Nothing,AbstractSweepWriter} = nothing,
    callback::Union{Nothing,Function} = nothing,
) where {S<:Symmetry}
    for (iq, q) in pairs(qs)
        λ, V = _with_precision(() -> _eigen_sorted(S, q, N, alphas), prec_bits)

        callback === nothing || callback(iq, q, λ, V)
        writer === nothing || write_step!(writer, iq, q, λ, V)
    end
    return nothing
end
