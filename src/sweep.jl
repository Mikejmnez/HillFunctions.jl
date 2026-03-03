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



"""
    collect_sweep_eigen(::Type{S}, qs, N, alphas; prec_bits=nothing)

Compute eigenpairs for each `q` in `qs` and keep results in memory.

Returns `(qvec, vals, vecs)` where:
- `qvec[i]` is the i-th q
- `vals[i]` is the eigenvalue vector for qvec[i] (length N)
- `vecs[i]` is the eigenvector matrix for qvec[i] (N×N)

Notes:
- Uses the same normalization and sorting as `even_eigen` / `odd_eigen`.
- Suitable for interactive exploration and plotting.
"""
function collect_sweep_eigen(
    ::Type{S},
    qs,
    N::Integer,
    alphas;
    prec_bits::Union{Nothing,Int} = nothing,
) where {S<:Symmetry}
    nq = length(qs)  # requires sized container; see variant below if qs may not have length

    qvec = Vector{Any}(undef, nq)
    vals = Vector{Any}(undef, nq)
    vecs = Vector{Any}(undef, nq)

    cb = (iq, q, λ, V) -> begin
        qvec[iq] = q
        vals[iq] = λ
        vecs[iq] = V
    end

    sweep_eigen(S, qs, N, alphas; prec_bits = prec_bits, callback = cb)
    return qvec, vals, vecs
end


"""
    collect_sweep_eigen_dense(::Type{S}, qs::AbstractVector, N, alphas; prec_bits=nothing)

Returns `(qvec, vals, vecs)` where:
- `qvec` is a copy of `qs`
- `vals` is `nq × N`
- `vecs` is `nq × N × N`  (indexing: vecs[iq, :, :] is the N×N eigenvector matrix)
"""
function collect_sweep_eigen_dense(
    ::Type{S},
    qs::AbstractVector,
    N::Integer,
    alphas;
    prec_bits::Union{Nothing,Int} = nothing,
) where {S<:Symmetry}
    nq = length(qs)
    qvec = copy(qs)

    # compute first to infer type
    λ1, V1 = _with_precision(() -> _eigen_sorted(S, qvec[1], N, alphas), prec_bits)
    T = eltype(λ1)

    vals = Matrix{T}(undef, nq, N)
    vecs = Array{T,3}(undef, nq, N, N)

    vals[1, :] .= λ1
    vecs[1, :, :] .= V1

    for iq = 2:nq
        λ, V = _with_precision(() -> _eigen_sorted(S, qvec[iq], N, alphas), prec_bits)
        vals[iq, :] .= λ
        vecs[iq, :, :] .= V
    end

    return qvec, vals, vecs
end
