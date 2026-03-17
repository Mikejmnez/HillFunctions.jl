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
    collect_sweep_eigen(::Type{S}, qs, N, alphas;
                        prec_bits=nothing, G=7)

Adaptive in-memory sweep.

For each `q`, chooses a truncation size `R` based on:

    4R^2 >= G^2 * abs(q) * maximum(abs, alphas)

and clamps `R` to `2 <= R <= N`.

Then computes eigenpairs from an R-sized matrix:
- Even: uses `N = R` (even_matrix is R×R)
- Odd:  uses `N = R+1` so odd_matrix is R×R

Returns `(qvec, vals, vecs)`:
- `qvec[i]`  = q
- `vals[i]`  = eigenvalues vector (length R)
- `vecs[i]`  = eigenvectors matrix (size R×R)

Note: `R` can be recovered as `length(vals[i])`.
"""
function collect_sweep_eigen(
    ::Type{S},
    qs::AbstractVector{Q},
    N::Integer,
    alphas::AbstractVector;
    prec_bits::Union{Nothing,Int} = nothing,
    G::Real = 7,
    Nmax::Union{Nothing,Int} = nothing,
) where {S<:Symmetry,Q}

    nq = length(qs)
    nq == 0 && return Q[], Vector{Vector{Any}}(), Vector{Matrix{Any}}()

    Rq = Q <: Real ? Q : real(Q)
    R0 = promote_type(Rq, eltype(alphas))
    Tr = _realfloat_type(R0)

    # Matrix/eigensystem element type dictated by q being real vs complex
    T = (Q <: Real) ? Tr : Complex{Tr}

    RTq = Base.promote_op(abs, Q)
    RTa = Base.promote_op(abs, eltype(alphas))
    RT = promote_type(RTq, RTa, Tr)

    amax = isempty(alphas) ? zero(RT) : RT(maximum(abs, alphas))

    @inline function estimate_R(q::Q)::Int
        X = RT(abs(q)) * amax
        X == zero(RT) && return 10
        R = ceil(Int, (RT(G) / RT(2)) * sqrt(X))
        return clamp(R, 10, N)
    end

    qvec = copy(qs)                          # Vector{Q}
    vals = Vector{Vector{T}}(undef, nq)      # ragged lengths allowed
    vecs = Vector{Matrix{T}}(undef, nq)

    @inbounds for iq = 1:nq
        q = qvec[iq]
        R = estimate_R(q)
        Nsolve = (S === Odd) ? (R + 1) : R

        λ, V = _with_precision(() -> _eigen_sorted(S, q, Nsolve, alphas; Nmax), prec_bits)

        # Force to the target element type T (ensures invariants like Complex{BigFloat} when q is that type)
        vals[iq] = T.(λ)
        vecs[iq] = T.(V)
    end

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
    qs::AbstractVector{Q},
    N::Integer,
    alphas::AbstractVector;
    prec_bits::Union{Nothing,Int} = nothing,
) where {S<:Symmetry,Q}

    nq = length(qs)
    nq == 0 && throw(ArgumentError("qs must be non-empty"))

    # internal solve size: Odd uses N+1 so odd_matrix is N×N
    Nsolve = (S === Odd) ? (N + 1) : N

    # type driven by q + alphas
    Rq = Q <: Real ? Q : real(Q)
    R0 = promote_type(Rq, eltype(alphas))
    Tr = _realfloat_type(R0)
    T = (Q <: Real) ? Tr : Complex{Tr}

    qvec = copy(qs)
    vals = Matrix{T}(undef, nq, N)
    vecs = Array{T,3}(undef, nq, N, N)

    @inbounds for iq = 1:nq
        q = qvec[iq]
        λ, V = _with_precision(() -> _eigen_sorted(S, q, Nsolve, alphas), prec_bits)

        vals[iq, :] .= T.(λ)
        vecs[iq, :, :] .= T.(V)
    end

    return qvec, vals, vecs
end
