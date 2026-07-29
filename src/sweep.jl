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
    progress = nothing,
    callback::Union{Nothing,Function} = nothing,
) where {S<:Symmetry}
    nq = progress === nothing ? nothing : length(qs)
    for (iq, q) in pairs(qs)
        λ, V = _with_precision(() -> _eigen_sorted(S, q, N, alphas), prec_bits)

        progress === nothing || progress(iq, nq, q)
        callback === nothing || callback(iq, q, λ, V)
        writer === nothing || write_step!(writer, iq, q, λ, V)

    end
    return nothing
end


function _adaptive_R(q, N::Integer, alphas::AbstractVector; G::Real = 7)
    Rq = q isa Real ? typeof(q) : real(typeof(q))
    R0 = promote_type(Rq, eltype(alphas))
    Tr = _realfloat_type(R0)

    RTq = Base.promote_op(abs, typeof(q))
    RTa = Base.promote_op(abs, eltype(alphas))
    RT = promote_type(RTq, RTa, Tr)

    amax = isempty(alphas) ? zero(RT) : RT(maximum(abs, alphas))
    X = RT(abs(q)) * amax
    X == zero(RT) && return 10

    R = ceil(Int, (RT(G) / RT(2)) * sqrt(X))
    return clamp(R, 10, N)
end


"""
    adaptive_eigen(::Type{S}, q, N, alphas; prec_bits=nothing, G=7, Nmax=nothing)

Compute one adaptive eigensolve using the same truncation policy as
`collect_sweep_eigen`.

For the given `q`, chooses a truncation size `R` based on:

    4R^2 >= G^2 * abs(q) * maximum(abs, alphas)

and clamps `R` to `10 <= R <= N`.

Then computes eigenpairs from an R-sized matrix:
- Even: uses `N = R` (even_matrix is R×R)
- Odd:  uses `N = R+1` so odd_matrix is R×R

Returns `(vals, vecs)`, with output element type determined by `q` and `alphas`.
"""
function adaptive_eigen(
    ::Type{S},
    q,
    N::Integer,
    alphas::AbstractVector;
    prec_bits::Union{Nothing,Int} = nothing,
    G::Real = 7,
    Nmax::Union{Nothing,Int} = nothing,
) where {S<:Symmetry}
    Tr = _base_real_type(q, alphas)
    T = q isa Real ? Tr : Complex{Tr}

    R = _adaptive_R(q, N, alphas; G)
    Nsolve = (S === Odd) ? (R + 1) : R

    λ, V = _with_precision(() -> _eigen_sorted(S, q, Nsolve, alphas; Nmax), prec_bits)

    return T.(λ), T.(V)
end


"""
    collect_sweep_eigen(::Type{S}, qs, N, alphas;
                        prec_bits=nothing, G=7)

Adaptive in-memory sweep.

For each `q`, chooses a truncation size `R` based on:

    4R^2 >= G^2 * abs(q) * maximum(abs, alphas)

and clamps `R` to `10 <= R <= N`.

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
    progress = nothing,
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

    qvec = copy(qs)                          # Vector{Q}
    vals = Vector{Vector{T}}(undef, nq)      # ragged lengths allowed
    vecs = Vector{Matrix{T}}(undef, nq)

    @inbounds for iq = 1:nq
        q = qvec[iq]
        λ, V = adaptive_eigen(S, q, N, alphas; prec_bits, G, Nmax)

        vals[iq] = eltype(λ) === T ? λ : T.(λ)
        vecs[iq] = eltype(V) === T ? V : T.(V)
        progress === nothing || progress(iq, nq, q)

    end

    return qvec, vals, vecs
end

"""
    collect_sweep_eigen_dense(::Type{S}, qs::AbstractVector, N, alphas; prec_bits=nothing)

Returns `(qvec, vals, vecs)` where:
- `qvec` is a copy of `qs`
- `vals` is `N × nq`
- `vecs` is `N × N × nq `  (indexing: vecs[:, :, iq] is the N×N eigenvector matrix)
"""
function collect_sweep_eigen_dense(
    ::Type{S},
    qs::AbstractVector{Q},
    N::Integer,
    alphas::AbstractVector;
    progress = nothing,
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
    vals = Matrix{T}(undef, N, nq)
    vecs = Array{T,3}(undef, N, N, nq)

    @inbounds for iq = 1:nq
        q = qvec[iq]
        λ, V = _with_precision(() -> _eigen_sorted(S, q, Nsolve, alphas), prec_bits)

        vals[:, iq] .= T.(λ)
        vecs[:, :, iq] .= T.(V)
        progress === nothing || progress(iq, nq, q)

    end

    return qvec, vals, vecs
end

"""
    sweep_eigvals_threaded(::Type{S}, qs, N, alphas; prec_bits=nothing, progress=nothing)

Eigenvalues-only sweep over `qs` (no eigenvectors computed), distributed across
Julia threads via `Threads.@threads`. Each `q` is solved independently with
`_eigvals_sorted`, exactly as in `even_eigvals`/`odd_eigvals` (`N` is passed
straight through: output length is `N` for `Even`, `N-1` for `Odd`).

Start Julia with more than one thread (`julia -t auto`, or set
`JULIA_NUM_THREADS`) to see a speedup; with a single thread this reduces to a
plain sequential loop.

Returns `(qvec, vals)`:
- `qvec` : copy of `qs`
- `vals` : matrix of sorted eigenvalues, one column per `q` (size `N × nq` for
  `Even`, `(N-1) × nq` for `Odd`)

`progress`, if given, is called as `progress(iq, nq, q)` after each solve.
Calls do not arrive in `iq` order under threading, so `progress` must not rely
on ordering or mutate shared state without synchronization.
"""
function sweep_eigvals_threaded(
    ::Type{S},
    qs::AbstractVector{Q},
    N::Integer,
    alphas::AbstractVector;
    prec_bits::Union{Nothing,Int} = nothing,
    progress = nothing,
) where {S<:Symmetry,Q}

    nq = length(qs)
    nq == 0 && throw(ArgumentError("qs must be non-empty"))

    Rq = Q <: Real ? Q : real(Q)
    R0 = promote_type(Rq, eltype(alphas))
    Tr = _realfloat_type(R0)
    T = (Q <: Real) ? Tr : Complex{Tr}

    qvec = copy(qs)
    Nout = (S === Odd) ? (N - 1) : N
    vals = Matrix{T}(undef, Nout, nq)

    Threads.@threads for iq = 1:nq
        q = qvec[iq]
        λ = _with_precision(() -> _eigvals_sorted(S, q, N, alphas), prec_bits)
        vals[:, iq] .= T.(λ)
        progress === nothing || progress(iq, nq, q)
    end

    return qvec, vals
end
