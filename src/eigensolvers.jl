# Sorting rule: increasing real part, then imag part (negative imag first), then |λ|
_sortperm(vals; digits::Int = 1) = sortperm(
    vals;
    by = λ ->
        (round(real(λ); digits = digits), round(imag(λ); digits = digits), abs(λ)),
)

# Build dense matrix for eigensolve (N is small; dense is fine)
_build_dense(::Type{Even}, q, N::Integer, alphas) = even_matrix(q, N, alphas)
_build_dense(::Type{Odd}, q, N::Integer, alphas) = odd_matrix(q, N, alphas)

# ---- Core eigensolvers (generic over symmetry) ----
function _eigvals_sorted(::Type{S}, q, N::Integer, alphas) where {S<:Symmetry}
    M = _build_dense(S, q, N, alphas)
    vals = eigvals(M)
    return vals[_sortperm(vals)]
end

function _eigen_sorted(
    ::Type{S},
    q,
    N::Integer,
    alphas;
    Nmax::Union{Nothing,Int} = nothing,
) where {S<:Symmetry}
    M = _build_dense(S, q, N, alphas)
    E = GenericSchur.eigen(M)

    vals = E.values
    vecs = copy(E.vectors)

    # sort eigenvalues of q is not real
    if !isreal(q)
        idx = _sortperm(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
    end

    _mathieu_normalize!(S, vecs)

    if Nmax === nothing
        return vals, vecs
    else
        n = min(Nmax, length(vals))
        return vals, vecs[1:n, 1:end]
    end
end

# ---- Precision-scoped wrappers ----
function _with_precision(f, prec_bits::Union{Nothing,Int})
    prec_bits === nothing ? f() : setprecision(BigFloat, prec_bits) do
        f()
    end
end

# --------------------------
# Public API (sorted by default)
# --------------------------

"""
    even_eigvals(q, N, alphas; prec_bits=nothing)

Eigenvalues for the EVEN matrix, sorted by (Re, Im, |λ|).
"""
function even_eigvals(
    q,
    N::Integer,
    alphas::AbstractVector;
    prec_bits::Union{Nothing,Int} = nothing,
)
    _with_precision(() -> _eigvals_sorted(Even, q, N, alphas), prec_bits)
end

"""
    odd_eigvals(q, N, alphas; prec_bits=nothing)

Eigenvalues for the ODD matrix, sorted by (Re, Im, |λ|).
"""
function odd_eigvals(
    q,
    N::Integer,
    alphas::AbstractVector;
    prec_bits::Union{Nothing,Int} = nothing,
)
    _with_precision(() -> _eigvals_sorted(Odd, q, N, alphas), prec_bits)
end

"""
    even_eigen(q, N, alphas; prec_bits=nothing)

Eigenpairs for the EVEN matrix, sorted by (Re, Im, |λ|).

Applies Mathieu conventions:
1) first component of each eigenvector scaled by 1/√2
2) bilinear normalization (no conjugation) with fac=2.
"""
function even_eigen(
    q,
    N::Integer,
    alphas::AbstractVector;
    prec_bits::Union{Nothing,Int} = nothing,
    Nmax::Union{Nothing,Int} = nothing,
)
    _with_precision(() -> _eigen_sorted(Even, q, N, alphas; Nmax), prec_bits)
end

"""
    odd_eigen(q, N, alphas; prec_bits=nothing)

Eigenpairs for the ODD matrix, sorted by (Re, Im, |λ|).

Applies bilinear normalization (no conjugation) with fac=1.
"""
function odd_eigen(
    q,
    N::Integer,
    alphas::AbstractVector;
    prec_bits::Union{Nothing,Int} = nothing,
    Nmax::Union{Nothing,Int} = nothing,
)
    _with_precision(() -> _eigen_sorted(Odd, q, N, alphas; Nmax), prec_bits)
end

# --------------------------
# Eigenvector correction near Exceptional Points (pilot test)
# --------------------------

# Undo the Mathieu-convention scaling on a single already-normalized
# eigenvector, recovering a vector parallel to the pre-normalization raw
# eigenvector (up to the irrelevant, uniform, per-column complex scalar that
# `_anorm_bilinear_cols!` divides out — see `src/normalize.jl`). Only `Even`
# needs undoing: it scales component 1 by `1/√2` before the (direction
# -preserving) bilinear-norm division; `Odd` applies only that uniform
# division, so it is a no-op.
_undo_mathieu_scale(::Type{Even}, v::AbstractVector) =
    (u = copy(v); u[1] *= sqrt(real(eltype(v))(2)); u)
_undo_mathieu_scale(::Type{Odd}, v::AbstractVector) = copy(v)

# Inverse iteration seeded from a known starting vector `v0` (rather than a
# random vector), so it can be warm-started from an existing, possibly
# ill-conditioned eigenvector estimate. `M = A - λI` is factorized once and
# reused every iteration (`M` doesn't change), since each factorization is
# the dominant cost — especially at BigFloat precision.
#
# When `λ` is already an accurate eigenvalue of `A` (the well-conditioned
# case), `A - λI` is singular to full working precision, and `M \ v` is
# dominated by rounding noise rather than signal — the iteration doesn't
# converge, it oscillates. So candidates are only accepted if they actually
# reduce the eigenvector residual `‖Av - λv‖` relative to the best seen so
# far (starting from the seed itself); an already-good seed can't be
# displaced by an unstable solve, while a genuinely bad seed still gets
# refined.
function _inverse_iteration(A, λ, v0; max_iter::Integer = 20, tol::Real = 1e-12)
    resid(u) = norm(A * u - λ * u)

    v = normalize(v0)
    best, best_resid = v, resid(v)
    F = lu(A - λ * I)
    for _ = 1:max_iter
        v_new = normalize(F \ v)
        r_new = resid(v_new)
        r_new < best_resid && ((best, best_resid) = (v_new, r_new))
        (norm(v_new - v) < tol || norm(v_new + v) < tol) && break
        v = v_new
    end
    return best
end

# Rotate `v` by a unit-modulus scalar to best align its phase with `v_ref`
# (minimizes ‖αv - v_ref‖ over |α|=1). Inverse iteration only recovers an
# eigenvector up to an arbitrary overall phase/sign, so without this,
# correcting one member of a near-degenerate pair can flip its sign relative
# to the other, corrupting their cross term even though each individually is
# now accurate.
function _align_phase(v::AbstractVector, v_ref::AbstractVector)
    c = dot(v, v_ref)
    iszero(c) && return v
    return (c / abs(c)) .* v
end

"""
    correct_eigenvector(::Type{S}, Ak, vals, vals_big, q, alphas, n;
                         max_iter=20, tol=1e-12, mismatch_rtol=1e-2)

Correct the near-degenerate pair of eigenvectors `Ak[:, n]` and `Ak[:, n+1]`
(column convention matching `GenericSchur.eigen(A).vectors` and
`_anorm_bilinear_cols!`), using the trusted, higher-precision eigenvalues as
anchors for inverse iteration. `Ak` is left unmodified; a new, corrected
`ComplexF64` matrix is returned with only columns `n` and `n+1` changed.

- `Ak`       : matrix of all `ComplexF64` eigenvectors, one per column.
- `vals`     : all `ComplexF64` eigenvalues, in the same order as `Ak`'s columns.
- `vals_big` : "true" eigenvalues (e.g. computed with `prec_bits` set), same order as `vals`.
- `q, alphas` : parameters used to rebuild the dense operator
  `A_big = _build_dense(S, big(q), size(Ak, 2), alphas)` at BigFloat precision
  — built once and reused for both modes; the truncation size is taken from
  `Ak` itself, since it must match the number of eigenvector columns.
- `n`        : index of the first mode of the pair to correct (`n+1` must also
  be a valid column of `Ak`).

The correction runs entirely at BigFloat precision (matrix, shift, and seed
all promoted), not just the eigenvalue: `vals_big[n]` is only an eigenvalue
of the *matrix built at that same precision*, not of the `ComplexF64` matrix
— using it as a shift for a `ComplexF64` solve makes `A - λI` singular to
Float64's full working precision purely from that representation mismatch,
independent of whether the eigenvector itself is genuinely ill-conditioned,
which corrupts the result instead of fixing it. Reconstructing the operator
from the (potentially ill-conditioned) eigenvectors themselves would just
recycle the same error, so it is always rebuilt from `q`/`alphas`.

`vals[n]` and `vals_big[n]` are expected to refer to the same mode for every
`n`; near a degenerate pair the two eigen-solves (different precisions) can
sort modes in swapped order, so each pair is checked against `mismatch_rtol`
— relative to `abs(vals_big[n])`, since eigenvalues here range from O(1) to
O(2000+) and a single absolute tolerance can't cover both — before that
column is corrected, to avoid silently correcting the wrong eigenvector.

Inverse iteration only recovers an eigenvector up to an arbitrary overall
phase/sign; each corrected eigenvector is rotated to align its phase with its
own seed before being stored, so that a near-degenerate partner's cross term
doesn't pick up a spurious sign flip.

Returns a new matrix with columns `n` and `n+1` replaced by their corrected
eigenvectors, restored to the same Mathieu-normalized convention.
"""
function correct_eigenvector(
    ::Type{S},
    Ak::AbstractMatrix,
    vals::AbstractVector,
    vals_big::AbstractVector,
    q,
    alphas,
    n::Integer;
    max_iter::Integer = 20,
    tol::Real = 1e-12,
    mismatch_rtol::Real = 1e-2,
) where {S<:Symmetry}
    n_modes = size(Ak, 2)
    length(vals) == n_modes || throw(
        ArgumentError("length(vals) = $(length(vals)) must match size(Ak,2) = $n_modes"),
    )
    length(vals_big) == n_modes || throw(
        ArgumentError(
            "length(vals_big) = $(length(vals_big)) must match size(Ak,2) = $n_modes",
        ),
    )
    1 <= n && n + 1 <= n_modes || throw(
        ArgumentError("n and n+1 must both be valid columns of Ak (n_modes = $n_modes)"),
    )

    A_big = _build_dense(S, big(q), n_modes, alphas)
    Ak_corrected = copy(Ak)

    for m in (n, n + 1)
        λ_big = vals_big[m]

        Δ = abs(vals[m] - ComplexF64(λ_big))
        mismatch_tol = mismatch_rtol * abs(λ_big)
        Δ > mismatch_tol && throw(
            ArgumentError(
                "vals[$m] = $(vals[m]) and vals_big[$m] = $(vals_big[m]) differ by " *
                "$Δ > mismatch_rtol * abs(vals_big[$m]) = $mismatch_tol; they may refer " *
                "to different modes (check sorting/indexing), or mismatch_rtol needs raising.",
            ),
        )

        v_seed = _undo_mathieu_scale(S, Complex{BigFloat}.(Ak[:, m]))
        v_corr = _inverse_iteration(A_big, λ_big, v_seed; max_iter, tol)
        v_corr = _align_phase(v_corr, v_seed)

        Vcol = reshape(v_corr, :, 1)
        _mathieu_normalize!(S, Vcol)
        Ak_corrected[:, m] .= ComplexF64.(vec(Vcol))
    end

    return Ak_corrected
end
