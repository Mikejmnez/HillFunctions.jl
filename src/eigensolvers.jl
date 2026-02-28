# Sorting rule: increasing real part, then imag part (negative imag first), then |λ|
_sortperm(vals; digits::Int=1) =
    sortperm(vals; by = λ -> (round(real(λ); digits=digits),
                              round(imag(λ); digits=digits),
                              abs(λ)))

# Build dense matrix for eigensolve (N is small; dense is fine)
_build_dense(::Type{Even}, q, N::Integer, alphas) = Matrix(even_matrix(q, N, alphas))
_build_dense(::Type{Odd},  q, N::Integer, alphas) = Matrix(odd_matrix(q, N, alphas))

# ---- Core eigensolvers (generic over symmetry) ----
function _eigvals_sorted(::Type{S}, q, N::Integer, alphas) where {S<:Symmetry}
    M = _build_dense(S, q, N, alphas)
    vals = GenericSchur.eigen(M).values
    return vals[_sortperm(vals)]
end

function _eigen_sorted(::Type{S}, q, N::Integer, alphas) where {S<:Symmetry}
    M = _build_dense(S, q, N, alphas)
    E = GenericSchur.eigen(M)

    vals = E.values
    vecs = copy(E.vectors)  # safe to mutate for normalization

    idx = _sortperm(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    _mathieu_normalize!(S, vecs)

    return vals, vecs
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
function even_eigvals(q, N::Integer, alphas::AbstractVector; prec_bits::Union{Nothing,Int}=nothing)
    _with_precision(() -> _eigvals_sorted(Even, q, N, alphas), prec_bits)
end

"""
    odd_eigvals(q, N, alphas; prec_bits=nothing)

Eigenvalues for the ODD matrix, sorted by (Re, Im, |λ|).
"""
function odd_eigvals(q, N::Integer, alphas::AbstractVector; prec_bits::Union{Nothing,Int}=nothing)
    _with_precision(() -> _eigvals_sorted(Odd, q, N, alphas), prec_bits)
end

"""
    even_eigen(q, N, alphas; prec_bits=nothing)

Eigenpairs for the EVEN matrix, sorted by (Re, Im, |λ|).

Applies Mathieu conventions:
1) first component of each eigenvector scaled by 1/√2
2) bilinear normalization (no conjugation) with fac=2.
"""
function even_eigen(q, N::Integer, alphas::AbstractVector;
                    prec_bits::Union{Nothing,Int}=nothing)
    _with_precision(() -> _eigen_sorted(Even, q, N, alphas), prec_bits)
end

"""
    odd_eigen(q, N, alphas; prec_bits=nothing)

Eigenpairs for the ODD matrix, sorted by (Re, Im, |λ|).

Applies bilinear normalization (no conjugation) with fac=1.
"""
function odd_eigen(q, N::Integer, alphas::AbstractVector;
                   prec_bits::Union{Nothing,Int}=nothing)
    _with_precision(() -> _eigen_sorted(Odd, q, N, alphas), prec_bits)
end