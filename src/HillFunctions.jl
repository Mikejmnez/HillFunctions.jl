module HillFunctions

using SparseArrays
using LinearAlgebra
using GenericSchur

export even_matrix, odd_matrix,
       even_eigvals, odd_eigvals,
       even_eigen,  odd_eigen

"""
    even_matrix(N, q, alphas)

Construct the (N+1)x(N+1) sparse "even" matrix with main diagonal 4r^2 for r=0:N,
and symmetric off-diagonals determined by `q` and `alphas`.

Convention: `alphas[2]` controls the ±1 diagonals, `alphas[3]` controls ±2, etc.
"""
function even_matrix(N::Integer, q, alphas::AbstractVector)
    N >= 0 || throw(ArgumentError("N must be ≥ 0"))

    # Promote everything to a consistent complex element type
    T = promote_type(typeof(q), Complex{eltype(alphas)}, Complex{BigFloat})
    qT = T(q)

    # Main diagonal: r = 0:N (length N+1)
    d = [T(BigFloat(4) * BigFloat(r)^2) for r in 0:N]
    A = spdiagm(0 => d)  # (N+1)x(N+1)

    # Add off-diagonals: offset = k = 1..min(N, length(alphas)-1)
    maxk = min(N, length(alphas) - 1)
    for k in 1:maxk
        α = alphas[k + 1]
        iszero(α) && continue
        len = (N + 1) - k
        vals = fill(qT * T(α), len) # constant along the diagonal in your current model
        # 🔹 Special correction for k = 1
        if k == 1
            vals[1] *= sqrt(big(2)) 
        end
        A = A + spdiagm(k => vals, -k => vals)
    end

    return A
end


"""
    odd_matrix(N, q, alphas)

Construct the (N+1)x(N+1) sparse "odd" matrix with main diagonal 4r^2 for r=1:N+1,
and the same off-diagonal convention as `even_matrix`.
"""
function odd_matrix(N::Integer, q, alphas::AbstractVector)
    N >= 0 || throw(ArgumentError("N must be ≥ 0"))

    T = promote_type(typeof(q), Complex{eltype(alphas)}, Complex{BigFloat})
    qT = T(q)

    # Main diagonal: r = 1:N+1 (length N+1)
    d = [T(BigFloat(4) * BigFloat(r)^2) for r in 1:(N + 1)]
    B = spdiagm(0 => d)

    maxk = min(N, length(alphas) - 1)
    for k in 1:maxk
        α = alphas[k + 1]
        iszero(α) && continue
        len = (N + 1) - k
        vals = fill(qT * T(α), len)
        B = B + spdiagm(k => vals, -k => vals)
    end

    return B
end

# --------------------------
# Symmetry types + pipeline
# --------------------------

abstract type Symmetry end
struct Even <: Symmetry end
struct Odd  <: Symmetry end

export Even, Odd

# Sorting rule: increasing real part, then imag part (negative imag first), then |λ|
_sortperm(vals) = sortperm(vals; by = λ -> (real(λ), imag(λ), abs(λ)))

# Build dense matrix for eigensolve (N is small; dense is fine)
_build_dense(::Type{Even}, N::Integer, q, alphas) = Matrix(even_matrix(N, q, alphas))
_build_dense(::Type{Odd},  N::Integer, q, alphas) = Matrix(odd_matrix(N, q, alphas))

# ---- Bilinear (no-conjugation) normalization of eigenvectors ----
# Normalizes EACH column v of V by:
#   nrm = fac*v[1]^2 + sum(v[2:end]^2), fac=2 for Even, 1 for Odd
# then v /= sqrt(nrm).
function _anorm_bilinear_cols!(V::AbstractMatrix, fac)
    for j in axes(V, 2)
        v = @view V[:, j]
        nrm = fac * (v[1] * v[1]) + sum(x -> x * x, @view v[2:end])
        v ./= sqrt(nrm)   # sqrt may be complex; matches NumPy behavior
    end
    return V
end

# ---- Apply Mathieu conventions to eigenvectors ----
# Even: first component scaled by 1/sqrt(2), then bilinear normalization with fac=2.
# Odd: bilinear normalization with fac=1.
function _mathieu_normalize!(::Type{Even}, V::AbstractMatrix)
    T = eltype(V)
    √2 = sqrt(real(T(2)))     # correct precision/type (e.g., BigFloat)
    V[1, :] ./= √2
    _anorm_bilinear_cols!(V, T(2))
    return V
end

function _mathieu_normalize!(::Type{Odd}, V::AbstractMatrix)
    T = eltype(V)
    _anorm_bilinear_cols!(V, T(1))
    return V
end

# ---- Core eigensolvers (generic over symmetry) ----
function _eigvals_sorted(::Type{S}, N::Integer, q, alphas) where {S<:Symmetry}
    M = _build_dense(S, N, q, alphas)
    vals = GenericSchur.eigen(M).values
    return vals[_sortperm(vals)]
end

function _eigen_sorted(::Type{S}, N::Integer, q, alphas) where {S<:Symmetry}
    M = _build_dense(S, N, q, alphas)
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
    even_eigvals(N, q, alphas; prec_bits=nothing)

Eigenvalues for the EVEN matrix, sorted by (Re, Im, |λ|).
"""
function even_eigvals(N::Integer, q, alphas::AbstractVector; prec_bits::Union{Nothing,Int}=nothing)
    _with_precision(() -> _eigvals_sorted(Even, N, q, alphas), prec_bits)
end

"""
    odd_eigvals(N, q, alphas; prec_bits=nothing)

Eigenvalues for the ODD matrix, sorted by (Re, Im, |λ|).
"""
function odd_eigvals(N::Integer, q, alphas::AbstractVector; prec_bits::Union{Nothing,Int}=nothing)
    _with_precision(() -> _eigvals_sorted(Odd, N, q, alphas), prec_bits)
end

"""
    even_eigen(N, q, alphas; prec_bits=nothing, normalize=true)

Eigenpairs for the EVEN matrix, sorted by (Re, Im, |λ|).

Applies Mathieu conventions:
1) first component of each eigenvector scaled by 1/√2
2) bilinear normalization (no conjugation) with fac=2.
"""
function even_eigen(N::Integer, q, alphas::AbstractVector;
                    prec_bits::Union{Nothing,Int}=nothing)
    _with_precision(() -> _eigen_sorted(Even, N, q, alphas), prec_bits)
end

"""
    odd_eigen(N, q, alphas; prec_bits=nothing, normalize=true)

Eigenpairs for the ODD matrix, sorted by (Re, Im, |λ|).

If `normalize=true` (default), applies bilinear normalization (no conjugation) with fac=1.
"""
function odd_eigen(N::Integer, q, alphas::AbstractVector;
                   prec_bits::Union{Nothing,Int}=nothing)
    _with_precision(() -> _eigen_sorted(Odd, N, q, alphas), prec_bits)
end

end # module HillFunctions