module HillFunctions

using SparseArrays
using LinearAlgebra
using GenericSchur

export even_matrix, odd_matrix,
       even_eigvals, odd_eigvals,
       even_eigen,  odd_eigen, 
       even_eigenfunctions,
       odd_eigenfunctions

# helper: ensure sqrt(2) etc never forces Complex{Int} matrices
_realfloat_type(::Type{T}) where {T} = T <: Integer ? Float64 : T

# Ensure alphas has length at least L by padding with zeros (no truncation).
function _pad_alphas(alphas::AbstractVector, L::Integer, ::Type{CT}) where {CT}
    if length(alphas) >= L
        return alphas  # keep as-is; we'll treat out-of-range as 0 anyway
    end
    out = Vector{CT}(undef, L)
    @inbounds begin
        for i in 1:length(alphas)
            out[i] = CT(alphas[i])
        end
        for i in (length(alphas)+1):L
            out[i] = zero(CT)
        end
    end
    return out
end


# Decide base real float type (preserves BigFloat if present, avoids Int)
function _base_real_type(q, alphas)
    Rq = q isa Real ? typeof(q) : real(typeof(q))          # e.g. Int64 for 100im, BigFloat for big(1)+0im
    R0 = promote_type(Rq, eltype(alphas))
    return _realfloat_type(R0)
end

# Decide matrix element type from q (Real -> real matrix; Complex -> complex matrix)
_matrix_eltype(q, Tr::Type) = q isa Real ? Tr : Complex{Tr}


"""
    even_matrix(q, N, alphas)

Construct the EVEN matrix in dense form.

- Input N is the size parameter; output is N×N.
- Out-of-range α_k are treated as 0.
- If `q` is real, the matrix is real symmetric (`Matrix{Tr}`).
- If `q` is complex (e.g. purely imaginary), the matrix is complex symmetric (`Matrix{Complex{Tr}}`).
"""
function even_matrix(q, N::Integer, alphas::AbstractVector)
    N ≥ 2 || throw(ArgumentError("N must be ≥ 2"))

    need = 2*(N-1)                # need up to α_{2(N-1)}
    Tr = _base_real_type(q, alphas)
    MT = _matrix_eltype(q, Tr)

    qT = MT(q)
    alphasT = _pad_alphas(alphas, need, MT)

    α(k::Integer) = (1 ≤ k ≤ length(alphasT)) ? alphasT[k] : zero(MT)

    A = zeros(MT, N, N)
    sqrt2 = MT(sqrt(Tr(2)))

    # first row/col: c = 1..N-1
    @inbounds for c in 1:(N-1)
        v = sqrt2 * qT * α(c)
        A[1, c+1] = v
        A[c+1, 1] = v
    end

    # block r,c = 1..N-1 (math indices), Julia i=r+1
    @inbounds for r in 1:(N-1)
        i = r + 1
        A[i, i] = MT(4) * MT(r)^2 + qT * α(2r)

        for c in (r+1):(N-1)
            j = c + 1
            v = qT * (α(abs(r - c)) + α(r + c))
            A[i, j] = v
            A[j, i] = v
        end
    end

    return q isa Real ? Symmetric(A) : A
end


"""
    odd_matrix(q, N, alphas)

Construct the ODD matrix in dense form.

- Input N is the size parameter; output is (N-1)×(N-1).
- Out-of-range α_k are treated as 0.
- If `q` is real, the matrix is real symmetric.
- If `q` is complex (e.g. purely imaginary), the matrix is complex symmetric.
"""
function odd_matrix(q, N::Integer, alphas::AbstractVector)
    N ≥ 3 || throw(ArgumentError("N must be ≥ 3 (odd matrix is (N-1)×(N-1))"))

    R = N - 1
    need = 2*R

    Tr = _base_real_type(q, alphas)
    MT = _matrix_eltype(q, Tr)

    qT = MT(q)
    alphasT = _pad_alphas(alphas, need, MT)

    α(k::Integer) = (1 ≤ k ≤ length(alphasT)) ? alphasT[k] : zero(MT)

    B = zeros(MT, R, R)

    @inbounds for r in 1:R
        B[r, r] = MT(4) * MT(r)^2 - qT * α(2r)

        for c in (r+1):R
            v = qT * (α(abs(r - c)) - α(r + c))
            B[r, c] = v
            B[c, r] = v
        end
    end

    return q isa Real ? Symmetric(B) : B
end


# --------------------------
# Symmetry types + pipeline
# --------------------------

abstract type Symmetry end
struct Even <: Symmetry end
struct Odd  <: Symmetry end

export Even, Odd

# round a Real/Complex to `digits` decimals for sorting
_roundkey(z, digits::Int) = z isa Complex ?
    complex(round(real(z); digits=digits), round(imag(z); digits=digits)) :
    round(z; digits=digits)

# Sorting rule: increasing real part, then imag part (negative imag first), then |λ|
# _sortperm(vals) = sortperm(vals; by = λ -> (real(λ), imag(λ), abs(λ)))
_sortperm(vals; digits::Int=1) =
    sortperm(vals; by = λ -> (round(real(λ); digits=digits),
                              round(imag(λ); digits=digits),
                              abs(λ)))

# Build dense matrix for eigensolve (N is small; dense is fine)
_build_dense(::Type{Even}, q, N::Integer, alphas) = Matrix(even_matrix(q, N, alphas))
_build_dense(::Type{Odd}, q, N::Integer, alphas) = Matrix(odd_matrix(q, N, alphas))

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
    sqrt2 = sqrt(real(T(2)))   # correct precision/type (e.g., BigFloat)
    V[1, :] ./= sqrt2
    _anorm_bilinear_cols!(V, T(2))
    return V
end

function _mathieu_normalize!(::Type{Odd}, V::AbstractMatrix)
    T = eltype(V)
    _anorm_bilinear_cols!(V, T(1))
    return V
end

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
    even_eigen(q, N, alphas; prec_bits=nothing, normalize=true)

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
    odd_eigen(q, N, alphas; prec_bits=nothing, normalize=true)

Eigenpairs for the ODD matrix, sorted by (Re, Im, |λ|).

If `normalize=true` (default), applies bilinear normalization (no conjugation) with fac=1.
"""
function odd_eigen(q, N::Integer,alphas::AbstractVector;
                   prec_bits::Union{Nothing,Int}=nothing)
    _with_precision(() -> _eigen_sorted(Odd, q, N, alphas), prec_bits)
end


"""
Compute all even eigenfunctions ϕ_{2n}(y).

Inputs:
- A : coefficient matrix (R × N) of even eigenvectors
      A[r+1, n+1] = A_{2r}^{(2n)}
- y : vector of points in [0, π]

Returns:
- Φ : matrix (length(y) × N)
      Φ[:,1] = ϕ₀(y)
      Φ[:,2] = ϕ₂(y)
      Φ[:,3] = ϕ₄(y)
      ...
"""
function even_eigenfunctions(A::AbstractMatrix, y::AbstractVector)

    R  = size(A, 1)
    # r = 0,1,2,...,R-1
    r = 0:R-1

    # Basis matrix: cos(2 r y)
    B = cos.(2 .* (y .* r'))   # size Ny × R

    # Copy coefficients
    C = copy(A)

    # Evaluate all eigenfunctions
    Φ = B * C

    return Φ
end


"""
Compute all odd eigenfunctions ϕ_{2n+2}(y).

Inputs:
- B : coefficient matrix (R × N) of odd eigenvectors
      B[r+1, n+1] = B_{2r+2}^{(2n+2)}
- y : vector of points in [0, π]

Returns:
- Φ : matrix (length(y) × N)
      Φ[:,1] = ϕ₂(y)
      Φ[:,2] = ϕ₄(y)
      Φ[:,3] = ϕ6(y)
      ...
"""
function odd_eigenfunctions(A::AbstractMatrix, y::AbstractVector)

    R  = size(A, 1)

    # r = 1,1,2,...,R
    r = 1:R

    # Basis matrix: sin(2 r y)
    B = sin.(2 .* (y .* r'))   # size Ny × R

    # Copy coefficients
    C = copy(A)

    # Evaluate all eigenfunctions
    Φ = B * C

    return Φ
end


end # module HillFunctions