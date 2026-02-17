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

# Internal sorting rule
_sortperm(vals) = sortperm(vals; by = λ -> (real(λ), imag(λ), abs(λ)))

"""
    even_eigvals(N, q, alphas; prec_bits=nothing)

Eigenvalues of the EVEN matrix, sorted by (Re, Im, |λ|).
"""
function even_eigvals(N::Integer, q, alphas::AbstractVector; prec_bits=nothing)
    if prec_bits !== nothing
        setprecision(BigFloat, prec_bits)
    end

    A = Matrix(even_matrix(N, q, alphas))
    vals = GenericSchur.eigen(A).values

    idx = _sortperm(vals)
    return vals[idx]
end


"""
    even_eigen(N, q, alphas; prec_bits=nothing)

Eigenpairs (values, vectors) of the EVEN matrix, sorted by (Re, Im, |λ|).
Vectors are right eigenvectors (columns).
"""
function even_eigen(N::Integer, q, alphas::AbstractVector; prec_bits=nothing)
    if prec_bits !== nothing
        setprecision(BigFloat, prec_bits)
    end

    A = Matrix(even_matrix(N, q, alphas))
    E = GenericSchur.eigen(A)

    vals = E.values
    vecs = E.vectors

    idx = _sortperm(vals)

    return vals[idx], vecs[:, idx]
end

"""
    odd_eigvals(N, q, alphas; prec_bits=nothing)

Eigenvalues of the ODD matrix, sorted by (Re, Im, |λ|).
"""
function odd_eigvals(N::Integer, q, alphas::AbstractVector; prec_bits::Union{Nothing,Int}=nothing)
    if prec_bits !== nothing
        setprecision(BigFloat, prec_bits)
    end
    B = Matrix(odd_matrix(N, q, alphas))
    vals = GenericSchur.eigen(B).values
    idx = _sortperm(vals)
    return vals[idx]
end

"""
    odd_eigen(N, q, alphas; prec_bits=nothing)

Eigenpairs (values, vectors) of the ODD matrix, sorted by (Re, Im, |λ|).
Vectors are right eigenvectors (columns).
"""
function odd_eigen(N::Integer, q, alphas::AbstractVector; prec_bits::Union{Nothing,Int}=nothing)
    if prec_bits !== nothing
        setprecision(BigFloat, prec_bits)
    end
    B = Matrix(odd_matrix(N, q, alphas))
    E = GenericSchur.eigen(B)
    vals, vecs = E.values, E.vectors
    idx = _sortperm(vals)
    return vals[idx], vecs[:, idx]

end

end # module HillFunctions