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
