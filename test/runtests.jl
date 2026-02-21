using HillFunctions
using Test

# Bilinear norm (no conjugation)
function bilinear_norm(v; symmetry::Symbol)
    fac = symmetry === :even ? 2 : 1
    return fac * (v[1] * v[1]) + sum(x -> x * x, @view v[2:end])
end

@testset "Mathieu bilinear normalization (even)" begin
    N = 8
    q = 1im
    alphas = [0, 1, zeros(N-1)...]

    vals, vecs = even_eigen(N, q, alphas)

    for j in axes(vecs, 2)
        nrm = bilinear_norm(@view vecs[:, j]; symmetry=:even)
        @test isapprox(nrm, one(nrm); atol=1e-8, rtol=1e-8)
    end
end

@testset "Mathieu bilinear normalization (odd)" begin
    N = 8
    q = 1im
    alphas = [0, 1, zeros(N-1)...]

    vals, vecs = odd_eigen(N, q, alphas)

    for j in axes(vecs, 2)
        nrm = bilinear_norm(@view vecs[:, j]; symmetry=:odd)
        @test isapprox(nrm, one(nrm); atol=1e-8, rtol=1e-8)
    end
end

@testset "BigFloat normalization" begin
    N = 6
    q = Complex{BigFloat}(0, 1)
    alphas = BigFloat[0, 1, zeros(N-1)...]

    vals, vecs = even_eigen(N, q, alphas; prec_bits=256)

    for j in axes(vecs, 2)
        nrm = bilinear_norm(@view vecs[:, j]; symmetry=:even)
        @test isapprox(nrm, one(nrm); atol=big"1e-40")
    end
end