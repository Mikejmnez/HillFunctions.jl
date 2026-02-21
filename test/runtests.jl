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

    vals, vecs = even_eigen(N, q, alphas; prec_bits=256);

    for j in axes(vecs, 2)
        nrm = bilinear_norm(@view vecs[:, j]; symmetry=:even)
        @test isapprox(nrm, one(nrm); atol=big"1e-40")
    end
end


@testset "even row 1 sum of squares = 1/2" begin
    N = 6
    q = 1im
    alphas = [0, 1, zeros(N-1)...]
    _, vecs = even_eigen(N, q, alphas);
    # test only first rwo
    nrm = sum(x -> x * x, @view vecs[1, :])
    @test isapprox(nrm, 0.5 + 0.0im; atol=1e-8, rtol=1e-8)
end



@testset "even rows 2:end sum of squares = 1" begin
    N = 6
    q = 1im
    alphas = [0, 1, zeros(N-1)...]
    _, vecs = even_eigen(N, q, alphas);

    target = one(eltype(vecs)) + 0im

    for r in 2:size(vecs, 1)
        nrm = sum(x -> x * x, @view vecs[r, :])
        @test isapprox(nrm, target; atol=1e-8, rtol=1e-8)
    end
end


@testset "odd rows sum of squares = 1" begin
    N = 6
    q = 1im
    alphas = [0, 1, zeros(N-1)...]
    _, vecs = odd_eigen(N, q, alphas);

    target = one(eltype(vecs)) + 0im

    for r in axes(vecs, 2)
        nrm = sum(x -> x * x, @view vecs[r, :])
        @test isapprox(nrm, target; atol=1e-8, rtol=1e-8)
    end
end