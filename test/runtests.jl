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

@testset "odd matrix q=1i" begin
    alphas = [1, 2, 3, 4, 5, 6, 7, 8]
    q = 1im
    N = 5
    B = odd_matrix(q, N, alphas)

    B_expected = ComplexF64[
        4-2im  -2im    -2im    -2im
        -2im   16-4im  -4im    -4im
        -2im   -4im    36-6im  -6im
        -2im   -4im    -6im    64-8im
    ]

    @test size(B) == (4,4)
    @test isapprox(B, B_expected; atol=1e-12, rtol=1e-12)

end

@testset "even matrix q=1i" begin
    alphas = [1, 2, 3, 4, 5, 6]
    q = 1im
    N = 4
    A = even_matrix(q, N, alphas)

    A_expected = ComplexF64[
        0.0+0.0im      0.0+1.41421im   0.0+2.82843im   0.0+4.24264im
        0.0+1.41421im  4.0+2.0im       0.0+4.0im       0.0+6.0im
        0.0+2.82843im  0.0+4.0im      16.0+4.0im       0.0+6.0im
        0.0+4.24264im  0.0+6.0im       0.0+6.0im      36.0+6.0im
    ]

    @test size(A) == (4,4)
    @test isapprox(A, A_expected; atol=1e-4, rtol=1e-4)

end

@testset "even_matrix preserves Complex{BigFloat} element type with q" begin
    N = 4
    q = Complex{BigFloat}(0, 1)
    alphas = [1, 2, 3, 4, 5, 6, 7, 8]

    A = even_matrix(q, N, alphas)

    @test eltype(A) == Complex{BigFloat}
end

@testset "even_matrix preserves Complex{BigFloat} element type with q and alphas" begin
    N = 4
    q = Complex{BigFloat}(0, 1)
    alphas = BigFloat[1,2,3,4,5,6]

    A = even_matrix(q, N, alphas)

    @test eltype(A) == Complex{BigFloat}
end


@testset "odd_matrix preserves Complex{BigFloat} element type with q and alphas" begin
    N = 5
    q = Complex{BigFloat}(0, 1)
    alphas = BigFloat[1,2,3,4,5,6,7,8]

    B = odd_matrix(q, N, alphas)

    @test eltype(B) == Complex{BigFloat}
end

@testset "odd_matrix preserves Complex{BigFloat} element type" begin
    N = 5
    q = Complex{BigFloat}(0, 1)
    alphas = [1, 2, 3, 4, 5, 6, 7, 8]

    B = odd_matrix(q, N, alphas)

    @test eltype(B) == Complex{BigFloat}
end
