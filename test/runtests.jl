using HillFunctions
using Test
using LinearAlgebra

# Bilinear norm (no conjugation)
function bilinear_norm(v; symmetry::Symbol)
    fac = symmetry === :even ? 2 : 1
    return fac * (v[1] * v[1]) + sum(x -> x * x, @view v[2:end])
end

# Bilinear inner product (NO conjugation)
bilinear_inner(v1, v2) = sum(v1 .* v2)


@testset "Real-q support" begin

    @testset "_base_real_type" begin
        # alphas always real (as you noted)
        alphas_f64 = [0.5, 1.25]
        alphas_i   = [1, 2, 3]

        # Real q + Float64 alphas -> Float64
        @test HillFunctions._base_real_type(2.0, alphas_f64) === Float64

        # Real integer q + integer alphas -> Float64 (via _realfloat_type)
        @test HillFunctions._base_real_type(2, alphas_i) === Float64

        # BigFloat q + Float64 alphas -> BigFloat (promote)
        @test HillFunctions._base_real_type(big"2.0", alphas_f64) === BigFloat

        # Purely imaginary (complex) q should still return the *real* base type
        @test HillFunctions._base_real_type(2.0im, alphas_f64) === Float64
        @test HillFunctions._base_real_type(big"2.0" * im, alphas_f64) === BigFloat
    end


    @testset "even_matrix with real q" begin
        N = 6
        alphas = [0.5]

        q = 2.0
        A = even_matrix(q, N, alphas)

        # 1) purely real element type
        @test eltype(A) <: Real

        # 2) symmetric numerically
        @test issymmetric(A)

        # 3) diagonal is real (redundant but explicit)
        @test all(isreal, diag(A))

        # 4) sanity: size is N×N
        @test size(A) == (N, N)

        # 5) BigFloat preserves BigFloat when q is BigFloat
        qB = big"2.0"
        AB = even_matrix(qB, N, alphas)
        @test eltype(AB) === BigFloat
        @test issymmetric(AB)
    end


    @testset "odd_matrix with real q" begin
        N = 7
        alphas = [0.5]

        q = 3.0
        B = odd_matrix(q, N, alphas)

        R = N - 1

        # 1) purely real element type
        @test eltype(B) <: Real

        # 2) symmetric numerically
        @test issymmetric(B)

        # 3) size matches (N-1)×(N-1)
        @test size(B) == (R, R)

        # 4) BigFloat preserves BigFloat when q is BigFloat
        qB = big"3.0"
        BB = odd_matrix(qB, N, alphas)
        @test eltype(BB) === BigFloat
        @test issymmetric(BB)
    end

end


@testset "Mathieu bilinear normalization (even)" begin
    N = 8
    q = 1im
    alphas = [0, 1, zeros(N-1)...]

    vals, vecs = even_eigen(q, N, alphas)

    for j in axes(vecs, 2)
        nrm = bilinear_norm(@view vecs[:, j]; symmetry=:even)
        @test isapprox(nrm, one(nrm); atol=1e-8, rtol=1e-8)
    end
end

@testset "Mathieu bilinear normalization (odd)" begin
    N = 8
    q = 1im
    alphas = [0, 1, zeros(N-1)...]

    vals, vecs = odd_eigen(q, N, alphas)

    for j in axes(vecs, 2)
        nrm = bilinear_norm(@view vecs[:, j]; symmetry=:odd)
        @test isapprox(nrm, one(nrm); atol=1e-8, rtol=1e-8)
    end
end

@testset "BigFloat normalization" begin
    N = 6
    q = Complex{BigFloat}(0, 1)
    alphas = BigFloat[0, 1, zeros(N-1)...]

    vals, vecs = even_eigen(q, N, alphas; prec_bits=256);

    for j in axes(vecs, 2)
        nrm = bilinear_norm(@view vecs[:, j]; symmetry=:even)
        @test isapprox(nrm, one(nrm); atol=big"1e-40")
    end
end


@testset "even row 1 sum of squares = 1/2" begin
    N = 6
    q = 1im
    alphas = [0, 1, zeros(N-1)...]
    _, vecs = even_eigen(q, N, alphas);
    # test only first rwo
    nrm = sum(x -> x * x, @view vecs[1, :])
    @test isapprox(nrm, 0.5 + 0.0im; atol=1e-8, rtol=1e-8)
end



@testset "even rows 2:end sum of squares = 1" begin
    N = 6
    q = 1im
    alphas = [0, 1, zeros(N-1)...]
    _, vecs = even_eigen(q, N, alphas);

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
    _, vecs = odd_eigen(q, N, alphas);

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

@testset "Even eigenvectors: bilinear column orthogonality" begin
    N = 35
    q = 100.0im
    alphas = [0.5]

    _, V = even_eigen(q, N, alphas)

    for j in axes(V, 1)
        for i in axes(V, 2)
            prod=bilinear_inner(V[i, :], V[j, :])
            if j!=i
                @test isapprox(prod, zero(prod); atol=1e-4, rtol=1e-4)
            else
                if j==1
                    @test isapprox(prod, 0.5*one(prod); atol=1e-4, rtol=1e-4)
                else
                    @test isapprox(prod, one(prod); atol=1e-4, rtol=1e-4)
                end
            end
        end
    end
    
end

@testset "Odd eigenvectors: bilinear column orthogonality" begin
    N = 25
    q = 100.0im
    alphas = [0.5]

    _, V = odd_eigen(q, N, alphas)

    for j in axes(V, 1)
        for i in axes(V, 2)
            prod=bilinear_inner(V[i, :], V[j, :])
            if j!=i
                @test isapprox(prod, zero(prod); atol=1e-4, rtol=1e-4)
            else
                @test isapprox(prod, one(prod); atol=1e-4, rtol=1e-4)
            end
        end
    end

end


@testset "Even eigenfunction mappings with normal modes" begin
    N = 25
    q = 100.0im
    alphas = [-0.5]
    y = collect(range(0, π; length=100));

    _, V = even_eigen(q, N, alphas)
    Phis_e = even_eigenfunctions(V, y);

    R  = size(V, 1)
    r = 0:R-1
    # Basis matrix: cos(2 r y)
    B = cos.(2 .* (y .* r'))   # size Ny × R

    for i in R
        if i==1
            fac=2
        else
            fac=1
        end
        prod = Phis_e * (fac .* vec(V[i, :]))
        @test isapprox(prod, B[:,i]; atol=1e-4, rtol=1e-4)
    end

end


@testset "Odd eigenfunction mappings with normal modes" begin
    N = 25
    q = 100.0im
    alphas = [-0.5]
    y = collect(range(0, π; length=100));

    _, V = odd_eigen(q, N, alphas)
    Phis_o = odd_eigenfunctions(V, y);

    R  = size(V, 1)
    r = 1:R
    # Basis matrix: cos(2 r y)
    B = sin.(2 .* (y .* r'))   # size Ny × R

    for i in R
        prod = Phis_o * vec(V[i, :])
        @test isapprox(prod, B[:,i]; atol=1e-4, rtol=1e-4)
    end

end