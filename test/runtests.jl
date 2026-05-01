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
        alphas_i = [1, 2, 3]

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
        nrm = bilinear_norm(@view vecs[:, j]; symmetry = :even)
        @test isapprox(nrm, one(nrm); atol = 1e-8, rtol = 1e-8)
    end
end

@testset "Mathieu bilinear normalization (odd)" begin
    N = 8
    q = 1im
    alphas = [0, 1, zeros(N-1)...]

    vals, vecs = odd_eigen(q, N, alphas)

    for j in axes(vecs, 2)
        nrm = bilinear_norm(@view vecs[:, j]; symmetry = :odd)
        @test isapprox(nrm, one(nrm); atol = 1e-8, rtol = 1e-8)
    end
end

@testset "BigFloat normalization" begin
    N = 6
    q = Complex{BigFloat}(0, 1)
    alphas = BigFloat[0, 1, zeros(N-1)...]

    vals, vecs = even_eigen(q, N, alphas; prec_bits = 256);

    for j in axes(vecs, 2)
        nrm = bilinear_norm(@view vecs[:, j]; symmetry = :even)
        @test isapprox(nrm, one(nrm); atol = big"1e-40")
    end
end


@testset "even row 1 sum of squares = 1/2" begin
    N = 6
    q = 1im
    alphas = [0, 1, zeros(N-1)...]
    _, vecs = even_eigen(q, N, alphas);
    # test only first rwo
    nrm = sum(x -> x * x, @view vecs[1, :])
    @test isapprox(nrm, 0.5 + 0.0im; atol = 1e-8, rtol = 1e-8)
end



@testset "even rows 2:end sum of squares = 1" begin
    N = 6
    q = 1im
    alphas = [0, 1, zeros(N-1)...]
    _, vecs = even_eigen(q, N, alphas);

    target = one(eltype(vecs)) + 0im

    for r = 2:size(vecs, 1)
        nrm = sum(x -> x * x, @view vecs[r, :])
        @test isapprox(nrm, target; atol = 1e-8, rtol = 1e-8)
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
        @test isapprox(nrm, target; atol = 1e-8, rtol = 1e-8)
    end
end

@testset "odd matrix q=1i" begin
    alphas = [1, 2, 3, 4, 5, 6, 7, 8]
    q = 1im
    N = 5
    B = odd_matrix(q, N, alphas)

    B_expected = ComplexF64[
        4-2im -2im -2im -2im
        -2im 16-4im -4im -4im
        -2im -4im 36-6im -6im
        -2im -4im -6im 64-8im
    ]

    @test size(B) == (4, 4)
    @test isapprox(B, B_expected; atol = 1e-12, rtol = 1e-12)

end

@testset "even matrix q=1i" begin
    alphas = [1, 2, 3, 4, 5, 6]
    q = 1im
    N = 4
    A = even_matrix(q, N, alphas)

    A_expected = ComplexF64[
        0.0+0.0im 0.0+1.41421im 0.0+2.82843im 0.0+4.24264im
        0.0+1.41421im 4.0+2.0im 0.0+4.0im 0.0+6.0im
        0.0+2.82843im 0.0+4.0im 16.0+4.0im 0.0+6.0im
        0.0+4.24264im 0.0+6.0im 0.0+6.0im 36.0+6.0im
    ]

    @test size(A) == (4, 4)
    @test isapprox(A, A_expected; atol = 1e-4, rtol = 1e-4)

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
    alphas = BigFloat[1, 2, 3, 4, 5, 6]

    A = even_matrix(q, N, alphas)

    @test eltype(A) == Complex{BigFloat}
end


@testset "odd_matrix preserves Complex{BigFloat} element type with q and alphas" begin
    N = 5
    q = Complex{BigFloat}(0, 1)
    alphas = BigFloat[1, 2, 3, 4, 5, 6, 7, 8]

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
                @test isapprox(prod, zero(prod); atol = 1e-4, rtol = 1e-4)
            else
                if j==1
                    @test isapprox(prod, 0.5*one(prod); atol = 1e-4, rtol = 1e-4)
                else
                    @test isapprox(prod, one(prod); atol = 1e-4, rtol = 1e-4)
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
                @test isapprox(prod, zero(prod); atol = 1e-4, rtol = 1e-4)
            else
                @test isapprox(prod, one(prod); atol = 1e-4, rtol = 1e-4)
            end
        end
    end

end


@testset "Even eigenfunction mappings with normal modes" begin
    N = 25
    q = 100.0im
    alphas = [-0.5]
    y = collect(range(0, π; length = 100));

    _, V = even_eigen(q, N, alphas)
    Phis_e = even_eigenfunctions(V, y);

    R = size(V, 1)
    r = 0:(R-1)
    # Basis matrix: cos(2 r y)
    B = cos.(2 .* (y .* r'))   # size Ny × R

    for i in R
        if i==1
            fac=2
        else
            fac=1
        end
        prod = Phis_e * (fac .* vec(V[i, :]))
        @test isapprox(prod, B[:, i]; atol = 1e-4, rtol = 1e-4)
    end

end


@testset "Odd eigenfunction mappings with normal modes" begin
    N = 25
    q = 100.0im
    alphas = [-0.5]
    y = collect(range(0, π; length = 100));

    _, V = odd_eigen(q, N, alphas)
    Phis_o = odd_eigenfunctions(V, y);

    R = size(V, 1)
    r = 1:R
    # Basis matrix: cos(2 r y)
    B = sin.(2 .* (y .* r'))   # size Ny × R

    for i in R
        prod = Phis_o * vec(V[i, :])
        @test isapprox(prod, B[:, i]; atol = 1e-4, rtol = 1e-4)
    end

end

@testset "HillFunctions sweep writer" begin

    try
        @eval using JLD2
        include("test_io_jld2.jl")
    catch err
        @info "Skipping JLD2 I/O tests (JLD2 not available in test env)." err
    end
end


@testset "test dense sweep of eigenpairs for range of q-values" begin
    qs = im .* range(0, 10)
    N = 40
    alphas = [-0.5]

    qvec, vals, vecs = collect_sweep_eigen_dense(Even, qs, N, alphas; prec_bits = 512)

    @test size(qvec) == (11,)
    @test size(vals) == (40, 11)
    @test size(vecs) == (40, 40, 11)
end

@testset "test sweep of eigenpairs for range of q-values" begin
    qs = im .* range(0, 10)
    N = 40
    alphas = [-0.5]

    qvec, vals, vecs = collect_sweep_eigen(Even, qs, N, alphas; prec_bits = 512)

    @test size(qvec) == (11,)
    @test size(vals) == (11,)
    @test size(vecs) == (11,)

    @test size(vals[1]) == (10,)
    @test size(vecs[1]) == (10, 10)

    @test size(vals[4]) == (10,)
    @test size(vecs[4]) == (10, 10)

    @test size(vals[10]) == (10,)
    @test size(vecs[10]) == (10, 10)

end


@testset "adaptive eigen matches ragged sweep entries" begin
    alphas = Float64[-0.5]
    N = 12
    Nmax = 5

    qsets = (
        ComplexF64[0.0im, 0.25im, 1.0im],
        Complex{BigFloat}[
            Complex(big"0.0", big"0.0"),
            Complex(big"0.0", big"0.25"),
            Complex(big"0.0", big"1.0"),
        ],
    )

    for qs in qsets
        for S in (HillFunctions.Even, HillFunctions.Odd)
            qvec, vals, vecs = collect_sweep_eigen(S, qs, N, alphas; Nmax)

            @test qvec == qs

            for iq in eachindex(qs)
                λ, V = adaptive_eigen(S, qs[iq], N, alphas; Nmax)

                @test vals[iq] ≈ λ
                @test vecs[iq] ≈ V
                @test size(vecs[iq], 1) == Nmax
                @test size(vecs[iq], 2) == length(vals[iq])
            end
        end
    end
end


@testset "Test that eigenpairs return using sweep collectors have correct types" begin
    alphas = Float64[-0.5]
    N = 6

    # Helper to test ragged collector output types
    function check_ragged(::Type{S}, qs, T_expected) where {S<:HillFunctions.Symmetry}
        qvec, vals, vecs = collect_sweep_eigen(S, qs, N, alphas; prec_bits = 128)

        @test eltype(qvec) === eltype(qs)
        @test length(vals) == length(qs)
        @test length(vecs) == length(qs)

        # Each entry is a vector/matrix with element type T_expected
        @test eltype(vals[1]) === T_expected
        @test eltype(vecs[1]) === T_expected

        # And qvec entries match qs element type
        @test qvec[1] isa eltype(qs)
    end

    # Helper to test dense collector output types
    function check_dense(::Type{S}, qs, T_expected) where {S<:HillFunctions.Symmetry}
        qvec, vals, vecs = collect_sweep_eigen_dense(S, qs, N, alphas; prec_bits = 128)

        @test eltype(qvec) === eltype(qs)
        @test eltype(vals) === T_expected
        @test eltype(vecs) === T_expected

        @test size(vals) == (N, length(qs))
        @test size(vecs) == (N, N, length(qs))
    end

    # ---- Case 1: q :: Float64 -> outputs Float64
    qs_real64 = [0.1, 0.2]
    for S in (HillFunctions.Even, HillFunctions.Odd)
        check_ragged(S, qs_real64, Float64)
        check_dense(S, qs_real64, Float64)
    end

    # ---- Case 2: q :: ComplexF64 -> outputs ComplexF64
    qs_c64 = ComplexF64[0.1im, 0.2im]
    for S in (HillFunctions.Even, HillFunctions.Odd)
        check_ragged(S, qs_c64, ComplexF64)
        check_dense(S, qs_c64, ComplexF64)
    end

    # ---- Case 3: q :: BigFloat -> outputs BigFloat
    qs_realbig = BigFloat[big"0.1", big"0.2"]
    for S in (HillFunctions.Even, HillFunctions.Odd)
        check_ragged(S, qs_realbig, BigFloat)
        check_dense(S, qs_realbig, BigFloat)
    end

    # ---- Case 4: q :: Complex{BigFloat} -> outputs Complex{BigFloat}
    qs_cbig = Complex{BigFloat}[Complex(big"0.0", big"0.1"), Complex(big"0.0", big"0.2")]
    for S in (HillFunctions.Even, HillFunctions.Odd)
        check_ragged(S, qs_cbig, Complex{BigFloat})
        check_dense(S, qs_cbig, Complex{BigFloat})
    end
end

@testset "check even sweeped ragged eigenfunctions map to normal modes" begin
    qs = ComplexF64.(range(0, 100, 11))
    Ny = 100
    y = range(0, 2*π; length = Ny)
    N = 100
    alphas = [-0.5]
    qvec, vals, Ak = collect_sweep_eigen(Even, qs, N, alphas)
    Phi_e = sweep_even_eigenfunctions(Ak, y)

    @test size(Phi_e[1]) == (Ny, 10)
    @test size(Phi_e[end]) == (Ny, 25)

    for n in range(1, length(Phi_e))
        V = Ak[n]
        Phi = Phi_e[n]
        R = size(V, 1)
        r = 0:(R-1)
        # Basis matrix: cos(2 r y)
        B = cos.(2 .* (y .* r'))   # size Ny × R

        for i in R
            if i==1
                fac=2
            else
                fac=1
            end
            prod = Phi * (fac .* vec(V[i, :]))
            @test isapprox(prod, B[:, i]; atol = 1e-4, rtol = 1e-4)
        end
    end

end

@testset "check odd sweeped ragged eigenfunctions map to normal modes" begin
    qs = ComplexF64.(range(0, 100, 11))
    Ny = 100
    y = range(0, 2*π; length = Ny)
    N = 100
    alphas = [-0.5]
    qvec, vals, Bk = collect_sweep_eigen(Odd, qs, N, alphas)
    Phi_o = sweep_odd_eigenfunctions(Bk, y)

    @test size(Phi_o[1]) == (Ny, 10)
    @test size(Phi_o[end]) == (Ny, 25)

    for n in range(1, length(Phi_o))
        V = Bk[n]
        Phi = Phi_o[n]
        R = size(V, 1)
        r = 1:R
        # Basis matrix: cos(2 r y)
        B = sin.(2 .* (y .* r'))   # size Ny × R

        for i in R
            prod = Phi * vec(V[i, :])
            @test isapprox(prod, B[:, i]; atol = 1e-4, rtol = 1e-4)
        end
    end
end

@testset "dense sweep q-last slices match direct eigenpairs" begin
    qs = ComplexF64[0.0im, 0.25im, 0.5im]
    N = 8
    alphas = Float64[-0.5]
    prec_bits = 128

    for S in (HillFunctions.Even, HillFunctions.Odd)
        qvec, vals, vecs = collect_sweep_eigen_dense(S, qs, N, alphas; prec_bits)

        Nsolve = S === HillFunctions.Odd ? N + 1 : N

        for iq in eachindex(qs)
            λ_direct, V_direct = HillFunctions._with_precision(
                () -> HillFunctions._eigen_sorted(S, qs[iq], Nsolve, alphas),
                prec_bits,
            )

            @test qvec[iq] == qs[iq]
            @test vals[:, iq] ≈ eltype(vals).(λ_direct)
            @test vecs[:, :, iq] ≈ eltype(vecs).(V_direct)
        end
    end
end
