using Test
using HillFunctions
using JLD2  # activates extension

@testset "JLD2 sweep writer" begin
    qs = [0.1im, 0.2im]
    N = 6
    alphas = [-0.5]

    path = joinpath(mktempdir(), "sweep.jld2")

    w = open_jld2_writer(path; nsteps=length(qs), meta=(N=N, alphas=alphas, symmetry=:even))

    sweep_eigen(Even, qs, N, alphas; prec_bits=128, writer=w)

    q1, λ1, V1 = load_step_jld2(path, 1)
    @test q1 == qs[1]
    @test length(λ1) == N
    @test size(V1) == (N, N)
end