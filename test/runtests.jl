using Junta, Test

@testset "junta" begin
    testfn = (x::BitVector) -> reduce(xor, x[[3,4]])
    dim = 4
    ϵ = 0.01
    error_prob = 1e-1

    t_junta = check_for_juntas_adaptive_simple(testfn, 3, ϵ, dim, error_prob)
    @test t_junta[1] == true

    te_junta = check_for_juntas_adaptive_simple(testfn, 2, ϵ, dim, error_prob)
    @test te_junta[1] == true

    f_junta = check_for_juntas_adaptive_simple(testfn, 1, ϵ, dim, error_prob)
    @test f_junta[1] == false
end

@testset "auto junta size" begin
    testindices = [1,4,5,7,8]
    testfn = (x::BitVector) -> reduce(xor, x[testindices])
    dim = 20
    ϵ = 1e-3
    error_prob = 1e-5

    (k, foundindices) = junta_size_adaptive_simple(testfn, ϵ, dim, error_prob)

    @test k == 5
    @test foundindices == testindices
end

@testset "hard function" begin
    testindices = [1,4,5,7,8]
    function testfn(x::BitVector)
        if x[9] == 1 && x[6] == 1 && x[3] == 1
            return reduce(xor, x[testindices])
        else
            return 1
        end
    end
    dim = 20
    ϵ = 1e-3
    error_prob = 1e-5

    (k, foundindices) = junta_size_adaptive_simple(testfn, ϵ, dim, error_prob)

    @test k == 8
    @test foundindices == [1,3,4,5,6,7,8,9]
end
