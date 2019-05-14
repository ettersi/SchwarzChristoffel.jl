using SchwarzChristoffel
using Test

using StaticArrays

@testset "integral" begin
    f = SchwarzChristoffelDerivate(SVector(0),SVector(0)); @test integral(f,1,1) ≈ 1
    f = SchwarzChristoffelDerivate(SVector(0),SVector(0)); @test integral(f,1,2) ≈ 2
    f = SchwarzChristoffelDerivate(SVector(1),SVector(0)); @test integral(f,1,2) ≈ 1

    f = SchwarzChristoffelDerivate(SVector(0),SVector(-1)); @test integral(f,1,1) ≈ 0.5
    f = SchwarzChristoffelDerivate(SVector(0),SVector(-1)); @test integral(f,1,2) ≈ 2
    f = SchwarzChristoffelDerivate(SVector(1),SVector(-1)); @test integral(f,1,2) ≈ 0.5

    f = SchwarzChristoffelDerivate(SVector(0),SVector(0.5)); @test integral(f,1,1) ≈ 2
    f = SchwarzChristoffelDerivate(SVector(0),SVector(0.5)); @test integral(f,1,2) ≈ 2*sqrt(2)
    f = SchwarzChristoffelDerivate(SVector(1),SVector(0.5)); @test integral(f,1,2) ≈ 2

    f = SchwarzChristoffelDerivate(SVector(-1,0),SVector(-1,-1)); @test integral(f,2,1) ≈ 5/6
    f = SchwarzChristoffelDerivate(SVector(-1,0),SVector(-1,-1)); @test integral(f,2,2) ≈ 14/3

    f = SchwarzChristoffelDerivate(SVector(-1e-8,0),SVector(0.5,0)); @test integral(f,2,1) ≈ (sqrt(1e8+1)-1)/5000

    f = SchwarzChristoffelDerivate(SVector(0,1),SVector(0)); @test segment(f,1) ≈ 1
end
