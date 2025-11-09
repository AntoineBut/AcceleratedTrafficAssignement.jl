using AcceleratedTrafficAssignement
using Test
using JET, JuliaFormatter, Aqua
using AcceleratedTrafficAssignement

@testset "AcceleratedTrafficAssignement.jl" begin
    @testset "Code Quality" begin
        @testset "Aqua" begin
            Aqua.test_all(AcceleratedTrafficAssignement; ambiguities = false)
        end
        @testset "JET" begin
            JET.test_package(AcceleratedTrafficAssignement; target_defined_modules = true)
        end
        @testset "JuliaFormatter" begin
            @test JuliaFormatter.format(AcceleratedTrafficAssignement; overwrite = false)
        end

    end

end
