using AcceleratedTrafficAssignement
using Test
using JET, JuliaFormatter, Aqua
using Graphs, Random, SimpleWeightedGraphs, FasterShortestPaths, SparseArrays
#using SuiteSparseMatrixCollection, HarwellRutherfordBoeing
#using BenchmarkTools, SparseArrays
using CUDA #, GPUArrays, GPUGraphs

Random.seed!(42)

function get_random_grid(n)
    g = grid((n, n))
    g_dir = SimpleDiGraph(nv(g))
    weights_dir = Dict{Tuple{Int,Int},Float64}()
    for e in edges(g)
        u = src(e)
        v = dst(e)
        if rand() > 0.45
            weight = rand() + 0.1
            weights_dir[(u, v)] = weight
            add_edge!(g_dir, u, v)
        end
        if rand() > 0.45
            weight = rand() + 0.1
            weights_dir[(v, u)] = weight
            add_edge!(g_dir, v, u)
        end
    end
    return g_dir, weights_dir
end


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
    g, w = get_random_grid(20)
    CH = compute_CH(g, w)

    include("ContractionsHierarchies.jl")
    @testset "Contraction Hierarchies" begin
        test_graph_contractions(CH)
    end
    include("Phast.jl")
    @testset "PHAST Queries" begin
        test_phast_queries(CH)
    end

end
