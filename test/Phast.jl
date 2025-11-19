# Tests for the PHAST querry algorithm

function test_phast_queries(
    CH::CHGraph,
    gpu_CH::gpu_CHGraph,
    ::Type{T} = Float64,
) where {T<:Real}
    g = CH.g
    g_w = digraph_to_weightedgraph(g, CH.weights)
    sources = rand(1:nv(g), 5)
    distances_dijkstra = fill(typemax(T), nv(g), length(sources))
    for (j, source) in enumerate(sources)
        distances_dijkstra[:, j] = dijkstra_shortest_paths(g_w, source).dists
    end
    distances_phast_cpu = shortest_path_CH(CH, sources).distances
    @test isapprox(distances_phast_cpu, distances_dijkstra)
    res_gpu = zeros(T, nv(g), length(sources))
    distances_phast_gpu = shortest_path_CH(gpu_CH, sources).device_distances
    copyto!(res_gpu, distances_phast_gpu)
    diff_matrix = res_gpu .== distances_dijkstra
    println(diff_matrix)
    println(sum(diff_matrix, dims=1))
    println(sum(diff_matrix, dims=2))
    @test isapprox(res_gpu, distances_dijkstra)
end

function digraph_to_weightedgraph(
    g::SimpleDiGraph,
    weights::Dict{Tuple{Int,Int},T},
) where {T<:Real}
    g_w = SimpleWeightedDiGraph(nv(g))
    sources = zeros(Int, ne(g))
    destinations = zeros(Int, ne(g))
    edge_weights = zeros(Float64, ne(g))
    for (i, e) in enumerate(edges(g))
        u = src(e)
        v = dst(e)
        weight = weights[(u, v)]
        sources[i] = u
        destinations[i] = v
        edge_weights[i] = weight
    end
    g_w = SimpleWeightedDiGraph(sources, destinations, edge_weights)
    return g_w
end
