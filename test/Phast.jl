# Tests for the PHAST querry algorithm

function test_phast_queries(CH::CHGraph)
    g = CH.g
    g_w = digraph_to_weightedgraph(g, CH.weights)
    weights_matrix = convert(SparseMatrixCSC{Float64,Int64}, adjacency_matrix(g_w))
    source = 1
    distances_phast = shortest_path_CH(CH, source).distances
    distances_dijkstra = dijkstra_shortest_paths(g_w, source, weights_matrix).dists

    @test isapprox(distances_phast, distances_dijkstra; atol = 1e-8)
end



function digraph_to_weightedgraph(g::SimpleDiGraph, weights::Dict{Tuple{Int,Int},Float64})
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
