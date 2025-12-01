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
    # CPU #
    res_cpu = shortest_path_CH(CH, sources)
    distances_phast_cpu = res_cpu.distances
    parents_phast_cpu = res_cpu.parents
    @test isapprox(distances_phast_cpu, distances_dijkstra)

    # GPU #
    res_gpu = shortest_path_CH(gpu_CH, sources)
    distances_phast_gpu = collect(res_gpu.cpu_distances)
    parents_phast_gpu = collect(res_gpu.cpu_parents)
    diff_matrix = .!(isapprox.(distances_phast_gpu, distances_phast_cpu))
    #println(diff_matrix)
    #println(sum(diff_matrix, dims=1))
    #println(sum(diff_matrix, dims=2))
    @test isapprox(distances_phast_gpu, distances_dijkstra)
    @test verify_parents(CH, sources, distances_phast_cpu, parents_phast_cpu)
    @test verify_parents(gpu_CH, sources, distances_phast_gpu, parents_phast_gpu)
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

function verify_parents(
    ch::AbstractCHGraph,
    origins::AbstractVector,
    distances::AbstractMatrix,
    parents::AbstractMatrix,
)
    err = 0
    for (i, o) in enumerate(origins)
        @test parents[o, i] == 0
        # If u is parent of v, then dist(u) + weight(u,v) == dist(v)
        for v = 1:nv(ch.g_augmented)
            p = parents[v, i]
            if p == -1 && distances[v, i] != typemax(eltype(distances))
                err += 1
                #println("Node $v from origin $o has no parent but distance is finite: $(distances[v, i])")
            end
            if p != 0 && p != -1
                edge = (p, v)
                w = get(ch.weights_augmented, edge, typemax(eltype(distances)))
                dist_check = distances[p, i] + w
                if !isapprox(dist_check, distances[v, i])
                    err += 1
                    #println("Distance mismatch for node $v from origin $o: computed $(distances[v, i]), expected $dist_check")
                end
            end
        end
    end
    return err == 0
end
