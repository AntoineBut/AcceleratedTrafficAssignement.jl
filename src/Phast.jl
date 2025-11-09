# Implementation of queries on Contraction Hierarchies contracted graphs

function shortest_path_CH(g_CH::CHGraph, source::Int)
    # Computes the shortest paths from source to all other nodes using the Contraction Hierarchy.
    # It uses a bidirectional Dijkstra search on the upward and downward graphs.
    g_up = g_CH.g_up
    g_down_rev = g_CH.g_down_rev
    distances = fill(Inf, nv(g_CH.g))
    forward!(g_up, source, distances)
    backward!(g_down_rev, distances)

    return distances

end

function forward!(g_up::G, source::Int, distances::Vector{Float64}) where {G<:AbstractGraph}
    # Performs a forward search on the upward graph from the source node.
    # Returns the shortest distances from source to all reachable nodes in g_up.

    visited = zeros(Bool, nv(g_up))
    queue = PriorityQueue{Int,Float64}()

    distances[source] = 0.0
    push!(queue, source => 0.0)

    while !isempty(queue)
        u, dist_u = popfirst!(queue)
        visited[u] = true

        for (v, edge_weight) in neighbors_and_weights(g_up, u)
            if visited[v]
                continue
            end
            new_dist = dist_u + edge_weight
            if new_dist < get(distances, v, Inf)
                if !(v in keys(queue))
                    push!(queue, v => new_dist)
                else
                    queue[v] = new_dist
                end
                distances[v] = new_dist

            end
        end
    end
end

function backward!(g_down_rev::G, distances::Vector{Float64}) where {G<:AbstractGraph}
    # Iterates through nodes in rank order.
    # For each node, recompute the shortest distance from incoming edges : d[v] = min(d[v], d[u] + w(u,v))

    for node = 1:nv(g_down_rev)
        for (u, edge_weight) in neighbors_and_weights(g_down_rev, node)
            new_dist = distances[u] + edge_weight
            if new_dist < distances[node]
                distances[node] = new_dist
            end
        end
    end
end

# Stolen from Guillaume
function neighbors_and_weights(g::SimpleWeightedDiGraph, u::Integer)
    w = g.weights
    interval = w.colptr[u]:(w.colptr[u+1]-1)
    return zip(view(w.rowval, interval), view(w.nzval, interval))
end

function gpu_shortest_path_CH(
    cpu_CH::CHGraph,
    gpu_CH::gpu_CHGraph,
    source::Int,
    distances1::CuArray{Float64,1},
    distances2::CuArray{Float64,1},
    gpu_levels = 10::Int64,
)
    # Computes the shortest paths from source to all other nodes using the Contraction Hierarchy.
    # It uses a bidirectional Dijkstra search on the upward and downward graphs.
    g_up = gpu_CH.g_up
    g_down_rev = gpu_CH.g_down_rev
    g_down_cpu = cpu_CH.g_down_rev
    levels = gpu_CH.levels
    levels_cpu = cpu_CH.levels

    distances = fill(Inf, nv(gpu_CH.g))
    forward!(g_up, source, distances)

    curr_level = CUDA.zeros(Bool, nv(gpu_CH.g))
    gpu_backward!(
        g_down_cpu,
        levels_cpu,
        g_down_rev,
        levels,
        curr_level,
        distances,
        distances1,
        distances2,
        gpu_levels,
    )

    return distances2

end

function gpu_backward!(
    g_down_cpu::SimpleWeightedDiGraph,
    levels_cpu::Vector{Int},
    g_down_rev::GPU_graph,
    levels::CuArray{Int64,1},
    curr_level::CuArray{Bool,1},
    distances::Vector{Float64},
    curr::CuArray{Float64,1},
    next::CuArray{Float64,1},
    gpu_levels::Int64,
) where {GPU_graph<:AbstractSparseGPUMatrix}
    # Iterates through nodes in rank order.
    # For each node, recompute the shortest distance from incoming edges : d[v] = min(d[v], d[u] + w(u,v))
    #println("GPU levels: $GPU_LEVELS")
    # First levels on CPU
    for node = 1:nv(g_down_cpu)
        if levels_cpu[node] <= gpu_levels
            break
        end
        for (u, edge_weight) in neighbors_and_weights(g_down_cpu, node)
            new_dist = distances[u] + edge_weight
            if new_dist < distances[node]
                distances[node] = new_dist
            end
        end
    end
    # Remaining levels on the GPU
    copyto!(curr, distances)
    copyto!(next, distances)
    for level = ceil(Int, gpu_levels):-1:0
        @. curr_level = (levels .== level) || (levels .== (level + 1))
        #println("$level --  $(sum(curr_level))")
        gpu_spmv!(
            next,
            g_down_rev,
            curr,
            mul = GPUGraphs_add,
            add = GPUGraphs_min,
            accum = GPUGraphs_min,
            mask = curr_level,
        )
        #gpu_spmv!(
        #    curr,
        #    g_down_rev,
        #    next,
        #    mul = GPUGraphs_add,
        #    add = GPUGraphs_min,
        #    accum = GPUGraphs_min,
        #    mask = curr_level,
        #)
        curr, next = next, curr

    end
end
