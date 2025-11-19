# Implementation of queries on Contraction Hierarchies contracted graphs
""" 
    PhastStorageCPU{T<:Real}
Data structure to store distances for PHAST queries on CPU.
"""
struct PhastStorageCPU{T<:Real}
    distances::Matrix{T}
end
function PhastStorageCPU(::Type{T}, nv::Int, nsources::Int = 1) where {T<:Real}
    distances = zeros(T, nv, nsources)
    return PhastStorageCPU{T}(distances)
end
""" 
    PhastStorageGPU{T<:Real,Gpu_Vd<:AbstractVector{T},Gpu_Vb<:AbstractVector{Bool}}
Data structure to store distances for PHAST queries on GPU.
"""
struct PhastStorageGPU{T<:Real,Gpu_Md<:AbstractMatrix{T},Gpu_Vb<:AbstractVector{Bool}}
    cpu_distances::Matrix{T}
    device_distances::Gpu_Md
    device_temp::Gpu_Md
    curr_level::Gpu_Vb
end
function PhastStorageGPU(
    device::B,
    ::Type{T},
    nv::Int,
    nsources::Int = 1,
) where {T<:Real,B<:KernelAbstractions.Backend}
    cpu_distances = fill(typemax(T), nv, nsources)
    device_distances = KernelAbstractions.zeros(device, T, nv, nsources)
    device_temp = KernelAbstractions.zeros(device, T, nv, nsources)
    curr_level = KernelAbstractions.zeros(device, Bool, nv)
    return PhastStorageGPU{T,typeof(device_distances),typeof(curr_level)}(
        cpu_distances,
        device_distances,
        device_temp,
        curr_level,
    )
end
"""
    shortest_path_CH(g_CH::CHGraph, source::Int)
Computes the shortest paths from source to all other nodes using the Contraction Hierarchy.
Allocates and returns a PhastStorageCPU instance.
"""
function shortest_path_CH(
    g_CH::CHGraph{G,G1,T},
    source::Int,
) where {G<:AbstractGraph,G1<:AbstractGraph,T<:Real}
    return shortest_path_CH(g_CH, [source])
end

"""
    shortest_path_CH(g_CH::CHGraph, sources::Vector{Int})
Computes the shortest paths from sources to all other nodes using the Contraction Hierarchy.
Allocates and returns a PhastStorageCPU instance.
"""
function shortest_path_CH(
    g_CH::CHGraph{G,G1,T},
    sources::Vector{Int},
) where {G<:AbstractGraph,G1<:AbstractGraph,T<:Real}
    storage = PhastStorageCPU(T, nv(g_CH.g), length(sources))
    shortest_path_CH!(g_CH, sources, storage)
    return storage
end

"""
    shortest_path_CH(
        g_CH::gpu_CHGraph,
        source::Int,
    ) where {T <:Real}
Computes the shortest paths from source to all other nodes using the Contraction Hierarchy on GPU.
Allocates a PhastStorageGPU instance.
"""
function shortest_path_CH(
    g_CH::gpu_CHGraph{G,G1,G2,Gpu_V,T},
    source::Int,
) where {
    G<:AbstractGraph,
    G1<:AbstractGraph,
    G2<:AbstractSparseGPUMatrix,
    Gpu_V<:AbstractVector,
    T<:Real,
}
    return shortest_path_CH(g_CH, [source])
end

"""
    shortest_path_CH(
        g_CH::gpu_CHGraph,
        sources::Vector{Int},
    ) where {T <:Real}
Computes the shortest paths from sources to all other nodes using the Contraction Hierarchy on GPU.
Allocates a PhastStorageGPU instance.
"""
function shortest_path_CH(
    g_CH::gpu_CHGraph{G,G1,G2,Gpu_V,T},
    sources::Vector{Int},
) where {
    G<:AbstractGraph,
    G1<:AbstractGraph,
    G2<:AbstractSparseGPUMatrix,
    Gpu_V<:AbstractVector,
    T<:Real,
}
    storage =
        PhastStorageGPU(get_backend(g_CH.g_down_rev_gpu), T, nv(g_CH.g), length(sources))
    shortest_path_CH!(g_CH, sources, storage)
    return storage
end

"""
    shortest_path_CH(
        g_CH::CHGraph
        source::Int,
        storage::PhastStorageCPU{T},
    ) where {T <:Real}

Computes the shortest paths from source to all other nodes using the Contraction Hierarchy on CPU.
Non-allocating version: fills the provided storage.
"""

"""
    shortest_path_CH(
        gpu_CH::gpu_CHGraph,
        sources::Vector{Int},
        storage::PhastStorageGPU{Gpu_V,T,Gpu_Vb},
    ) where {T <:Real,Gpu_V<:AbstractVector{T},Gpu_Vb<:AbstractVector{Bool}}
Computes the shortest paths from sources to all other nodes using the Contraction Hierarchy on GPU.
Non-allocating version: fills the provided storage.
"""
function shortest_path_CH!(
    g_CH::CHGraph,
    sources::Vector{Int},
    storage::PhastStorageCPU{T},
) where {T<:Real}
    # Computes the shortest paths from sources to all other nodes using the Contraction Hierarchy.
    storage.distances .= typemax(T)
    g_up = g_CH.g_up
    g_down_rev = g_CH.g_down_rev
    forward!(g_up, sources, storage.distances)
    backward!(g_down_rev, storage.distances)
end

function shortest_path_CH!(
    gpu_CH::gpu_CHGraph,
    sources::Vector{Int},
    storage::PhastStorageGPU{T,Gpu_Md,Gpu_Vb},
) where {T<:Real,Gpu_Md<:AbstractMatrix{T},Gpu_Vb<:AbstractVector{Bool}}
    # Computes the shortest paths from source to all other nodes using the Contraction Hierarchy.

    storage.cpu_distances .= typemax(T)
    forward!(gpu_CH.g_up, sources, storage.cpu_distances)

    gpu_backward!(gpu_CH, storage)
end


function forward!(
    g_up::G,
    sources::Vector{Int},
    distances::Matrix{T},
) where {G<:AbstractGraph,T<:Real}
    # Performs a forward search on the upward graph from the source node.
    # Returns the shortest distances from source to all reachable nodes in g_up.

    for (i, source) in enumerate(sources) # Iterate over sources
        visited = Set{Int}()
        queue = PriorityQueue{Int,T}()
        distances[source, i] = zero(T)
        push!(queue, source => zero(T))
        while !isempty(queue)
            u, dist_u = popfirst!(queue)
            push!(visited, u)
            for (v, edge_weight) in neighbors_and_weights(g_up, u)
                if v in visited
                    continue
                end
                new_dist = dist_u + edge_weight
                if new_dist < distances[v, i]
                    if !(v in keys(queue))
                        push!(queue, v => new_dist)
                    else
                        queue[v] = new_dist
                    end
                    distances[v, i] = new_dist
                end
            end
        end
    end
end

function backward!(g_down_rev::G, distances::Matrix{T}) where {G<:AbstractGraph,T<:Real}
    # Iterates through nodes in rank order.
    # For each node, recompute the shortest distance from incoming edges : d[v] = min(d[v], d[u] + w(u,v))

    for node = 1:nv(g_down_rev)
        for i = 1:size(distances, 2) # Iterate over sources
            for (u, edge_weight) in neighbors_and_weights(g_down_rev, node)
                new_dist = distances[u, i] + edge_weight
                if new_dist < distances[node, i]
                    distances[node, i] = new_dist
                end
            end
        end
    end
end

function gpu_backward!(gpu_CH::gpu_CHGraph, storage::PhastStorageGPU)
    # Iterates through nodes in rank order.
    # For each node, recompute the shortest distance from incoming edges : d[v] = min(d[v], d[u] + w(u,v))
    levels_cpu = collect(gpu_CH.levels)
    g_down_cpu = gpu_CH.g_down_rev_cpu
    # First levels on CPU
    distances = storage.cpu_distances
    for node = 1:gpu_CH.cpu_process
        for i = 1:size(distances, 2) # Iterate over sources
            for (u, edge_weight) in neighbors_and_weights(g_down_cpu, node)
                new_dist = distances[u, i] + edge_weight
                if new_dist < distances[node, i]
                    distances[node, i] = new_dist
                end
            end
        end
        #if levels_cpu[node] != levels_cpu[node + 1]
        #    println("Processed level $(levels_cpu[node]) on CPU up to node $node")
        #end
    end

    # Then levels on GPU
    curr_level = storage.curr_level
    # Remaining levels on the GPU
    curr = storage.device_distances
    next = storage.device_temp

    #TODO: only transfer the nodes whose distance has been set in forward pass
    copyto!(curr, distances)
    next .= curr

    gpu_levels = gpu_CH.gpu_levels
    levels = gpu_CH.levels
    g_down_gpu = gpu_CH.g_down_rev_gpu

    for level = gpu_levels+1:-1:1
        #@. curr_level = (levels == level) || (levels == (level + 1))
         (
            next,
            g_down_gpu,
            curr,
            mul = GPUGraphs_add,
            add = GPUGraphs_min,
            accum = GPUGraphs_min,
            range = gpu_CH.level_ranges[level],
            #mask = curr_level,
        )
        #gpu_spmv!(
        #    curr,
        #    g_down_gpu,
        #    next,
        #    mul = GPUGraphs_add,
        #    add = GPUGraphs_min,
        #    accum = GPUGraphs_min,
        #    mask = curr_level,
        #)
        curr, next = next, curr
    end
    #copyto!(storage.cpu_distances, curr)
end

# Stolen from Guillaume
function neighbors_and_weights(g::SimpleWeightedDiGraph, u::Integer)
    w = g.weights
    interval = w.colptr[u]:(w.colptr[u+1]-1)
    return zip(view(w.rowval, interval), view(w.nzval, interval))
end
