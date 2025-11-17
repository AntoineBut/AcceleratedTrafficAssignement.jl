# Implementation of queries on Contraction Hierarchies contracted graphs
""" 
    PhastStorageCPU{T<:Real}
Data structure to store distances for PHAST queries on CPU.
"""
struct PhastStorageCPU{T<:Real}
    distances::Vector{T}
end
function PhastStorageCPU(::Type{T}, nv::Int) where {T<:Real}
    distances = zeros(T, nv)
    return PhastStorageCPU{T}(distances)
end
""" 
    PhastStorageGPU{T<:Real,Gpu_Vd<:AbstractVector{T},Gpu_Vb<:AbstractVector{Bool}}
Data structure to store distances for PHAST queries on GPU.
"""
struct PhastStorageGPU{T<:Real,Gpu_Vd<:AbstractVector{T},Gpu_Vb<:AbstractVector{Bool}}
    cpu_distances::Vector{T}
    device_distances::Gpu_Vd
    device_temp::Gpu_Vd
    curr_level::Gpu_Vb
end
function PhastStorageGPU(
    ::Type{T},
    nv::Int,
    device::B,
) where {T<:Real,B<:KernelAbstractions.Backend}
    cpu_distances = fill(typemax(T), nv)
    device_distances = KernelAbstractions.zeros(device, T, nv)
    device_temp = KernelAbstractions.zeros(device, T, nv)
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
    storage = PhastStorageCPU(T, nv(g_CH.g))
    shortest_path_CH!(g_CH, source, storage)
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
    storage = PhastStorageGPU(T, nv(g_CH.g), get_backend(g_CH.g_down_rev_gpu))
    shortest_path_CH!(g_CH, source, storage)
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
        source::Int,
        storage::PhastStorageGPU{Gpu_V,T,Gpu_Vb},
    ) where {T <:Real,Gpu_V<:AbstractVector{T},Gpu_Vb<:AbstractVector{Bool}}
Computes the shortest paths from source to all other nodes using the Contraction Hierarchy on GPU.
Non-allocating version: fills the provided storage.
"""
function shortest_path_CH!(
    g_CH::CHGraph,
    source::Int,
    storage::PhastStorageCPU{T},
) where {T<:Real}
    # Computes the shortest paths from source to all other nodes using the Contraction Hierarchy.
    storage.distances .= typemax(T)
    g_up = g_CH.g_up
    g_down_rev = g_CH.g_down_rev
    forward!(g_up, source, storage.distances)
    backward!(g_down_rev, storage.distances)
end

function shortest_path_CH!(
    gpu_CH::gpu_CHGraph,
    source::Int,
    storage::PhastStorageGPU{T,Gpu_V,Gpu_Vb},
) where {T<:Real,Gpu_V<:AbstractVector{T},Gpu_Vb<:AbstractVector{Bool}}
    # Computes the shortest paths from source to all other nodes using the Contraction Hierarchy.

    storage.cpu_distances .= typemax(T)
    forward!(gpu_CH.g_up, source, storage.cpu_distances)

    gpu_backward!(gpu_CH, storage)
end


function forward!(
    g_up::G,
    source::Int,
    distances::Vector{T},
) where {G<:AbstractGraph,T<:Real}
    # Performs a forward search on the upward graph from the source node.
    # Returns the shortest distances from source to all reachable nodes in g_up.

    visited = Set{Int}()
    queue = PriorityQueue{Int,T}()

    distances[source] = zero(T)
    push!(queue, source => zero(T))

    while !isempty(queue)
        u, dist_u = popfirst!(queue)
        push!(visited, u)

        for (v, edge_weight) in neighbors_and_weights(g_up, u)
            if v in visited
                continue
            end
            new_dist = dist_u + edge_weight
            if new_dist < get(distances, v, typemax(T))
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

function backward!(g_down_rev::G, distances::Vector{T}) where {G<:AbstractGraph,T<:Real}
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

function gpu_backward!(gpu_CH::gpu_CHGraph, storage::PhastStorageGPU)
    # Iterates through nodes in rank order.
    # For each node, recompute the shortest distance from incoming edges : d[v] = min(d[v], d[u] + w(u,v))
    g_down_cpu = gpu_CH.g_down_rev_cpu
    # First levels on CPU
    distances = storage.cpu_distances
    for node = 1:gpu_CH.cpu_process
        for (u, edge_weight) in neighbors_and_weights(g_down_cpu, node)
            new_dist = distances[u] + edge_weight
            if new_dist < distances[node]
                distances[node] = new_dist
            end
        end
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

    for level = gpu_levels:-1:1
        #@. curr_level = (levels .== level) || (levels .== (level + 1))
        gpu_spmv!(
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
