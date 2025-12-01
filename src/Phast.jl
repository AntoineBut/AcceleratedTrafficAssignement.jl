# Implementation of queries on Contraction Hierarchies contracted graphs
""" 
    PhastStorageCPU{T<:Real}
Data structure to store distances for PHAST queries on CPU.
"""
struct PhastStorageCPU{T<:Real}
    distances::Matrix{T}
    parents::Matrix{Int}
end
function PhastStorageCPU(::Type{T}, nv::Int, nsources::Int = 1) where {T<:Real}
    distances = zeros(T, nv, nsources)
    parents = fill(-1, nv, nsources)
    return PhastStorageCPU{T}(distances, parents)
end
""" 
    PhastStorageGPU{T<:Real,Gpu_Vd<:AbstractVector{T},Gpu_Vb<:AbstractVector{Bool}}
Data structure to store distances for PHAST queries on GPU.
"""
struct PhastStorageGPU{
    T<:Real,
    Gpu_Md<:AbstractMatrix{T},
    Gpu_Mp<:AbstractMatrix{Int},
    Gpu_Vb<:AbstractVector{Bool},
}
    cpu_distances::Matrix{T}
    cpu_parents::Matrix{Int}
    device_distances::Gpu_Md
    device_parents::Gpu_Mp
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
    cpu_parents = fill(-1, nv, nsources)
    device_distances = KernelAbstractions.zeros(device, T, nv, nsources)
    device_parents = KernelAbstractions.zeros(device, Int, nv, nsources)
    device_temp = KernelAbstractions.zeros(device, T, nv, nsources)
    curr_level = KernelAbstractions.zeros(device, Bool, nv)
    return PhastStorageGPU{
        T,
        typeof(device_distances),
        typeof(device_parents),
        typeof(curr_level),
    }(
        cpu_distances,
        cpu_parents,
        device_distances,
        device_parents,
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
    storage.parents .= -1
    g_up = g_CH.g_up
    g_down_rev = g_CH.g_down_rev
    forward!(g_up, sources, storage.distances, storage.parents)
    backward!(g_down_rev, storage.distances, storage.parents)
end

function shortest_path_CH!(
    gpu_CH::gpu_CHGraph,
    sources::Vector{Int},
    storage::PhastStorageGPU{T,Gpu_Md,Gpu_Mp,Gpu_Vb},
) where {
    T<:Real,
    Gpu_Md<:AbstractMatrix{T},
    Gpu_Mp<:AbstractMatrix{Int},
    Gpu_Vb<:AbstractVector{Bool},
}
    # Computes the shortest paths from source to all other nodes using the Contraction Hierarchy.

    storage.cpu_distances .= typemax(T)
    storage.cpu_parents .= -1
    forward!(gpu_CH.g_up, sources, storage.cpu_distances, storage.cpu_parents)
    gpu_backward!(gpu_CH, storage)
end


function forward!(
    g_up::G,
    sources::Vector{Int},
    distances::Matrix{T},
    parents::Matrix{Int},
) where {G<:AbstractGraph,T<:Real}
    # Performs a forward search on the upward graph from the source node.
    # Returns the shortest distances from source to all reachable nodes in g_up.

    for (i, source) in enumerate(sources) # Iterate over sources
        visited = Set{Int}()
        queue = PriorityQueue{Int,T}()
        distances[source, i] = zero(T)
        parents[source, i] = 0
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
                    parents[v, i] = u
                    distances[v, i] = new_dist
                end
            end
        end
    end
end

function backward!(
    g_down_rev::G,
    distances::Matrix{T},
    parents::Matrix{Int},
) where {G<:AbstractGraph,T<:Real}
    # Iterates through nodes in rank order.
    # For each node, recompute the shortest distance from incoming edges : d[v] = min(d[v], d[u] + w(u,v))

    for node = 1:nv(g_down_rev)
        for i = 1:size(distances, 2) # Iterate over sources
            for (u, edge_weight) in neighbors_and_weights(g_down_rev, node)
                new_dist = distances[u, i] + edge_weight
                if new_dist < distances[node, i]
                    distances[node, i] = new_dist
                    parents[node, i] = u
                end
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
    parents = storage.cpu_parents
    for node = 1:gpu_CH.cpu_process
        for i = 1:size(distances, 2) # Iterate over sources
            for (u, edge_weight) in neighbors_and_weights(g_down_cpu, node)
                new_dist = distances[u, i] + edge_weight
                if new_dist < distances[node, i]
                    distances[node, i] = new_dist
                    parents[node, i] = u
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
    parents_gpu = storage.device_parents

    #TODO: only transfer the nodes whose distance has been set in forward pass
    copyto!(curr, distances)
    next .= curr
    copyto!(parents_gpu, parents)

    gpu_levels = gpu_CH.gpu_levels
    g_down_gpu = gpu_CH.g_down_rev_gpu

    for level = gpu_levels:-1:1
        phast_spmm!(next, parents_gpu, g_down_gpu, curr, gpu_CH.level_ranges[level])
        phast_spmm!(curr, parents_gpu, g_down_gpu, next, gpu_CH.level_ranges[level])
        # Swap curr and next
        #curr, next = next, curr

    end
    copyto!(storage.cpu_distances, curr)
    copyto!(storage.cpu_parents, parents_gpu)
end

# Stolen from Guillaume
function neighbors_and_weights(g::SimpleWeightedDiGraph, u::Integer)
    w = g.weights
    interval = w.colptr[u]:(w.colptr[u+1]-1)
    return zip(view(w.rowval, interval), view(w.nzval, interval))
end

function phast_spmm!(
    distances::DistMat,
    parents::ParentsMat,
    A::SparseGPUMatrixCSR{Tv,Ti},
    B::InputMat,
    range::UnitRange{Int},
) where {
    Tv,
    Ti<:Integer,
    ResType<:Number,
    InputType<:Number,
    DistMat<:AbstractMatrix{ResType},
    ParentsMat<:AbstractMatrix{Int},
    InputMat<:AbstractMatrix{InputType},
}
    monoid_neutral_element = typemax(ResType)
    NTuple = (size(B, 2), length(range))
    #NTuple = (length(range), size(B, 2))
    backend = get_backend(A)

    kernel! = phast_kernel!(backend)
    kernel!(
        distances,
        parents,
        A.rowptr,
        A.colval,
        A.nzval,
        B,
        range.start,
        monoid_neutral_element,
        ndrange = NTuple,
    )
end

@kernel function phast_kernel!(
    distance,
    parents,
    @Const(a_row_ptr),
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(B),
    @Const(range_start),
    monoid_neutral_element, # e.g., Inf for min
)
    col_B_C, row = @index(Global, NTuple)
    #row, col_B_C = @index(Global, NTuple)
    row += range_start - 1
    acc = monoid_neutral_element
    parent = -1
    for i = a_row_ptr[row]:(a_row_ptr[row+1]-1)
        col_A = a_col_val[i]
        new_dist = a_nz_val[i] + B[col_A, col_B_C]
        new_best = new_dist < acc

        acc = ifelse(new_best, new_dist, acc)
        parent = ifelse(new_best, col_A, parent)
    end

    parents[row, col_B_C] =
        ifelse(distance[row, col_B_C] <= acc, parents[row, col_B_C], parent)
    distance[row, col_B_C] = min(distance[row, col_B_C], acc)
end
