using DataStructures

# Re-order vertices according to DFS
function reorder_vertices_dfs(graph::G, source::Int = 1) where {G<:AbstractGraph}
    visited = zeros(Bool, nv(graph))
    order = zeros(Int, nv(graph))
    index = 1
    source = source

    while index <= nv(graph)
        index = dfs!(graph, source, visited, order, index)
        # Find next unvisited node
        source = findfirst(!, visited)
        if source === nothing
            return order
        end
    end

end

function dfs!(
    graph::G,
    source::Int,
    visited::Vector{Bool},
    order::Vector{Int},
    index::Int,
) where {G<:AbstractGraph}
    stack = Stack{Int}()
    push!(stack, source)
    while !isempty(stack)
        node = pop!(stack)
        if !visited[node]
            order[node] = index
            index += 1
            visited[node] = true
            for neighbor in reverse(outneighbors(graph, node))
                push!(stack, neighbor)
            end
        end
    end
    return index
end

function permuted_graph(order::Vector{Int}, graph::G) where {G<:AbstractGraph}
    n = nv(graph)
    graph_permuted = SimpleDiGraph(n)
    indices = Vector{Int}(undef, n)
    for i = 1:n
        indices[order[i]] = i
    end
    for e in edges(graph)
        u = src(e)
        v = dst(e)
        add_edge!(graph_permuted, indices[u], indices[v])
    end
    return graph_permuted, indices
end

function permuted_graph(
    order::Vector{Int},
    graph::G,
    old_weights::Dict{Tuple{Int,Int},Float64},
) where {G<:AbstractGraph}
    n = nv(graph)
    graph_permuted = SimpleDiGraph(n)
    weights_permuted = Dict{Tuple{Int,Int},Float64}()
    indices = zeros(Int, n)
    for i = 1:n
        indices[order[i]] = i
    end
    for e in edges(graph)
        u = src(e)
        v = dst(e)
        add_edge!(graph_permuted, indices[u], indices[v])
        weights_permuted[(indices[u], indices[v])] = old_weights[(u, v)]
    end
    return graph_permuted, weights_permuted, indices
end
