# Implementation of queries on Contraction Hierarchies contracted graphs

function shortest_path_CH(g_CH::CHGraph, source::Int)
	# Computes the shortest paths from source to all other nodes using the Contraction Hierarchy.
	# It uses a bidirectional Dijkstra search on the upward and downward graphs.
	g_up = g_CH.g_up
	g_down_rev = g_CH.g_down_rev
	weights = g_CH.weights_augmented
	node_order = g_CH.node_order
	distances = fill(Inf, nv(g_CH.g))
	forward!(g_up, weights, source, distances)
	backward!(node_order, g_down_rev, weights, source, distances)

	return distances

end

function forward!(g_up::G, weights::Dict{Tuple{Int,Int},Float64}, source::Int, distances::Vector{Float64}) where G <: AbstractGraph
	# Performs a forward search on the upward graph from the source node.
	# Returns the shortest distances from source to all reachable nodes in g_up.

	visited = zeros(Bool, nv(g_up))
	queue = PriorityQueue{Int, Float64}()

	distances[source] = 0.0
	push!(queue, source => 0.0)

	while !isempty(queue)
		u , dist_u = popfirst!(queue)
		visited[u] = true

		for (v, edge_weight) in neighbors_and_weights(g_up, u)
			if visited[v]
				continue
			end
			new_dist = dist_u + edge_weight
			if new_dist < get(distances, v, Inf)
				if !(v in keys(queue))
					push!(queue, v =>new_dist)
				else
					queue[v] = new_dist
				end
				distances[v] = new_dist
				
			end
		end
	end
end

function backward!(node_order::Vector{Int}, g_down_rev::G, weights::Dict{Tuple{Int,Int},Float64}, source::Int, distances::Vector{Float64}) where G <: AbstractGraph
	# Iterates through nodes in reverse rank order.
	# For each node, recompute the shortest distance from incoming edges : d[v] = min(d[v], d[u] + w(u,v))

	for node in 1:nv(g_down_rev)
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
    interval = w.colptr[u]:(w.colptr[u + 1] - 1)
    return zip(view(w.rowval, interval), view(w.nzval, interval))
end