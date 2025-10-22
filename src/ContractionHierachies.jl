# This file contains all the core functions for CH graph preprocessing

# The CHGraph type represents a contraction hierarchy graph

struct CHGraph{
	G <: AbstractGraph
}
	g::G # Original graph
	weights::Dict{Tuple{Int,Int},Float64} # Edge weights
	node_order::Vector{Int} # Ordering of nodes
	g_augmented::G # Augmented graph with shortcuts
	weights_augmented::Dict{Tuple{Int,Int},Float64} # Weights of augmented graph
	g_up::G # Upward graph
	g_down::G # Downward graph
end




function compute_CH(graph::G, weights::Dict{Tuple{Int,Int},Float64}) where G <: AbstractGraph
	# The CH algorithm computes a contraction hierarchy for the given graph.
	
	# Create the CH representation of the graph: augment with shortcuts
	# The node ordering is also computed during this step.
	g_augmented, weights_augmented, node_order = augment_graph(graph, weights)

	# Compute g_up and g_down graphs using g_augmented.
	g_up, g_down = compute_up_down_graphs(g_augmented, weights, node_order)

	# We return the CHGraph, 
	return CHGraph(graph, weights, node_order, g_augmented, weights_augmented, g_up, g_down)
end

function augment_graph(graph::G, org_weights::Dict{Tuple{Int,Int},Float64}) where G <: AbstractGraph
	# This function augments the graph by adding shortcuts and computes the node ordering.
	
	queue = PriorityQueue{Int, Int}() # Node priority queue
	node_order = Vector{Int}(undef, nv(graph)) # Node ordering, will be filled during contraction
	order_index = 0 # Current index in the ordering

	g_augmented = deepcopy(graph) # Start with the original graph
	weights = deepcopy(org_weights) # Start with original weights

	# For all nodes, run witness search to determine the number of shortcuts needed
	# The initial priority is the edge difference
	s = 0
	for node in 1:nv(graph)
		ed = edge_difference(g_augmented, weights, node)
		s += ed
		enqueue!(queue, node, ed)
	end
	#println("Initial total edge difference: $s")

	# Simplified contraction process: contract nodes in order of priority
	while !isempty(queue)
		node, p = popfirst!(queue)
		# Assigne lowest order to the contracted node
		node_order[node] = order_index
		order_index += 1

		# For each neighbor, add shortcuts as needed
		inneighbors_list = inneighbors(g_augmented, node)
		outneighbors_list = outneighbors(g_augmented, node)

		in_weights = [weights[(m, node)] for m in inneighbors_list]
		out_weights = [weights[(node, m)] for m in outneighbors_list]
		search_dist = maximum(vcat(in_weights, [0.0])) + maximum(vcat(out_weights, [0.0]))

		for n in inneighbors_list
			witness_distances = witness_search(g_augmented, weights, n, node, search_dist)
			for (i, m) in enumerate(outneighbors_list)
				if m == n
					continue
				end
				direct_distance = weights[(n, node)] + weights[(node, m)]
				if witness_distances[i] > direct_distance
					# Add shortcut edge
					add_edge!(g_augmented, n, m)
					push!(weights, (n, m) => direct_distance)
				end
			end
		end
	end
	return g_augmented, weights, node_order
end

function edge_difference(g::G, weights::Dict{Tuple{Int,Int},Float64}, node::Int) where G <: AbstractGraph
	# This function computes the edge difference for a given node.
	# Edge difference = (#shortcuts added) - (#edges of the node)
	
	inneighbors_list = inneighbors(g, node)
	outneighbors_list = outneighbors(g, node)
	in_weights = [weights[(m, node)] for m in inneighbors_list]
	out_weights = [weights[(node, m)] for m in outneighbors_list]
	search_dist = maximum(vcat(in_weights, [0.0])) + maximum(vcat(out_weights, [0.0]))

	num_edges = length(inneighbors_list) + length(outneighbors_list)
	num_shortcuts_added = 0

	# For each pair of neighbors, check if a shortcut is needed
	for n in inneighbors_list
		witness_distances = witness_search(g, weights, n, node, search_dist)

		for (i, m) in enumerate(outneighbors_list)
			
			direct_distance = weights[(n, node)] + weights[(node, m)]
			if witness_distances[i] > direct_distance
				num_shortcuts_added += 1
			end
		end
		
	end
	#println("Node $node: Edge difference = $num_shortcuts_added - $num_edges")
	return num_shortcuts_added - num_edges
end

function witness_search(g::G, weights::Dict{Tuple{Int,Int},Float64}, source::Int, skip::Int, max_distance::Float64) where G <: AbstractGraph
	# This function performs a witness search from source to all neighbors of skip.
	# It avoids going through the skip node.
	# Its stops when all neighbors have been visited or if distances exceed max_distance.
	# It returns the shortest distance found to neighbors of skip or Inf if it exceeds max_distance.
	
	# Implement a Dijkstra-like search with early stopping
	distances = Dict{Int, Float64}()
	visited = Set{Int}()
	queue = PriorityQueue{Int, Float64}()

	distances[source] = 0.0
	push!(queue, source => 0.0)

	targets = outneighbors(g, skip)

	while !isempty(queue)
		u , dist_u = popfirst!(queue)
		if u in visited
			continue
		end
		push!(visited, u)

		# Early stopping if we exceed max_distance
		if dist_u > max_distance
			break
		end

		for v in neighbors(g, u)
			if v == skip || v in visited
				continue
			end
			edge_weight = weights[(u, v)]
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

	distances_to_targets = [get(distances, t, Inf) for t in targets]
	return distances_to_targets
end

function compute_up_down_graphs(g_augmented::G, weights::Dict{Tuple{Int,Int},Float64}, node_order::Vector{Int}) where G <: AbstractGraph
	# This function computes the upward and downward graphs from the augmented graph.
	
	# g_up contains edges from lower to higher order nodes
	# g_down contains edges from higher to lower order nodes
	# The weights of both graphs are the same as in g_augmented.

	g_up = SimpleDiGraph(nv(g_augmented))
	g_down = SimpleDiGraph(nv(g_augmented))

	for e in edges(g_augmented)
		u = src(e)
		v = dst(e)
		if node_order[u] < node_order[v]
			add_edge!(g_up, u, v)
		else
			add_edge!(g_down, u, v)
		end
	end

	return g_up, g_down
end

