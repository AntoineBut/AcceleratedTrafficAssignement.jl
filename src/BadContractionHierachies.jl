# This implementation is awfully inefficient and is kept for reference only.

# The CHGraph type represents a contraction hierarchy graph

struct CHGraph{
	G <: AbstractGraph
}
	g::G # Original graph
	node_order::Vector{Int} # Ordering of nodes
	g_augmented::G # Augmented graph with shortcuts
	g_up::G # Upward graph
	g_down::G # Downward graph
end




function compute_CH(graph::G where G <: AbstractGraph)
	# The CH algorithm computes a contraction hierarchy for the given graph.
	
	# Create the CH representation of the graph: augment with shortcuts
	# The node ordering is also computed during this step.
	g_augmented, node_order = augment_graph(graph)

	# Compute g_up and g_down graphs using g_augmented.
	g_up, g_down= compute_up_down_graphs(g_augmented, node_order)

	# We return the CHGraph, 
	return CHGraph(graph, node_order, g_augmented, g_up, g_down)
end

function augment_graph(graph::G where G <: AbstractGraph)
	# This function augments the graph by adding shortcuts and computes the node ordering.
	
	queue = PriorityQueue{Int, Int}() # Node priority queue
	node_order = Vector{Int}(undef, nv(graph)) # Node ordering, will be filled during contraction
	order_index = 0 # Current index in the ordering
	contracted = zeros(Bool, nv(graph)) # Track contracted nodes

	graph = deepcopy(graph) # Work on a copy of the graph as we will modify it
	g_augmented = deepcopy(graph) # Start with the original graph

	# For all nodes, run witness search to determine the number of shortcuts needed
	# The initial priority is the edge difference
	s = 0
	for node in 1:nv(graph)
		ed = edge_difference(graph, node)
		s += ed
		enqueue!(queue, node, 5*ed)
	end
	#println("Initial total edge difference: $s")

	# Simplified contraction process: contract nodes in order of priority
	while !isempty(queue)
		node, p = popfirst!(queue)
		# Assigne lowest order to the contracted node
		node_order[node] = order_index
		order_index += 1
		contracted[node] = true
		WS = 0
		# For each neighbor, add shortcuts as needed
		inneighbors_list = collect(inneighbors(graph, node))
		outneighbors_list = collect(outneighbors(graph, node))

		# Handle isolated, source, and sink nodes
		# We only remove edges because removing vertices messes up the indexing
		if isempty(inneighbors_list) || isempty(outneighbors_list)
			remove_node!(graph, node, inneighbors_list, outneighbors_list)
			continue
		end
		
		in_weights = [get_weight(graph, n, node) for n in inneighbors_list]
		out_weights = [get_weight(graph, node, m) for m in outneighbors_list]

		for n in inneighbors_list
			search_dist = get_weight(graph, n, node) + maximum(out_weights)
			witness_distances = witness_search(graph, n, node, search_dist)
			WS += 1
			for (i, m) in enumerate(outneighbors_list)
				
				direct_distance = get_weight(graph, n, node) + get_weight(g_augmented, node, m)
				if witness_distances[i] > direct_distance
					# Add shortcut edge
					add_edge!(graph, n, m, direct_distance)
					add_edge!(g_augmented, n, m, direct_distance)
					# Update priority queue if needed
					#if !contracted[m]
					#	p_m = queue[m]
					#	queue[m] = p_m + 1 # Increase priority due to new shortcut
					#end
				end
			end
		end

		# Finally, remove the edges of the contracted node
		remove_node!(graph, node, inneighbors_list, outneighbors_list)
	end
	return g_augmented, node_order
end

function remove_node!(g::G, node::Int, inneighbors::Vector{Int}, outneighbors::Vector{Int}) where G <: AbstractGraph
	# This function removes a node from the graph by removing all its edges.
	# It updates the weights dictionary accordingly.
	
	# Remove in-edges
	for n in inneighbors
		rem_edge!(g, n, node)
		
	end
	# Remove out-edges
	for m in outneighbors
		rem_edge!(g, node, m)
	end
end


function edge_difference(g::G where G <: AbstractGraph, node::Int)
	# This function computes the edge difference for a given node.
	# Edge difference = (#shortcuts added) - (#edges of the node)
	
	inneighbors_list = inneighbors(g, node)
	outneighbors_list = outneighbors(g, node)
	if isempty(inneighbors_list) || isempty(outneighbors_list)
		return - (length(inneighbors_list) + length(outneighbors_list))
	end
	in_weights = [get_weight(g, node, m) for m in inneighbors_list]
	out_weights = [get_weight(g, node, m) for m in outneighbors_list]

	num_edges = length(inneighbors_list) + length(outneighbors_list)
	num_shortcuts_added = 0

	# For each pair of neighbors, check if a shortcut is needed
	for n in inneighbors_list
		search_dist = get_weight(g, n, node) + maximum(out_weights)
		witness_distances = witness_search(g, n, node, search_dist)
		for (i, m) in enumerate(outneighbors_list)
			if m == n
				continue
			end
			direct_distance = get_weight(g, n, node) + get_weight(g, node, m)
			if witness_distances[i] > direct_distance
				num_shortcuts_added += 1
			end
		end
		
	end
	return num_shortcuts_added - num_edges
end

function witness_search(g::G where G <: AbstractGraph, source::Int, skip::Int, max_distance::Float64)
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

		for v in outneighbors(g, u)
			if v == skip || v in visited
				continue
			end
			edge_weight = get_weight(g, u, v)
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

function compute_up_down_graphs(g_augmented::G where G <: AbstractGraph, node_order::Vector{Int})
	# This function computes the upward and downward graphs from the augmented graph.
	
	# g_up contains edges from lower to higher order nodes
	# g_down contains edges from higher to lower order nodes

	edge_list_up = (Int[], Int[], Float64[])
	edge_list_down = (Int[], Int[], Float64[])

	for e in edges(g_augmented)
		u = src(e)
		v = dst(e)
		if node_order[u] < node_order[v]
			push!(edge_list_up[1], u)
			push!(edge_list_up[2], v)
			push!(edge_list_up[3], get_weight(g_augmented, u, v))
		else
			push!(edge_list_down[1], u)
			push!(edge_list_down[2], v)
			push!(edge_list_down[3], get_weight(g_augmented, u, v))
		end
	end
	# constructor: SimpleWeightedDiGraph(srcs, dsts, weights)
	g_up = SimpleWeightedDiGraph(edge_list_up[1], edge_list_up[2], edge_list_up[3])
	g_down = SimpleWeightedDiGraph(edge_list_down[1], edge_list_down[2], edge_list_down[3])
	return g_up, g_down
end

