# This file contains all the core functions for CH graph preprocessing

# The CHGraph type represents a contraction hierarchy graph

struct CHGraph{
	G <: AbstractGraph,
	G2 <: AbstractGraph
}
	g::G # Original graph
	weights::Dict{Tuple{Int,Int},Float64} # Edge weights
	node_order::Vector{Int} # Ordering of nodes
	levels::Vector{Int} # Levels of nodes in the hierarchy
	g_augmented::G # Augmented graph with shortcuts
	weights_augmented::Dict{Tuple{Int,Int},Float64} # Weights of augmented graph
	g_up::G2 # Upward graph
	g_down_rev::G2 # Downward graph, stored reversed for easier backward search
	reordering::Vector{Int} # Reordering of nodes by levels (used to map back)
end

function cost(edge_diff::Int, n_contr_neighbors::Int)
	# A cost function to prioritize nodes during contraction
	return 10 * edge_diff + min(n_contr_neighbors, 10)
end


function compute_CH(graph::G, weights::Dict{Tuple{Int,Int},Float64}) where G <: AbstractGraph
	# The CH algorithm computes a contraction hierarchy for the given graph.
	
	# Create the CH representation of the graph: augment with shortcuts
	# The node ordering is also computed during this step.
	weights_augmented = deepcopy(weights)
	g_augmented = deepcopy(graph)
	node_order, levels = augment_graph!(graph, g_augmented, weights, weights_augmented)
	# Re-order nodes by levels
	reordering = sortperm(levels, rev=true)

	#reordering = collect(1:nv(graph))
	g_augmented_p, weights_augmented_p, indices = permuted_graph(reordering, g_augmented, weights_augmented)
	graph_p, weights_p, _ = permuted_graph(reordering, graph, weights)
	#g_augmented_p, weights_augmented_p = g_augmented, weights_augmented
	node_order_p = node_order[reordering]
	levels_p = levels[reordering]
	# Compute g_up and g_down graphs using g_augmented.
	g_up, g_down_rev = compute_up_down_graphs(g_augmented_p, weights_augmented_p, node_order_p)

	# We return the CHGraph, 
	return CHGraph(graph_p, weights_p, node_order_p, levels_p, g_augmented_p, weights_augmented_p, g_up, g_down_rev, indices)
end

function augment_graph!(org_graph::G, g_augmented::G, org_weights::Dict{Tuple{Int,Int},Float64}, weights_augmented::Dict{Tuple{Int,Int},Float64}) where G <: AbstractGraph
	# This function augments the graph by adding shortcuts and computes the node ordering.
	graph = deepcopy(org_graph) # Work on a copy of the graph as we will modify it
	weights = deepcopy(org_weights) # Start with original weights
	
	node_order = Vector{Int}(undef, nv(graph)) # Node ordering, will be filled during contraction
	order_index = 0 # Current index in the ordering
	processed = zeros(Bool, nv(graph)) # Track processed nodes
	n_contr_neighbors = zeros(Int, nv(graph)) # Track number of shortcuts per node
	levels = zeros(Int, nv(graph)) # Track levels of nodes (not used currently)


	# For all nodes, run witness search to determine the number of shortcuts needed
	# The initial priority is the edge difference
	queue = recompute_queue(graph, weights, processed, n_contr_neighbors)

	# Simplified contraction process: contract nodes in order of priority
	while !isempty(queue)
		## Progress
		order_index += 1
		progress = order_index / length(node_order)
		if order_index % 100 == 0
			done = Int(floor(progress * 10))
			print("\r [" * repeat('█', done) * repeat('*', 10 - done) * "] - $(round(progress * 100, digits=2))%")
		end

		## Recompute
		interval = Int(ceil(0.1 * length(node_order)))
		if order_index % interval == 0
			# Recompute priorities every 10% of contractions to reflect graph changes
			queue = recompute_queue(graph, weights, processed, n_contr_neighbors)
		end

		## Select next node to contract
		found = false
		node = 0
		while !found
			node, _ = popfirst!(queue)
			if isempty(queue)
				found = true
				break
			end
			# Lazy updating
			cost_node = cost(edge_difference(graph, weights, node), n_contr_neighbors[node])
			node2, p2 = peek(queue)
			if cost_node <= p2
				found = true
			else
				# Reinsert with updated priority
				enqueue!(queue, node, cost_node)
			end
		end
		
		## Contracting the selected node
		processed[node] = true
		node_order[node] = order_index
	
		inneighbors_list = collect(inneighbors(graph, node))
		outneighbors_list = collect(outneighbors(graph, node))

		# No shortcuts need for these nodes
		if isempty(inneighbors_list) || isempty(outneighbors_list)
			remove_node!(graph, weights, node, inneighbors_list, outneighbors_list, levels)
			continue
		end

		in_weights = [weights[(n, node)] for n in inneighbors_list]
		out_weights = [weights[(node, m)] for m in outneighbors_list]

		# Explore neighbors
		for n in inneighbors_list
			search_dist = maximum(in_weights) + maximum(out_weights)
			witness_distances = witness_search(graph, weights, n, node, search_dist)
			for (i, m) in enumerate(outneighbors_list)
				n_contr_neighbors[n] += 1
				n_contr_neighbors[m] += 1
				direct_distance = weights[(n, node)] + weights[(node, m)]
				if witness_distances[i] > direct_distance
					# Add shortcut edge
					add_edge!(g_augmented, n, m) # For storing all shortcuts
					add_edge!(graph, n, m) # For computing further shortcuts
					push!(weights, (n, m) => direct_distance)
					push!(weights_augmented, (n, m) => direct_distance)
					
				end
			end
		end
		# Finally, remove the edges of the contracted node
		remove_node!(graph, weights, node, inneighbors_list, outneighbors_list, levels)
	end
	println() # New line after progress bar
	return node_order, levels
end

function remove_node!(g::G, weights::Dict{Tuple{Int,Int},Float64}, node::Int, inneighbors::Vector{Int}, outneighbors::Vector{Int}, levels::Vector{Int}) where G <: AbstractGraph
	# This function removes a node from the graph and updates the weights dictionary accordingly.
	# We only remove edges because removing vertices messes up the indexing
	# Remove all incoming edges to the node
	if !isempty(inneighbors)
		for n in inneighbors
			levels[n] = max(levels[n], levels[node] + 1)
			rem_edge!(g, n, node)
			delete!(weights, (n, node))
		end
	end

	# Remove all outgoing edges from the node
	if !isempty(outneighbors)
		for m in outneighbors
			levels[m] = max(levels[m], levels[node] + 1)
			rem_edge!(g, node, m)
			delete!(weights, (node, m))
		end
	end
end

function recompute_queue(g::G, weights::Dict{Tuple{Int,Int},Float64}, processed::Vector{Bool}, n_contr_neighbors::Vector{Int}) where G <: AbstractGraph
	# This function recomputes the priority queue based on current edge differences.
	
	new_queue = PriorityQueue{Int, Int}()
	for node in 1:nv(g)
		if !processed[node]
			ed = edge_difference(g, weights, node)
			enqueue!(new_queue, node, cost(ed, n_contr_neighbors[node]))
		end
	end
	return new_queue
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
		push!(visited, u)

		# Early stopping if we exceed max_distance
		if dist_u > max_distance
			break
		end

		for v in outneighbors(g, u)
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
	# Weighted graphs use CSC matrix, so incremental edge addition is not possible.

	
	# Need to build array representation
	sources_up = Int[]
	targets_up = Int[]
	weights_up = Float64[]

	sources_down = Int[]
	targets_down = Int[]
	weights_down = Float64[]

	for e in edges(g_augmented)
		u = src(e)
		v = dst(e)
		weight = weights[(u, v)]
		if node_order[u] < node_order[v]
			push!(sources_up, u)
			push!(targets_up, v)
			push!(weights_up, weight)
		else
			# g_down is stored reversed for easier backward search (need access to incoming edges)
			push!(sources_down, v)
			push!(targets_down, u)
			push!(weights_down, weight)
		end
	end

	g_up = SimpleWeightedDiGraph(sources_up, targets_up, weights_up)
	g_down = SimpleWeightedDiGraph(sources_down, targets_down, weights_down)

	return g_up, g_down
end