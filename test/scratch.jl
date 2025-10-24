using Graphs, Random, SimpleWeightedGraphs, DataStructures
using AcceleratedTrafficAssignement, BenchmarkTools, SparseArrays
# Set random seed for reproducibility
Random.seed!(42)
g = grid((300, 300))
g_1 = SimpleDiGraph(nv(g))
weights_1 = Dict{Tuple{Int,Int},Float64}()
# Assign random weights to edges
for e in edges(g)
	u = src(e)
	v = dst(e)
	if rand() > 0.45
		weight = rand() + 0.1
		push!(weights_1, (u, v) => weight)
		add_edge!(g_1, u, v)
	end
	if rand() > 0.45
		weight = rand() + 0.1
		push!(weights_1, (v, u) => weight)
		add_edge!(g_1, v, u)
	end
end

#order = reorder_vertices_dfs(g_1, 1);

order = collect(1:nv(g_1));
g_w, weights = permuted_graph(order, g_1, weights_1);


function launch()
	println("###### Running CH ######")
	@time CH = compute_CH(g_w, weights)
	println("OG:$(ne(CH.g)) - UP:$(ne(CH.g_up)) - DOWN:$(ne(CH.g_down)) - AUG:$(ne(CH.g_augmented))");
end

function prof()
	@profview compute_CH(g_w, weights)
end


function bench()
	@benchmark compute_CH(g_w, weights)
end
@time CH = compute_CH(g_w, weights);
println("OG:$(ne(CH.g)) - UP:$(ne(CH.g_up)) - DOWN:$(ne(CH.g_down)) - AUG:$(ne(CH.g_augmented))");

distances = shortest_path_CH(CH, 1);

g_ref = CH.g;
weights_ref = CH.weights;
# Verify correctness with Dijkstra, requires a weight matrix (SparseMatrixCSC)
weights_matrix = convert(SparseMatrixCSC{Float64, Int64}, adjacency_matrix(g_ref));
for e in edges(g_ref)
	u = src(e)
	v = dst(e)
	weights_matrix[u, v] = weights_ref[(u, v)]
end

distances_ref = dijkstra_shortest_paths(g_ref, 1, weights_matrix).dists
println(isapprox(distances, distances_ref))
function verify_levels(CH::CHGraph)
	err = 0
	g_down = CH.g_down
	levels = CH.levels
	for e in edges(g_down)
		u = src(e)
		v = dst(e)
		if levels[u] <= levels[v]
			err += 1
		end
	end
	if err > 0
		error("Level verification failed in g_down: $err violations found out of $(ne(g_down)).")
	else
		println("All levels verified in g_down.")
	end
end
verify_levels(CH)
error("Stop here")
# Verify : If (u, v) ∈ g_down, then level(u) > level(v).
@benchmark distances = shortest_path_CH(CH, 1)
@benchmark distances_ref = dijkstra_shortest_paths(CH.g, 1, weights_matrix).dists


#weights_matrix_T = convert(SparseMatrixCSC{Float64, Int64}, adjacency_matrix(g_w, dir=:in));
#for e in edges(g_w)
#	u = src(e)
#	v = dst(e)
#	weights_matrix_T[v, u] = weights[(u, v)]
#end
#augmented_graph = augment_graph(g_w, weights)
#println(augmented_graph)


#println("Node order: $(CHW.node_order)");
#g_w2 = SimpleWeightedDiGraph(nv(g))
## Assign random weights to edges
#for e in edges(g_w)
#	u = src(e)
#	v = dst(e)
#	weight = weights[(u,v)]
#	add_edge!(g_w2, u, v, weight)
#end

#augmented_graph = augment_graph(g_w2)
#CH = compute_CH(g_w2);
#println("OG:$(ne(CH.g)) - UP:$(ne(CH.g_up)) - DOWN:$(ne(CH.g_down)) - AUG:$(ne(CH.g_augmented))");
#println("Node order: $(CH.node_order)");
