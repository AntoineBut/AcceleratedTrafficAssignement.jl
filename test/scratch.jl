using Graphs, Random, SimpleWeightedGraphs, DataStructures, AcceleratedTrafficAssignement, BenchmarkTools
# Set random seed for reproducibility
Random.seed!(42)
g = grid((200, 500))
g_w = SimpleDiGraph(nv(g))
weights = Dict{Tuple{Int,Int},Float64}()
# Assign random weights to edges
for e in edges(g)
	u = src(e)
	v = dst(e)
	if rand() > 0.45
		weight = rand() + 0.1
		push!(weights, (u, v) => weight)
		add_edge!(g_w, u, v)
	end
	if rand() > 0.45
		weight = rand() + 0.1
		push!(weights, (v, u) => weight)
		add_edge!(g_w, v, u)
	end
end

@time CH = compute_CH(g_w, weights);
println("OG:$(ne(CH.g)) - UP:$(ne(CH.g_up)) - DOWN:$(ne(CH.g_down)) - AUG:$(ne(CH.g_augmented))");

function run()
	println("###### Running CH ######")
	CH = compute_CH(g_w, weights)
	println("OG:$(ne(CH.g)) - UP:$(ne(CH.g_up)) - DOWN:$(ne(CH.g_down)) - AUG:$(ne(CH.g_augmented))");

	#println("###### Running CH ######")
	#CH = compute_CH(g_w2)
	#println("OG:$(ne(CH.g)) - UP:$(ne(CH.g_up)) - DOWN:$(ne(CH.g_down)) - AUG:$(ne(CH.g_augmented))");
end

function prof()
	@profview compute_CH(g_w, weights)
end


function bench()
	@benchmark compute_CH(g_w, weights)
end



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
