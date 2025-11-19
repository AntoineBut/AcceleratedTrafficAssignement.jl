using Graphs, Random, SimpleWeightedGraphs, DataStructures
using AcceleratedTrafficAssignement, FasterShortestPaths
using SuiteSparseMatrixCollection, HarwellRutherfordBoeing, GraphIO.EdgeList
using BenchmarkTools, SparseArrays, GPUArrays, GPUGraphs, KernelAbstractions
using Metal

#backend=CUDABackend()
backend = MetalBackend()
T = Float32
nsources = 128
Random.seed!(42)
function load_dimacs(path::String)
    g = SimpleDiGraph(0)
    weights = Dict{Tuple{Int,Int},T}()
    open(path, "r") do io
        for line in eachline(io)
            if startswith(line, "p")
                parts = split(line)
                n = parse(Int, parts[3])
                m = parse(Int, parts[4])
                g = SimpleDiGraph(n)
            elseif startswith(line, "a")
                parts = split(line)
                u = parse(Int, parts[2])
                v = parse(Int, parts[3])
                weight = parse(T, parts[4])
                weights[(u, v)] = weight
                add_edge!(g, u, v)

            end
        end
    end
    return g, weights
end
DATA = true
g_1 = SimpleDiGraph(0)
weights_1 = Dict{Tuple{Int,Int},T}()
if DATA
    #path = "data/USA-road-t.NY.gr"
    #g_1 = loadgraph(path, "ca", EdgeListFormat())
    #self_loops = []
    ## Assign weights to edges
    #for e in edges(g_1)
    #    u = src(e)
    #    v = dst(e)
    #    if u == v
    #        push!(self_loops, e)
    #        continue
    #    end
    #    weights_1[(u, v)] = 1.0 #+ rand() * 9.0 
    #end
    #for e in self_loops
    #    rem_edge!(g_1, e)
    #end
    #path = "data/USA-road-t.W.gr"
    path = "data/USA-road-t.COL.gr"
    #path = "data/USA-road-t.NY.gr"
    g_1, weights_1 = load_dimacs(path)

else
    # Set random seed for reproducibility
    g = grid((100, 100))
    g_1 = SimpleDiGraph(nv(g))
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
end
order = reorder_vertices_dfs(g_1, 1);
g_w, weights = permuted_graph(order, g_1, weights_1);

#g_w, weights = g_1, weights_1;

function digraph_to_weightedgraph(
    g::SimpleDiGraph,
    weights::Dict{Tuple{Int,Int},T},
) where {T}
    g_w = SimpleWeightedDiGraph(nv(g))
    sources = zeros(Int, ne(g))
    destinations = zeros(Int, ne(g))
    edge_weights = zeros(T, ne(g))
    for (i, e) in enumerate(edges(g))
        u = src(e)
        v = dst(e)
        weight = weights[(u, v)]
        sources[i] = u
        destinations[i] = v
        edge_weights[i] = weight
    end
    g_w = SimpleWeightedDiGraph(sources, destinations, edge_weights)
    return g_w
end

function launch()
    println("###### Running CH ######")
    @time CH = compute_CH(g_w, weights)
    println(
        "Vertices:$(nv(g_w)) OG:$(ne(CH.g)) - UP:$(ne(CH.g_up)) - DOWN:$(ne(CH.g_down_rev)) - AUG:$(ne(CH.g_augmented))",
    );
end

function prof()
    #profview compute_CH(g_w, weights)
    @profview for _ = 1:100
        distances = shortest_path_CH(CH, start);
    end
end


function bench()
    @benchmark compute_CH(g_w, weights)
end
@time CH = compute_CH(g_w, weights);
gpu_ch = to_device(CH, backend);

#@profview CH = compute_CH(g_w, weights);®
#error("Stop here")
println(
    "Vertices:$(nv(g_w)) OG:$(ne(CH.g)) - UP:$(ne(CH.g_up)) - DOWN:$(ne(CH.g_down_rev)) - AUG:$(ne(CH.g_augmented))",
);

#@profview for _ in 1:10
#	distances = shortest_path_CH(CH, 1);
#end
sources = rand(1:nv(g_w), nsources)
sources_ch = CH.reordering[sources]
@time distances = shortest_path_CH(CH, sources_ch).distances;
@time distances_gpu = shortest_path_CH(gpu_ch, sources_ch).device_distances;
diff = abs.(distances .- collect(distances_gpu)) .> 1e-6
println(isapprox(distances, collect(distances_gpu)))

g_ref = g_w;
weights_ref = weights;
weighted__g = digraph_to_weightedgraph(g_ref, weights_ref);
# Verify correctness with Dijkstra, requires a weight matrix (SparseMatrixCSC)
weights_matrix = convert(SparseMatrixCSC{T,Int64}, adjacency_matrix(g_ref));
for e in edges(g_ref)
    u = src(e)
    v = dst(e)
    weights_matrix[u, v] = weights_ref[(u, v)]
end
distances_ref = zeros(T, nv(g_w), nsources);
@time for (j, source) in enumerate(sources)
    distances_ref[:, j] = dijkstra_shortest_paths(g_ref, source, weights_matrix).dists
end

storage = DijkstraHeapStorage(weighted__g)
distances_ref2 = zeros(T, nv(g_w), nsources);
@time for (j, source) in enumerate(sources)
    custom_dijkstra!(storage, weighted__g, source)
    distances_ref2[:, j] = storage.dists
end

println(isapprox(distances_ref, distances_ref2))

println(isapprox(distances[CH.reordering, :], distances_ref))
println(isapprox(distances[CH.reordering, :], distances_ref2))
println(isapprox(collect(distances_gpu[CH.reordering, :]), distances_ref))
println(isapprox(collect(distances_gpu[CH.reordering, :]), distances_ref2))
function verify_levels(CH::CHGraph)
    err = 0
    g_down_rev = CH.g_down_rev
    levels = CH.levels
    for e in edges(g_down_rev)
        # the edge (u, v) in g_down_rev is stored reversed
        v = src(e)
        u = dst(e)

        if levels[u] <= levels[v]
            err += 1
            println(
                "Level violation: level($u) = $(levels[u]) <= level($v) = $©(levels[v])",
            )
        end
    end
    if err > 0
        println(
            "Level verification failed in g_down: $err violations found out of $(ne(g_down_rev)).",
        )
    else
        println("All levels verified in g_down.")
    end
end
verify_levels(CH)
# Verify : If (u, v) ∈ g_down, then level(u) > level(v).

storage_cpu = PhastStorageCPU(T, nv(g_w), nsources)
storage_gpu = PhastStorageGPU(backend, T, nv(g_w), nsources)

t1 = @benchmark for source in sources
    custom_dijkstra!(storage, weighted__g, source)
end
display(t1)
t2 = @benchmark shortest_path_CH!(CH, sources, storage_cpu)
display(t2)
t3 = @benchmark shortest_path_CH!(gpu_ch, sources, storage_gpu)
display(t3)
println("\n ### Speedup-cpu: $(median(t1.times) ./ median(t2.times))x ### \n")
println("\n ### Speedup-gpu: $(median(t1.times) ./ median(t3.times))x ### \n")

error("Stop here")
@profview for _ = 1:100
    shortest_path_CH!(gpu_ch, sources_ch, storage_gpu);
end
#weights_matrix_T = convert(SparseMatrixCSC{T, Int64}, adjacency_matrix(g_w, dir=:in));
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
