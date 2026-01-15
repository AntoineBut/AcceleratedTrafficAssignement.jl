using Graphs, Random, SimpleWeightedGraphs, DataStructures
using AcceleratedTrafficAssignement, FasterShortestPaths
using SuiteSparseMatrixCollection, HarwellRutherfordBoeing, GraphIO.EdgeList
using BenchmarkTools, SparseArrays, GPUArrays, GPUGraphs, KernelAbstractions
using Metal

using DataFrames, CSV
using AcceleratedTrafficAssignement

#backend=CUDABackend()
backend = MetalBackend()
T = Float32
nsources = 32
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

function benchmark_ch(G, W)
    CH = compute_CH(G, W)
    b = @benchmark begin
        compute_CH($G, $W)
    end
    t1 = median(b.times)
    s = ne(CH.g_augmented)
    b2 = @benchmark begin
        compute_CH($CH.g, $CH.weights; old_CH = $CH)
    end 
    t2 = median(b2.times)
    return (t1, t2, s)
end

DATASETS = ["data/USA-road-t.NY.gr"]#, "data/USA-road-t.BAY.gr"] 
names = ["NY"] #, "BAY"]
df = DataFrame(info = String[], dataset = String[], augmented_size = Float64[],median_time = Float64[], median_time_recompute = Float64[])
dataset = DATASETS[1]
println("Benchmarking dataset: ", dataset)
g_1 = SimpleDiGraph(0)
weights_1 = Dict{Tuple{Int,Int},T}()
g_1, weights_1 = load_dimacs(dataset)
order = reorder_vertices_dfs(g_1, 1);
g_w, weights = permuted_graph(order, g_1, weights_1);
# create result dataframe
# columns: :info, :dataset, :augmented_size :median_time, :median_time_recompute
(t1, t2, s) = benchmark_ch(g_w, weights)
push!(df, ("Limited Lazy Updates",names[1],  s, t1 / 1e9, t2 / 1e9))

error("stop here")
for (i, dataset) in enumerate(DATASETS)
    println("Benchmarking dataset: ", dataset)
    g_1 = SimpleDiGraph(0)
    weights_1 = Dict{Tuple{Int,Int},T}()

    g_1, weights_1 = load_dimacs(dataset)

    order = reorder_vertices_dfs(g_1, 1);
    g_w, weights = permuted_graph(order, g_1, weights_1);
    # create result dataframe
    # columns: :info, :dataset, :augmented_size :median_time, :median_time_recompute
    (t1, t2, s) = benchmark_ch(g_w, weights)
    push!(df, ("4",names[i],  s, t1 / 1e9, t2 / 1e9))
end
# Save dataframe to CSV
#CSV.write("out/results_strategies2.csv", df)

#@benchmark recomputed_CH = compute_CH(CH.g, CH.weights; old_CH = CH);

#println("Recomputed CH:");
#rintln(
#    "Vertices:$(nv(recomputed_CH.g)) OG:$(ne(recomputed_CH.g)) - UP:$(ne(recomputed_CH.g_up)) - DOWN:$(ne(recomputed_CH.g_down_rev)) - AUG:$(ne(recomputed_CH.g_augmented))",
#);
#SIZES = [50, 75, 100, 150, 200, 300, 400, 600, 800]
# create result dataframe
# columns: :size, :median_time, :augmented_size, :median_time_recompute
df = DataFrame(size = Int[], median_time = Float64[], augmented_size = Float64[], median_time_recompute = Float64[])

#if DATA == false
#    for i in eachindex(SIZES)
#        println("Benchmarking size: ", SIZES[i])
#        n = SIZES[i]
#        # Generate a random grid graph
#        g = Graphs.grid((n, n))
#        g_1 = SimpleDiGraph(nv(g))
#        # Assign random weights to edges
#        for e in edges(g)
#            u = src(e)
#            v = dst(e)
#            if rand() > 0.45
#                weight = rand() + 5
#                push!(weights_1, (u, v) => weight)
#                add_edge!(g_1, u, v)
#            end
#            if rand() > 0.45
#                weight = rand() + 5
#                push!(weights_1, (v, u) => weight)
#                add_edge!(g_1, v, u)
#            end
#        end
#        order = reorder_vertices_dfs(g_1, 1);
#        g_w2, weights2 = permuted_graph(order, g_1, weights_1);
#       
#        (t1, t2, s) = benchmark_ch(g_w2, weights2)
#        #println(ne(g_w2), " -> ", ne(CH.g_augmented))
#        push!(df, (n*n, t1 / 1e9, s/ 1e6, t2 / 1e9))
#
#    end
#    CSV.write("out/results_synthetic2.csv", df)
#end