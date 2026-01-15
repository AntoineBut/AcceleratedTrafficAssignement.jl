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


function benchmark_phast(G, W, CH, nsources=32)
    gpu_ch = to_device(CH, backend)

    sources = rand(1:nv(G), nsources)
    sources_ch = CH.inv_reordering[sources]
    if nsources <= 128 # Compute directly
        b_cpu = @benchmark shortest_path_CH($CH, $sources_ch);
        b_gpu = @benchmark shortest_path_CH($gpu_ch, $sources_ch);
    else
        # For large number of sources, batch them by 128
        b_cpu = @benchmark begin
            for i in 1:128:$nsources
                batch = $sources_ch[i:min(i+127, $nsources)]
                shortest_path_CH($CH, batch)
            end
        end
        b_gpu = @benchmark begin
            for i in 1:128:$nsources
                batch = $sources_ch[i:min(i+127, $nsources)]
                shortest_path_CH($gpu_ch, batch)
            end
        end
    end

    weighted__g = digraph_to_weightedgraph(G, W);
    storage = DijkstraHeapStorage(weighted__g)

    b_ref = @benchmark for (j, source) in enumerate($sources)
        custom_dijkstra!($storage, $weighted__g, source)
    end
    return (b_cpu, b_gpu, b_ref)
end

DATASETS = ["data/USA-road-t.NY.gr", "data/USA-road-t.BAY.gr"] 
names = ["NY", "BAY"]
#nsources_iter = [32]
nsources_iter = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# create result dataframe
# columns: info, name, nsources, time_cpu, time_gpu, time_ref
#df = DataFrame(info = String[], name = String[], nsources = Int[], time_cpu = Float64[], time_gpu = Float64[], time_ref = Float64[])
info = "baseline"
for (i, dataset) in enumerate(DATASETS)
    println("Benchmarking dataset: ", dataset)
    g_1, weights_1 = load_dimacs(dataset)

    order = reorder_vertices_dfs(g_1, 1);
    g_w, weights = permuted_graph(order, g_1, weights_1);
    @time CH = compute_CH(g_w, weights)
    @time recomputed_CH = compute_CH(CH.g, CH.weights; old_CH = CH);
    for n in nsources_iter
        println("Benchmarking nsources: ", n)
        b_cpu, b_gpu, b_ref = benchmark_phast(g_w, weights, CH, n)
        push!(df, (info, names[i], n, median(b_cpu.times) / 1e6, median(b_gpu.times) / 1e6, median(b_ref.times) / 1e6))
    end
end


