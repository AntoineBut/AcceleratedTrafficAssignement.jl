module AcceleratedTrafficAssignement

using Graphs
using SimpleWeightedGraphs
using DataStructures
using Base.Threads


include("utils.jl")
include("ContractionHierachies.jl")
include("Phast.jl")


export CHGraph,
    augment_graph,
    edge_difference,
    witness_search,
    compute_up_down_graphs,
    compute_CH,
    shortest_path_CH,
    reorder_vertices_dfs,
    permuted_graph
end
