module AcceleratedTrafficAssignement

using Graphs
using SimpleWeightedGraphs
using DataStructures

include("ContractionHierachies.jl")

export CHGraph,
augment_graph, 
edge_difference, 
witness_search,
compute_up_down_graphs,
compute_CH
end
