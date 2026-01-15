# This file contains the code to perform traffic assignment using the User Equilibrium (UE) model.

"""
	Performs an iteration of traffic assignment using the User Equilibrium (UE) model.

	Parameters:
	- ch: An AbstractCHGraph object representing the contraction hierarchy of the transportation network.
	- cost_vec: A vector representing the current travel costs on each link.
	- od_matrix: A (k x 2) matrix representing the origin-destination pairs.
	- flow_vec: A vector representing the flow on each link in the network, to be updated.


"""
function assign_flow!(
    flow_vec::AbstractVector,
    ch::AbstractCHGraph,
    cost_vec::AbstractVector,
    link_ids::SparseMatrixCSC{Int,Int},
    demand::Dict{Tuple{Int,Int},T},
    origins::AbstractVector,
    destinations::AbstractVector,
    zone_nodes::UnitRange{Int},
) where {T<:Real}

    inv_reordering = ch.inv_reordering
    # Map origins and destinations to the reordered graph
    origins = inv_reordering[origins]
    destinations = inv_reordering[destinations]

    link_ids_dict = sparseCSC_to_dict(link_ids; inv_reordering = inv_reordering)
    # Storage for shortest paths
    shortest_paths_storage = PhastStorageCPU(T, nv(ch.g), length(origins))

    # Build weights dictionary
    cost_dict = Dict{Tuple{Int,Int},T}()
    for e in edges(ch.g)
        u = src(e)
        v = dst(e)
        cost_dict[(u, v)] = cost_vec[link_ids_dict[(u, v)]]
    end

    # Step 1: Recompute CH (only update weights, not the ordering)
    ch = compute_CH(
        ch.g,
        cost_dict;
        old_CH = ch,
    )

    # Step 2: Compute shortest paths based on current travel times
    shortest_path_CH!(ch, origins, shortest_paths_storage)

    # Step 3: Assign flows based on shortest paths
    unpack_paths!(
        ch,
        demand,
        origins,
        destinations,
        shortest_paths_storage,
        link_ids_dict,
        flow_vec,
    )
end

function unpack_paths!(
    ch::AbstractCHGraph,
    demand::Dict{Tuple{Int,Int},T},
    origins::AbstractVector,
    destinations::AbstractVector,
    storage::PhastStorageCPU,
    link_ids_dict::Dict{Tuple{Int,Int},Int},
    flow_vec::AbstractVector,
) where {T<:Real}
    # Unpack the shortest paths and assign flows to the original graph
    flow_vec_augmented = Dict{Tuple{Int,Int},T}()

    # 1 : Assign flow on the augmented graph
    for (i, o) in enumerate(origins)
        for d in destinations
            if o != d && haskey(demand, (o, d))
                dem = demand[o, d]
                curr = d
                parent = storage.parents[curr, i]
                while curr != o && parent != -1 # -1 indicates no parent
                    edge = (parent, curr)
                    flow_vec_augmented[edge] = dem + get(flow_vec_augmented, edge, zero(T))
                    curr = parent
                    parent = storage.parents[curr, i]
                end
            end
        end
    end
    # 2 : Map flows back to the original graph. The ordering of the shortucts is important here.
    for (u, v, skipped) in reverse(ch.shortcuts)
        edge_shortcut = (u, v)
        edge1 = (u, skipped)
        edge2 = (skipped, v)
        flow_shortcut = get(flow_vec_augmented, edge_shortcut, zero(T))
        link_id1 = link_ids_dict[edge1]
        link_id2 = link_ids_dict[edge2]
      
        flow_vec[link_id1] += flow_shortcut
        flow_vec[link_id2] += flow_shortcut
    end
end


function sparseCSC_to_dict(
    m::SparseMatrixCSC{T,Int};
    inv_reordering::AbstractVector{Int} = 1:size(m, 1),
) where {T<:Real}
    reordering = invperm(inv_reordering)
    d = Dict{Tuple{Int,Int},T}()
    sizehint!(d, nnz(m))
    for col = 1:size(m, 2)
        for row_ptr = m.colptr[col]:(m.colptr[col+1]-1)
            row = m.rowval[row_ptr]
            value = m.nzval[row_ptr]
            d[(reordering[row], reordering[col])] = value
        end
    end
    return d
end

