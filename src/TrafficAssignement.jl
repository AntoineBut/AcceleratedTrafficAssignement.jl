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
	link_ids::AbstractVector,
    demand::Dict{Tuple{Int,Int},T},
    origins::AbstractVector,
    destinations::AbstractVector,
	zone_nodes::AbstractVector,
) where {T<:Real}

	link_ids_dict = sparseCSC_to_dict(link_ids)
    # Storage for shortest paths
    shortest_paths_storage = PhastStorageCPU{T}(nv(ch.g), length(origins))

    # Step 1: Recompute CH (only update weights, not the ordering)
    update_CH_weights!(ch, cost_vec)

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

function update_CH_weights!(ch::AbstractCHGraph, cost_vec::AbstractVector)
    # Update the weights of the CH graph based on the new cost vector
    # TODO
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
	flow_vec_augmented = Dict{Tuple{Int,Int}, T}()

    # 1 : Assign flow on the augmented graph
    for (i, o) in enumerate(origins)
        for d in destinations
            if o != d && haskey(demand, (o, d))
				dem = demand[o, d]
                curr = d
				parent = storage.parents[curr, i]
                while curr != o
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
		flow_vec[link_ids_dict[edge1]] += flow_shortcut
		flow_vec[link_ids_dict[edge2]] += flow_shortcut
	end
end


function sparseCSC_to_dict(m::SparseMatrixCSC{T,Int}) where {T<:Real}
	d = Dict{Tuple{Int,Int},T}()
	sizehint!(d, nnz(m))
	for col in 1:size(m, 2)
		for row_ptr in m.colptr[col]:(m.colptr[col+1]-1)
			row = m.rowval[row_ptr]
			value = m.nzval[row_ptr]
			d[(row, col)] = value
		end
	end
	return d
end
