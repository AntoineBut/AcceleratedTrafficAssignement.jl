# tests for the CH pre-processing



function test_graph_contractions(CH::CHGraph)
    g = CH.g
    @test length(CH.node_order) == nv(g)
    @test length(CH.reordering) == nv(g)
    @test length(CH.levels) == nv(g)

    # Thest that the weigh dicts have ne(g) entries
    @test length(CH.weights) == ne(g)
    @test length(CH.weights_augmented) == ne(CH.g_augmented)

    # Edges are either in upward or downward graph
    total_edges = ne(CH.g_up) + ne(CH.g_down_rev)
    @test total_edges == ne(CH.g_augmented)


    @test length(CH.levels) == nv(g)
    # Check that upward and downward graphs only have edges going up and down respectively
    @test verify_ordering(CH) == (true, true)
    # Check that levels are consistent
    @test verify_levels(CH)

    # Check that ordering and reordering are consistent
    @test length(unique(CH.node_order)) == nv(g)
    @test length(unique(CH.reordering)) == nv(g)

    # Check that the nodes are indeed re-ordered by levels
    sorted_levels = sort(CH.levels, rev = true)
    @test sorted_levels == CH.levels

end


function verify_ordering(CH::CHGraph)
    err_up = 0
    err_down = 0
    g_up = CH.g_up
    g_down_rev = CH.g_down_rev
    levels = CH.levels
    order = CH.node_order
    # Edges in g_up should go from lower to higher order
    for e in edges(g_up)
        u = src(e)
        v = dst(e)

        if order[u] >= order[v]
            err_up += 1
        end
    end

    # Edges in g_down_rev should go from higher to lower order
    for e in edges(g_down_rev)
        # the edge (u, v) in g_down_rev is stored reversed
        v = src(e)
        u = dst(e)

        if order[u] <= order[v]
            err_down += 1
        end
    end
    return err_up == 0, err_down == 0
end

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

        end
    end
    return err == 0
end
