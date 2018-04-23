module TestBPR

    using Base.Test
    using DataFrames
    #include("BPR.jl")
    import BPR

    # this doesn't actually properly validate the interface
    #function validate_biter(B::BPRIter)
    #    ret = true
    #    for user in B.users
    #        if B.nprods != (length(B.pos_prods[user]) + length(B.neg_prods[user]) + 2)
    #            warn("User $(user) has wrong number of products.")
    #            ret = false
    #        elseif B.pos_holdouts[user] ∈ B.pos_prods[user]
    #            warn("User $(user) has holdout in positive prods.")
    #            ret = false
    #        elseif B.neg_holdouts[user] ∈ B.neg_prods[user]
    #            warn("User $(user) has holdout in negative prods.")
    #            ret = false
    #        elseif B.prods != IntSet([B.pos_prods[user]; B.neg_prods[user];
    #                              B.pos_holdouts[user]; B.neg_holdouts[user]])
    #            warn("User $(user) is missing products.")
    #            ret = false
    #        end
    #    end
    #    return ret
    #end

    X = sprand(300, 400, 0.1)
    biter = BPR.BPRIter(X)
    @test isa(biter, BPR.BPRIterBits)
    @test isa(BPR.bpr(biter, 10, 0.01, 0.01, 0.01, 0.01; tol=0.01,
                      max_iters=10, min_iters=3),
              BPR.BPRResult)
    df = BPR.grid_search(biter; ks=[1], loop_size=16,  max_iters=2)
    @test isa(df, DataFrame)

    #@test validate_biter(biter)
    #@test validate_biter(biter2)

end #module
