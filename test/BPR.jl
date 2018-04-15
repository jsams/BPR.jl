module TestBPR

    using Base.Test
    using BPR

    function validate_biter(B::BPR_iter)
        ret = true
        for user in B.users
            if B.nprods != (length(B.pos_prods[user]) + length(B.neg_prods[user]) + 2)
                warn("User $(user) has wrong number of products.")
                ret = false
            elseif B.pos_holdouts[user] ∈ B.pos_prods[user]
                warn("User $(user) has holdout in positive prods.")
                ret = false
            elseif B.neg_holdouts[user] ∈ B.neg_prods[user]
                warn("User $(user) has holdout in negative prods.")
                ret = false
            elseif B.prods != IntSet([B.pos_prods[user]; B.neg_prods[user];
                                  B.pos_holdouts[user]; B.neg_holdouts[user]])
                warn("User $(user) is missing products.")
                ret = false
            end
        end
        return ret
    end

    X = sprand(300, 400, 0.1)
    biter = BPR_iter(X)
    biter2 = BPR_iter(biter)

    @test validate_biter(biter)
    @test validate_biter(biter2)

end #module
