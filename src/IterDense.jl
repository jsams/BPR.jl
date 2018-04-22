struct BPRIterDense <: AbstractBPRIter
    nusers::Integer
    nprods::Integer
    users::UnitRange{<:Integer}
    prods::IntSet
    pos_prods::AbstractArray{<:AbstractArray{<:Integer, 1}, 1}
    neg_prods::AbstractArray{<:AbstractArray{<:Integer, 1}, 1}
    pos_holdouts::AbstractArray{<:Integer, 1}
    neg_holdouts::AbstractArray{<:Integer, 1}
end

"""
    data is a item x user array, result is an infinite iterator over Ds

    where the non-zero entries indicate that the user has
    consumed/purchased/rated that item. Because we compute out-of-sample AUC,
    each user needs at least 2 rated items (1 to train on, 1 to hold out).
    There should be no unconsumed items. Also, no user should have consumed
    every item, but this should not be an issue in real-world data.

    Sparse matrix is supported.

    See Rendle 2009 paper for definition of the uniform sampling strategy."""
function BPRIterDense(data::AbstractArray{T, 2}) where T
    # do any users see only one item? filter out b/c can't do hold out with
    # them
    gooddata = verifydata(data)
    nprods, nusers = size(gooddata)
    users = 1:nusers
    prods = IntSet(1:nprods)
    pos_prods = [find(gooddata[:, user] .> 0) for user in users]
    neg_prods = [collect(setdiff(prods, posprod)) for posprod in pos_prods]
    pos_holdouts = zeros(eltype(pos_prods[1]), nusers)
    neg_holdouts = zeros(eltype(neg_prods[1]), nusers)
    for user in users
        pitemidx = rand(1:size(pos_prods[user], 1))
        nitemidx = rand(1:size(neg_prods[user], 1))
        pos_holdouts[user] = pos_prods[user][pitemidx]
        neg_holdouts[user] = neg_prods[user][nitemidx]
        deleteat!(pos_prods[user], pitemidx)
        deleteat!(neg_prods[user], nitemidx)
    end
    return BPRIterDense(nusers, nprods, users, prods, pos_prods, neg_prods,
                        pos_holdouts, neg_holdouts)
end

function BPRIterDense(B::BPRIterDense)
    pos_prods = deepcopy(B.pos_prods)
    neg_prods = deepcopy(B.neg_prods)
    pos_holdouts = deepcopy(B.pos_holdouts)
    neg_holdouts = deepcopy(B.neg_holdouts)
    for user in B.users
        Np = size(pos_prods[user], 1) + 1
        pidx = rand(1:Np)
        if pidx != Np
            pos_prods[user][pidx], pos_holdouts[user] = pos_holdouts[user], pos_prods[user][pidx]
        end
        Nn = size(neg_prods[user], 1) + 1
        nidx = rand(1:Nn)
        if nidx != Nn
            neg_prods[user][nidx], neg_holdouts[user] = neg_holdouts[user], neg_prods[user][nidx]
        end
    end
    return BPRIterDense(B.nusers, B.nprods, copy(B.users), copy(B.prods),
                        pos_prods, neg_prods, pos_holdouts, neg_holdouts)
end

@inline function draw_posneg(user::Integer, B::BPRIterDense)
    return (rand(B.pos_prods[user]), rand(B.neg_prods[user]))
end

@inline function draw_holdout(user::Integer, B::BPRIterDense)
    return (B.pos_holdouts[user], B.neg_holdouts[user])
end

