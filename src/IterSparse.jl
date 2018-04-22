struct BPRIterSparse <: AbstractBPRIter
    nusers::Integer
    nprods::Integer
    users::UnitRange{<:Integer}
    prods::Set
    pos_prods::AbstractArray{Set{<:Integer}, 1}
    pos_holdouts::AbstractArray{<:Integer, 1}
    neg_holdouts::AbstractArray{<:Integer, 1}
end

function BPRIterSparse(data::AbstractArray{T, 2}) where T
    gooddata = verifydata(data)
    nprods, nusers = size(gooddata)
    users = 1:nusers
    prods = Set(1:nprods)
    pos_prods = [Set(find(gooddata[:, user] .> 0)) for user in users]
    S = eltype(pos_prods[1])
    pos_holdouts = zeros(S, nusers)
    neg_holdouts = zeros(S, nusers)
    for user in users
        pitem = rand(pos_prods[user])
        pos_holdouts[user] = pop!(pos_prods[user], pitem)
        neg_holdouts[user] = rand(setdiff(prods, pos_prods[user]))
    end
    return BPRIterSparse(nusers, nprods, users, prods, pos_prods, pos_holdouts,
                         neg_holdouts)
end

function Base.next(B::BPRIterSparse, state)
    user = rand(B.users)
    pos_prod = @views rand(B.pos_prods[user])
    neg_prod = @views rand(setdiff(setdiff(B.prods, B.pos_prods[user]),
                                   Set(B.neg_holdouts[user])))
    return (user, pos_prod, neg_prod), nothing
end

Base.iteratoreltype(bpr::BPRIterSparse) = typeof((bpr.users[1],
                                                 zero(eltype(bpr.pos_prods[1])),
                                                 zero(eltype(bpr.pos_prods[1]))))

