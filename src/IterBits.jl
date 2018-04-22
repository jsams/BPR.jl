# does NOT support offset bitsets
"randomly sample from the complement of the set over the range of range"
function rand_compl(b::BitSet, range::UnitRange)
    while true
        n = rand(range)
        !(n in b) && return n
    end
end

function rand_compl(b::BitSet, range::UnitRange, N::Integer)
    return [rand_compl(b, range) for _ in 1:N]
end

struct BPRIterBits <: AbstractBPRIter
    nusers::Integer
    nprods::Integer
    users::UnitRange{<:Integer}
    prods::BitSet
    pos_prods::AbstractArray{BitSet, 1}
    pos_holdouts::AbstractArray{<:Integer, 1}
    neg_holdouts::AbstractArray{<:Integer, 1}
end

function BPRIterBits(data::AbstractArray{T, 2}) where T
    gooddata = verifydata(data)
    nprods, nusers = size(gooddata)
    users = 1:nusers
    prods = BitSet(1:nprods)
    pos_prods = [BitSet(find(gooddata[:, user] .> 0)) for user in users]
    pos_holdouts = [rand(pos_prods[u]) for u in users]
    neg_holdouts = [rand_compl(pos_prods[u], 1:nprods) for u in users]
    return BPRIterBits(nusers, nprods, users, prods, pos_prods, pos_holdouts,
                       neg_holdouts)
end

function BPRIterBits(B::BPRIterBits)
    pos_holdouts = [rand(B.pos_prods[u]) for u in B.users]
    neg_holdouts = [rand_compl(B.pos_prods[u], 1:B.nprods) for u in B.users]
    return BPRIterBits(B.nusers, B.nprods, copy(B.users), copy(B.prods),
                       deepcopy(B.pos_prods), pos_holdouts, neg_holdouts)
end

@inline function nextpos(user::Integer, B::BPRIterBits)
    while true
        pp = @views rand(B.pos_prods[user])
        pp != B.pos_holdouts[user] && return pp
    end
end

@inline function nextneg(user::Integer, B::BPRIterBits)
    while true
        np = @views rand_compl(B.pos_prods[user], 1:B.nprods)
        np != B.neg_holdouts[user] && return np
    end
end

@inline function draw_posneg(user::Integer, B::BPRIterBits)
    return (nextpos(user, B), nextneg(user, B))
end

@inline function draw_holdout(user::Integer, B::BPRIterBits)
    return (B.pos_holdouts[user], B.neg_holdouts[user])
end

