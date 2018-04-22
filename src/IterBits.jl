# does NOT support offset bitsets

"""
    for 0.6 compatability, replace instances of IntSet with BitSet in 0.7+
    calls to this random() function should then be replace by rand() and this
    deleted.

    looks like this is MUCH faster if you specify range for the function"""
function random(b::IntSet; range::UnitRange=UnitRange(extrema(b)...))
    while true
        n = rand(range)
        n in b && return n
    end
end


"randomly sample from the complement of the set over the range of range"
function rand_compl(b::IntSet, range::UnitRange)
    while true
        n = rand(range)
        !(n in b) && return n
    end
end

function rand_compl(b::IntSet, range::UnitRange, N::Integer)
    return [rand_compl(b, range) for _ in 1:N]
end

struct BPRIterBits <: AbstractBPRIter
    nusers::Integer
    nprods::Integer
    users::UnitRange{<:Integer}
    prods::IntSet
    pos_prods::AbstractArray{IntSet, 1}
    pos_holdouts::AbstractArray{<:Integer, 1}
    neg_holdouts::AbstractArray{<:Integer, 1}
end

function BPRIterBits(data::AbstractArray{T, 2}) where T
    gooddata = verifydata(data)
    nprods, nusers = size(gooddata)
    users = 1:nusers
    prods = IntSet(1:nprods)
    pos_prods = [IntSet(find(gooddata[:, user] .> 0)) for user in users]
    pos_holdouts = [random(pos_prods[u]; range=1:nprods) for u in users]
    neg_holdouts = [rand_compl(pos_prods[u], 1:nprods) for u in users]
    return BPRIterBits(nusers, nprods, users, prods, pos_prods, pos_holdouts,
                       neg_holdouts)
end

function BPRIterBits(B::BPRIterBits)
    pos_holdouts = [random(B.pos_prods[u]; range=1:B.nprods) for u in B.users]
    neg_holdouts = [rand_compl(B.pos_prods[u], 1:B.nprods) for u in B.users]
    return BPRIterBits(B.nusers, B.nprods, copy(B.users), copy(B.prods),
                       deepcopy(B.pos_prods), pos_holdouts, neg_holdouts)
end

@inline function nextpos(user::Integer, B::BPRIterBits)
    while true
        pp = @views random(B.pos_prods[user]; range=1:B.nprods)
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

