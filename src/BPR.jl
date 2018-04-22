module BPR

import Base
using DataFrames
using ProgressMeter

export BPRResult, BPRIter, BPRIterDense, BPRIterSparse, bpr, auc_insamp,
       auc_outsamp, auc_outsamp2, grid_search


# e.g. to allow change to tol for new run
mutable struct BPRResult
    # convergence properties
    converged::Bool
    value::Real
    bpr_opt::Real
    auc_insample::Real
    auc_outsample::Real
    auc_outsample2::Real
    iters::Integer
    # run settings
    k::Integer
    λw::Real
    λhp::Real
    λhn::Real
    α::Real
    tol::Real
    max_iters::Integer
    min_iters::Integer
    min_auc::Real
    # results
    W::AbstractArray{<:Real, 2}
    H::AbstractArray{<:Real, 2}
end

function Base.string(B::BPRResult)
    """
        BPRResult:
          k:              $(size(B.W, 2))
          converged:      $(B.converged)
          value:          $(B.value)
          bpr_opt:        $(B.bpr_opt)
          auc_insample:   $(B.auc_insample)
          auc_outsample:  $(B.auc_outsample)
          auc_outsample2:  $(B.auc_outsample2)
          iters:          $(B.iters)"""
end

Base.show(io::Base.IO, B::BPRResult) = print(io, string(B))

# needs nusers, nprods, and next() / iterator protocol

# data properties in convenient format with iterator protocol
abstract type  AbstractBPRIter end

Base.string(bpr::AbstractBPRIter) = "$(bpr.nusers) x $(bpr.nprods) $(typeof(bpr))"
Base.show(io::Base.IO, bpr::AbstractBPRIter) = print(io, string(bpr))

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
    data is a user x item array, result is an infinite iterator over Ds

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
    good_users = find(x -> (x>1),
                      sum(x -> x > 0, data, 2))
    if size(good_users, 1) < size(data, 1)
        warn("$(size(data, 1) - size(good_users, 1)) users were removed for insufficient data.")
        data = data[good_users, :]
    end
    if any(sum(data, 1) .== 0)
        error("some columns sum to 0. Fix your data. bailing.")
    end
    nusers, nprods = size(data)
    users = 1:nusers
    prods = IntSet(1:nprods)
    pos_prods = [find(data[user, :] .> 0) for user in users]
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

Base.start(bpr::AbstractBPRIter) = nothing

function Base.next(bpr::BPRIterDense, state)
    user = rand(bpr.users)
    # views don't work here, and i'm not sure that they are necessary
    pos_prod = rand(bpr.pos_prods[user])
    neg_prod = rand(bpr.neg_prods[user])
    return (user, pos_prod, neg_prod), nothing
end

Base.done(bpr::AbstractBPRIter, state) = false

Base.iteratorsize(bpr::AbstractBPRIter) = Base.IsInfinite()

Base.iteratoreltype(bpr::BPRIterDense) = typeof((bpr.users[1], bpr.pos_prods[1][1],
                                             bpr.neg_prods[1][1]))


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
    good_users = find(x -> (x>1),
                      sum(x -> x > 0, data, 2))
    if size(good_users, 1) < size(data, 1)
        warn("$(size(data, 1) - size(good_users, 1)) users were removed for insufficient data.")
        data = data[good_users, :]
    end
    if any(sum(data, 1) .== 0)
        error("some columns sum to 0. Fix your data. bailing.")
    end
    nusers, nprods = size(data)
    users = 1:nusers
    prods = Set(1:nprods)
    pos_prods = [Set(find(data[user, :] .> 0)) for user in users]
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

# the defaults for our best guess of the most performant type
BPRIter(data::AbstractArray) = BPRIterDense(data)
BPRIter(B::BPRIterDense) = BPRIterDense(B)


"""
    find optimal W and H matrix for bpr matrix factorization, returns BPRResult
    * biter: object from AbstractBPRIter
    * k: number of dimensions to learn
    * λw: regularization on user features, 
    * λhp: regularization on positive updates on items
    * λhm: regularization on negative updates on items
    * α: learning rate
    * tol: when objective (BPR-OPT) has changed less than tol on average
           over loop_size iterations, consider converged
    * loop_size: how many loops to average change in in-sample AUC and BPR-OPT
    * max_iters: how many iterations over loop_size to try before giving up.
      0=never give up
    * min_iters: run at least this many iterations before allowing convergence.
    * min_auc: a secondary convergence criterion, in-sample AUC has to be at
      least this good before being considered converged
    * W: an initialized W parameter matrix (nuser x k)
    * H: an initialized H parameter matrix (nprod x k)"""
function bpr(biter::AbstractBPRIter, k, λw, λhp, λhn, α;
             tol=1e-5, loop_size=4096, max_iters=0, min_iters=1, min_auc=0.0,
             W=randn(biter.nusers, k), H=randn(biter.nprods, k))
    if biter.nusers < k | biter.nprods < k
        error("Number of rows and columns must both be greater than k, $k")
    end
    # ensure these are defined outside of while scope
    iters = 0 # need to loop at least twice to see the improvement in bpr-opt
    converged = false
    stepsize = 1.0
    bpr_old = 2.0 # make sure first loop doesn't look like convergence
    bpr_new = 0.0
    cur_auc = 0.0
    progress = ProgressThresh(tol, "BPR-Opt:")
    if min_auc > 0
        auc_progress = ProgressThresh(1-min_auc, "AUC:")
    end
    info("Starting BPR loop.")
    while true
        bpr_new = 0.0
        cur_auc = 0.0
        @inbounds for _ in 1:loop_size # simd be might be bad with random next()
        #@inbounds @simd for _ in 1:loop_size # simd be might be bad with random next()
            (user, pos_prod, neg_prod), _ = next(biter, nothing) # expensive: 222
            wuf = @view(W[user, :])
            hif = @view(H[pos_prod, :])
            hjf = @view(H[neg_prod, :])
            xuij = wuf' * hif - wuf' * hjf
            exuij = exp(-xuij)
            sig = exuij / (1 + exuij)
            wuf[:] .= wuf .+ α .* (sig .* (hif .- hjf) .+ λw .* wuf) # wuf_grad = hif - hjf
            hif[:] .= hif .+ α .* (sig .* wuf .+ λhp .* hif) # hif_grad = wuf
            hjf[:] .= hjf .+ α .* (sig .* -wuf .+ λhn .* hjf) # hjf_grad = -wuf
            xuij = wuf' * hif - wuf' * hjf
            # the bpr criterion
            bpr_new += (log(1 / (1 + exp(-xuij))) - λw * norm(wuf) -
                        λhp * norm(hif) - λhn * norm(hjf))
            # auc
            cur_auc += (xuij > 0)
        end
        iters += 1
        cur_auc /= loop_size
        bpr_new /= loop_size
        stepsize = abs(bpr_new - bpr_old)
        bpr_old = bpr_new
        ProgressMeter.update!(progress, stepsize)
        if min_auc > 0
            ProgressMeter.update!(auc_progress, cur_auc)
        end
        if (iters > min_iters) & (stepsize < tol) & (cur_auc > min_auc)
            converged = true
            break
        elseif (max_iters > 0) & (iters >= max_iters)
            ProgressMeter.cancel(progress, "max iters reached")
            if min_auc > 0
                ProgressMeter.cancel(auc_progress, "max iters reached")
            end
            warn("max iters reached without convergence, breaking.")
            break
        elseif isnan(stepsize)
            ProgressMeter.cancel(progress, "NaNs found")
            if min_auc > 0
                ProgressMeter.cancel(auc_progress, "NaNs found")
            end
            warn("stepsize is nan, breaking.")
            break
        end
    end
    auc_oos = auc_outsamp(biter, W, H)
    auc_oos2 = auc_outsamp2(biter, W, H)
    return BPRResult(converged, stepsize, bpr_new, cur_auc, auc_oos, auc_oos2,
                     iters, k, λw, λhp, λhn, α , tol, max_iters, min_iters,
                     min_auc, W, H)
end

function bpr(data::AbstractArray{<:Real, 2}, k, λw, λhp, λhn, α; 
             tol=1e-5, loop_size=4096, max_iters=0, min_iters=1, min_auc=0.0,
             W=randn(size(data, 1), k), H=randn(size(data, 2), k))
    nusers, nprods = size(data)
    if nusers < k | nprods < k
        error("Number of rows and columns must both be greater than k, $k")
    end
    # quick lookups for each user of consumed and unconsumed items
    info("Initializing iterator")
    biter = BPRIter(data)
    return bpr(biter, k, λw, λhp, λhn, α;
               tol=tol, loop_size=loop_size, max_iters=max_iters,
               min_iters=min_iters, min_auc=min_auc, W=W, H=H)
end

function bpr(data::AbstractArray{<:Real, 2}, B::BPRResult)
    bpr(data, B.k, B.λw, B.λhp, B.λhn, B.α; 
        tol=B.tol, max_iters=B.max_iters, min_iters=B.min_iters,
        min_auc=B.min_auc, W=copy(B.W), H=copy(B.H))
end

function bpr(biter::AbstractBPRIter, B::BPRResult)
    bpr(biter, B.k, B.λw, B.λhp, B.λhn, B.α;
        tol=B.tol, max_iters=B.max_iters, min_iters=B.min_iters, min_auc=B.min_auc,
        W=copy(B.W), H=copy(B.H))
end

# AUC computed on random in-sample
function auc_insamp(biter::AbstractBPRIter, W::AbstractArray{<:Real, 2},
                    H::AbstractArray{<:Real, 2}; iters=4096)
    sm = 0
    @inbounds @simd for _ in 1:iters
        (user, pos_prod, neg_prod), _ = next(biter, nothing) # expensive: 222
        wuf = @view(W[user, :])
        hif = @view(H[pos_prod, :])
        hjf = @view(H[neg_prod, :])
        sm += (wuf' * hif - wuf' * hjf) > 0
    end
    return sm / iters
end

auc_insamp(biter::AbstractBPRIter, B::BPRResult; iters=4096) = auc_insamp(biter, B.W, B.H; iters=iters)

# AUC on hold out sample
#function auc_outsamp(biter::BPRIter, W::AbstractArray{Real, 2}, H::AbstractArray{Real, 2})
function auc_outsamp(biter::AbstractBPRIter, W, H)
    sm = 0
    @inbounds @simd for user in 1:biter.nusers
        rand_pos = rand(biter.pos_prods[user])
        rand_neg = rand(biter.neg_prods[user])
        wuf = @view(W[user, :])
        hif_hold = @view(H[biter.pos_holdouts[user], :])
        hjf_hold = @view(H[biter.neg_holdouts[user], :])
        hif_rand = @view(H[rand(biter.pos_prods[user]), :])
        hjf_rand = @view(H[rand(biter.neg_prods[user]), :])
        sm += ((wuf' * hif_hold - wuf' * hjf_rand) > 0) + ((wuf' * hif_rand - wuf' * hjf_hold) > 0)
    end
    return sm / (2 * biter.nusers)
end

auc_outsamp(biter::AbstractBPRIter, B::BPRResult) = auc_outsamp(biter, B.W, B.H)

function auc_outsamp2(biter::AbstractBPRIter, W, H)
    sm = 0
    @inbounds @simd for user in 1:biter.nusers
        wuf = @view(W[user, :])
        hif_hold = @view(H[biter.pos_holdouts[user], :])
        hjf_hold = @view(H[biter.neg_holdouts[user], :])
        sm += ((wuf' * hif_hold - wuf' * hjf_hold) > 0)
    end
    return sm / biter.nusers
end

auc_outsamp2(biter::AbstractBPRIter, B::BPRResult) = auc_outsamp2(biter, B.W, B.H)

function grid_search(data::AbstractArray{<:Real, 2}; sample_count=1,
                     ks=Integer.(linspace(10, 100, 3)),
                     λws=-linspace(0.001, 0.1, 3),
                     λhps=-linspace(0.001, 0.1, 3),
                     λhns=-linspace(0.001, 0.1, 3),
                     αs=linspace(0.001, 0.1, 3),
                     tol=1e-5, loop_size=4096, max_iters=0, min_iters=1,
                     min_auc=0.0)
    iterover = repeat(reshape(collect(Iterators.product(ks, λws, λhps, λhns, αs)),
                              :), inner=[sample_count])
    biterorig = BPRIter(data)
    results = pmap(params -> begin
            biter = BPRIter(biterorig)
            k, λw, λhp, λhn, α = params
            res = bpr(biter, k, λw, λhp, λhn, α;
                      tol=tol, loop_size=loop_size, max_iters=max_iters,
                      min_iters=min_iters, min_auc=min_auc)
            df = DataFrame(converged = res.converged,
                           value = res.value,
                           bpr_opt = res.bpr_opt,
                           auc_insample = res.auc_insample,
                           auc_outsample = res.auc_outsample,
                           auc_outsample2 = res.auc_outsample2,
                           iters = res.iters,
                           k = res.k,
                           lw = res.λw,
                           lhp = res.λhp,
                           lhn = res.λhn,
                           alpha = res.α,
                           tol = res.tol,
                           max_iters = res.max_iters,
                           min_iters = res.min_iters,
                           min_auc = res.min_auc)
            return df
        end,
        iterover)
    df = vcat(results...)
    return df
end

end #module

