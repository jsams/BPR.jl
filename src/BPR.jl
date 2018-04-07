module BPR

import Base
using ProgressMeter
using StatsBase

export BPRResult, BPR_iter, BPR
# Implements Rendle et at 2009


# test with:
#include("BPR.jl")
#using BPR
#X = sprand(3000, 3000, 0.05)
#biter = BPR.BPR_iter(X)
#@time bpr = BPR.bpr(biter, 10, 0.01, 0.01, 0.01, 0.01; tol=0.01, max_iters=10)
#
#Profile.clear()
#Profile.clear_malloc_data()
#@profile bpr = BPR.bpr(biter, 10, 0.01, 0.01, 0.01, 0.01; tol=0.001, max_iters=100)
#Profile.print(maxdepth=9)




# preferably be able to pass in a BPR object and have it restart stuff from
# where things left off with the given data
# or be able to specify at U and V instead of random init
# tolerance and that kind of shit
# probably best to set defaults
# max iters
# allow non-random init?
# * data: a user x item matrix to learn (sparse or otherwise)
# * k: number of dimensions to learn
# * α: learning rate
# * λw: regularization on user features, 
# * λhp: regularization on positive updates on items
# * λhm: regularization on negative updates on items
# assumes no user has 0 consumed items nor 100% consumed items

struct BPRResult
    # convergence properties
    converged::Bool
    value::Real
    bpr_opt::Real
    auc::Real
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
"""converged:	$(B.converged)
value:	$(B.value)
bpr_opt:	$(B.bpr_opt)
auc:	$(B.auc)
iters:	$(B.iters)"""
end

Base.show(io::Base.IO, B::BPRResult) = show(io, string(B))

# data properties in convenient format with iterator protocol
struct BPR_iter{T<:Real}
    data::AbstractArray{T, 2}
    nusers::Integer
    nprods::Integer
    users::UnitRange{<:Integer}
    prods::IntSet
    weights::AbstractArray{<:Real, 1}
    pos_prods::AbstractArray{<:AbstractArray{<:Integer, 1}, 1}
    neg_prods::AbstractArray{<:AbstractArray{<:Integer, 1}, 1}
    neg_weights::AbstractArray{<:StatsBase.AbstractWeights, 1}
end

function BPR_iter(data::AbstractArray{T, 2}) where T
    nusers, nprods = size(data)
    users = 1:nusers
    prods = IntSet(1:nprods)
    weights = vec(log.(sum(data, 1)))
    pos_prods = [find(data[user, :] .> 0) for user in users]
    neg_prods = [sort(collect(setdiff(prods, posprod))) for posprod in pos_prods]
    neg_weights = [FrequencyWeights(weights[negprod]) for negprod in neg_prods]
    return BPR_iter(data, nusers, nprods, users, prods, weights, pos_prods,
                    neg_prods, neg_weights)
end

Base.start(bpr::BPR_iter) = nothing

function Base.next(bpr::BPR_iter, state)
    user = rand(bpr.users)
    # views don't work here, and i'm not sure that they are necessary
    pos_prod = rand(bpr.pos_prods[user])
    #neg_prod = sample(bpr.neg_prods[user], bpr.neg_weights[user])
    neg_prod = rand(bpr.neg_prods[user])
    return (user, pos_prod, neg_prod), nothing
end

Base.done(bpr::BPR_iter, state) = false

Base.iteratorsize(bpr::BPR_iter) = Base.IsInfinite()

Base.iteratoreltype(bpr::BPR_iter) = typeof((bpr.users[1], bpr.pos_prods[1][1],
                                             bpr.neg_prods[1][1]))


function bpr(biter::BPR_iter, k, λw, λhp, λhn, α; 
             tol=1e-5, loop_size=256, max_iters=0, min_iters=1, min_auc=0.0,
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
        @inbounds @simd for _ in 1:loop_size # simd be might be bad with random next()
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
            warn("max iters reached without convergence, breaking.")
            break
        elseif isnan(stepsize)
            warn("stepsize is nan, breaking.")
            break
        end
    end

    return BPRResult(converged, stepsize, bpr_new, cur_auc, iters, k, λw, λhp,
                     λhn, α , tol, max_iters, min_iters, min_auc, W, H)
end

function bpr(data::AbstractArray{<:Real, 2}, k, λw, λhp, λhn, α; 
             tol=1e-5, loop_size=256, max_iters=0, min_iters=1, min_auc=0.0,
             W=randn(size(data, 1), k), H=randn(size(data, 2), k))
    nusers, nprods = size(data)
    if nusers < k | nprods < k
        error("Number of rows and columns must both be greater than k, $k")
    end
    # quick lookups for each user of consumed and unconsumed items
    info("Initializing iterator")
    biter = BPR_iter(data)
    return bpr(biter, k, λw, λhp, λhn, α;
               tol=tol, loop_size=loop_size, max_iters=max_iters,
               min_iters=min_iters, min_auc=min_auc, W=W< H=H)
end

function bpr(data::AbstractArray{<:Real, 2}, B::BPRResult)
    bpr(data, B.k, B.λw, B.λhp, B.λhn, B.α; 
        tol=B.tol, max_iters=B.max_iters, min_iters=B.min_iters,
        min_auc=B.min_auc, W=copy(B.W), H=copy(B.H))
end

function bpr(biter::BPR_iter, B::BPRResult)
    bpr(biter, B.k, B.λw, B.λhp, B.λhn, B.α; 
        tol=B.tol, max_iters=B.max_iters, min_iters=B.min_iters, min_auc=B.min_auc,
        W=copy(B.W), H=copy(B.H))
end

function auc(biter::BPR_iter, W::AbstractArray{<:Real, 2},
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

auc(biter::BPR_iter, B::BPRResult; iters=4096) = return auc(biter, B.W, B.H; iters=iters)



#function BPR_AUC(allbutone, holdouts, ...)
#    # remove one data item per user
#    bpr = BPR(allbutone, ...)
#    auc = AUC(bpr, holdouts)
#
#
#
#end





end #module

