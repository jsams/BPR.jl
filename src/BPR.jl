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
    converged::Integer
    value::Real
    iters::Integer
    k::Integer
    λw::Real
    λhp::Real
    λhn::Real
    α::Real
    tol::Real
    max_iters::Integer
    num_converged::Integer
    W::AbstractArray{<:Real, 2}
    H::AbstractArray{<:Real, 2}
end

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
    neg_prod = sample(bpr.neg_prods[user], bpr.neg_weights[user])
    return (user, pos_prod, neg_prod), nothing
end

Base.done(bpr::BPR_iter, state) = false

Base.iteratorsize(bpr::BPR_iter) = Base.IsInfinite()

Base.iteratoreltype(bpr::BPR_iter) = typeof((bpr.users[1], bpr.pos_prods[1][1],
                                             bpr.neg_prods[1][1]))


function bpr(biter::BPR_iter, k, λw, λhp, λhn, α; 
             tol=1e-5, loop_size=256, max_iters=0, num_converged=1,
             W=randn(biter.nusers, k), H=randn(biter.nprods, k))
    wuf_new = zeros(k)
    hif_new = zeros(k)
    hjf_new = zeros(k)
    iters = 0
    converged = 0
    stepsize = 1 / tol # ensure this is defined outside of while scope
    progress = ProgressThresh(tol, "Searching:")
    info("Starting BPR loop.")
    @inbounds while true
        @simd for _ in 1:loop_size
            (user, pos_prod, neg_prod), _ = next(biter, nothing) # expensive: 222
            wuf = @view(W[user, :])
            hif = @view(H[pos_prod, :])
            hjf = @view(H[neg_prod, :])
            xuij = wuf' * hif - wuf' * hjf
            #xuij = sum(wuf .* hif) - sum(wuf .* hjf) # this seems more expensive, but hard to tell
            exuij = exp(-xuij)
            sig = exuij / (1 + exuij)
            wuf_new[:] .= wuf .+ α .* (sig .* (hif .- hjf) .+ λw .* wuf) # wuf_grad = hif - hjf
            hif_new[:] .= hif .+ α .* (sig .* wuf .+ λhp .* hif) # hif_grad = wuf
            hjf_new[:] .= hjf .+ α .* (sig .* -wuf .+ λhn .* hjf) # hjf_grad = -wuf
            #stepsize = sum(abs, wuf .- wuf_new) .+ sum(abs, hif .- hif_new) .+ sum(abs, hjf .- hjf_new) # expensive! 478
            stepsize = sum(abs.(wuf .- wuf_new) .+ abs.(hif .- hif_new) .+ abs.(hjf .- hjf_new)) # expensive! 478
            wuf[:] = wuf_new
            hif[:] = hif_new
            hjf[:] = hjf_new
        end
        iters += 1
        ProgressMeter.update!(progress, stepsize)
        if stepsize < tol
            converged += 1
            if converged >= num_converged
                break
            end
        elseif ((max_iters > 0) & (iters > max_iters))
            warn("max iters reached without convergence, breaking.")
            break
        elseif isnan(stepsize)
            warn("stepsize is nan, breaking.")
            break
        else
            converged = 0
        end
    end

    return BPRResult(converged, stepsize, iters, k, λw, λhp, λhn, α , tol,
                     max_iters, num_converged, W, H)
end

function bpr(data::AbstractArray{<:Real, 2}, k, λw, λhp, λhn, α; 
             tol=1e-5, loop_size=256, max_iters=0, num_converged=1,
             W=randn(size(data, 1), k), H=randn(size(data, 2), k))
    nusers, nprods = size(data)
    if nusers < k | nprods < k
        error("Number of rows and columns must both be greater than k, $k")
    end
    # quick lookups for each user of consumed and unconsumed items
    info("Initializing iterator")
    biter = BPR_iter(data)
    return bpr(biter, k, λw, λhp, λhn, α,
               tol=tol, loop_size=loop_size, max_iters=max_iters,
               num_converged=num_converged, W=W< H=H)
end

function bpr(data::AbstractArray{<:Real, 2}, B::BPRResult)
    bpr(data, B.k, B.λw, B.λhp, B.λhn, B.α; 
        tol=B.tol, max_iters=B.max_iters, num_converged=B.num_converged,
        W=copy(B.W), H=copy(B.H))
end

function bpr(biter::BPR_iter, B::BPRResult)
    bpr(biter, B.k, B.λw, B.λhp, B.λhn, B.α; 
        tol=B.tol, max_iters=B.max_iters, num_converged=B.num_converged,
        W=copy(B.W), H=copy(B.H))
end

#function BPR_AUC(allbutone, holdouts, ...)
#    # remove one data item per user
#    bpr = BPR(allbutone, ...)
#    auc = AUC(bpr, holdouts)
#
#
#
#end





end #module

