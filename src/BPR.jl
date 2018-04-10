module BPR

import Base
using ProgressMeter

export BPRResult, BPR_iter, bpr, auc_insamp, auc_outsamp, auc_outsamp2, grid_search


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

# data properties in convenient format with iterator protocol
struct BPR_iter{T<:Real}
    data::AbstractArray{T, 2}
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
function BPR_iter(data::AbstractArray{T, 2}) where T
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
    neg_prods = [sort(collect(setdiff(prods, posprod))) for posprod in pos_prods]
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
    return BPR_iter(data, nusers, nprods, users, prods, pos_prods, neg_prods,
                    pos_holdouts, neg_holdouts)
end
Base.string(bpr::BPR_iter) = "$(bpr.nusers) x $(bpr.nprods) BPR_iter"
Base.show(io::Base.IO, bpr::BPR_iter) = print(io, string(bpr))
Base.start(bpr::BPR_iter) = nothing

function Base.next(bpr::BPR_iter, state)
    user = rand(bpr.users)
    # views don't work here, and i'm not sure that they are necessary
    pos_prod = rand(bpr.pos_prods[user])
    neg_prod = rand(bpr.neg_prods[user])
    return (user, pos_prod, neg_prod), nothing
end

Base.done(bpr::BPR_iter, state) = false

Base.iteratorsize(bpr::BPR_iter) = Base.IsInfinite()

Base.iteratoreltype(bpr::BPR_iter) = typeof((bpr.users[1], bpr.pos_prods[1][1],
                                             bpr.neg_prods[1][1]))

"""
    find optimal W and H matrix for bpr matrix factorization, returns BPRResult
    * biter: object from BPR_iter
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
               min_iters=min_iters, min_auc=min_auc, W=W, H=H)
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

# AUC computed on random in-sample
function auc_insamp(biter::BPR_iter, W::AbstractArray{<:Real, 2},
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

auc_insamp(biter::BPR_iter, B::BPRResult; iters=4096) = auc_insamp(biter, B.W, B.H; iters=iters)

# AUC on hold out sample
#function auc_outsamp(biter::BPR_iter, W::AbstractArray{Real, 2}, H::AbstractArray{Real, 2})
function auc_outsamp(biter::BPR_iter, W, H)
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

auc_outsamp(biter::BPR_iter, B::BPRResult) = auc_outsamp(biter, B.W, B.H)

function auc_outsamp2(biter::BPR.BPR_iter, W, H)
    sm = 0
    @inbounds @simd for user in 1:biter.nusers
        wuf = @view(W[user, :])
        hif_hold = @view(H[biter.pos_holdouts[user], :])
        hjf_hold = @view(H[biter.neg_holdouts[user], :])
        sm += ((wuf' * hif_hold - wuf' * hjf_hold) > 0)
    end
    return sm / biter.nusers
end

auc_outsamp2(biter::BPR_iter, B::BPRResult) = auc_outsamp2(biter, B.W, B.H)

function grid_search(data::AbstractArray{<:Real, 2};
                     ks=linspace(10, 100, 3),
                     λws=linspace(0.001, 0.1, 3),
                     λhps=linspace(0.001, 0.1, 3),
                     λhns=linspace(0.001, 0.1, 3),
                     αs=linspace(0.001, 0.1, 3),
                     tol=1e-5, loop_size=4096, max_iters=0, min_iters=1,
                     min_auc=0.0)
    iterover = reshape(collect(Iterators.product(ks, λws, λhps, λhns, αs)),  :)
    results = pmap(params -> begin
            biter = BPR.BPR_iter(data)
            k, λw, λhp, λhn, α = params
            res = BPR.bpr(biter, k, λw, λhp, λhn, α;
                      tol=tol, loop_size=loop_size, max_iters=max_iters,
                      min_iters=min_iters, min_auc=min_auc)
            (k, λw, λhp, λhn, α, res)
        end,
        iterover)
    return results
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

