module BPR

import Base
using DataFrames
using ProgressMeter

export BPRResult, BPRIter, BPRIterBits, BPRIterDense, BPRIterSparse, bpr, auc_insamp,
       auc_outsamp, auc_outsamp2, grid_search, draw_upn_tup, draw_posneg, draw_holdout


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
          k:              $(size(B.W, 1))
          converged:      $(B.converged)
          value:          $(B.value)
          bpr_opt:        $(B.bpr_opt)
          auc_insample:   $(B.auc_insample)
          auc_outsample:  $(B.auc_outsample)
          auc_outsample2:  $(B.auc_outsample2)
          iters:          $(B.iters)"""
end

Base.show(io::Base.IO, B::BPRResult) = print(io, string(B))

function verifydata(data::AbstractArray{T, 2}) where T
    good_users = find(x -> (x>1), # which users have more than 1 product consumed
                      sum(x -> x > 0, data, 1)) # number of products per user
    if size(good_users, 1) < size(data, 2)
        warn("$(size(data, 2) - size(good_users, 1)) users were removed for insufficient data.")
        data = data[:, good_users]
    end
    if any(sum(data, 2) .== 0)
        error("some columns sum to 0. Fix your data. bailing.")
    end
    return data
end

include("IterAbstract.jl")
include("IterDense.jl")
include("IterSparse.jl")
include("IterBits.jl")

# the defaults for our best guess of the most performant type
BPRIter(data::AbstractArray) = BPRIterBits(data)
BPRIter(B::BPRIterBits) = BPRIterBits(B)


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
    * W: an initialized W parameter matrix (k x nuser)
    * H: an initialized H parameter matrix (k x nprod)"""
function bpr(biter::AbstractBPRIter, k, λw, λhp, λhn, α;
             tol=1e-5, loop_size=4096, max_iters=0, min_iters=1, min_auc=0.0,
             W=randn(k, biter.nusers), H=randn(k, biter.nprods))
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
    if min_iters > 1
        iters_progress = Progress(min_iters, "Min Iters:")
    end
    info("Starting BPR loop.")
    while true
        bpr_new = 0.0
        cur_auc = 0.0
        @inbounds for _ in 1:loop_size # simd be might be bad with overlapping assignment 
        #@inbounds @simd for _ in 1:loop_size
            user, post_prod, neg_prod = draw_upn_tup(biter)
            wuf = @view(W[:, user])
            hif = @view(H[:, pos_prod])
            hjf = @view(H[:, neg_prod])
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
        if min_iters > 1
            ProgressMeter.update!(iters_progress, cur_auc)
        end
        if (iters > min_iters) & (stepsize < tol) & (cur_auc > min_auc)
            converged = true
            break
        elseif (max_iters > 0) & (iters >= max_iters)
            ProgressMeter.cancel(progress, "max iters reached")
            if min_auc > 0
                ProgressMeter.cancel(auc_progress, "max iters reached")
            end
            if min_iters > 1
                ProgressMeter.cancel(iters_progress, "max iters reached")
            end
            warn("max iters reached without convergence, breaking.")
            break
        elseif isnan(stepsize)
            ProgressMeter.cancel(progress, "NaNs found")
            if min_auc > 0
                ProgressMeter.cancel(auc_progress, "NaNs found")
            end
            if min_iters > 1
                ProgressMeter.cancel(iters_progress, "NaNs found")
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
             W=randn(k, size(data, 2)), H=randn(k, size(data, 1)))
    nprods, nusers = size(data)
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
        user, pos_prod, neg_prod = draw_upn_tup(biter)
        wuf = @view(W[:, user])
        hif = @view(H[:, pos_prod])
        hjf = @view(H[:, neg_prod])
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
        pos_hold, neg_hold = draw_holdout(user, biter)
        pos_rand, neg_rand = draw_posneg(user, biter)
        wuf = @view(W[:, user])
        hif_hold = @view(H[:, pos_hold])
        hjf_hold = @view(H[:, neg_hold])
        hif_rand = @view(H[:, pos_rand])
        hjf_rand = @view(H[:, neg_rand])
        sm += ((wuf' * hif_hold - wuf' * hjf_rand) > 0) + ((wuf' * hif_rand - wuf' * hjf_hold) > 0)
    end
    return sm / (2 * biter.nusers)
end

auc_outsamp(biter::AbstractBPRIter, B::BPRResult) = auc_outsamp(biter, B.W, B.H)

function auc_outsamp2(biter::AbstractBPRIter, W, H)
    sm = 0
    @inbounds @simd for user in 1:biter.nusers
        ph, nh = draw_holdout(user, biter)
        wuf = @view(W[:, user])
        hif_hold = @view(H[:, ph])
        hjf_hold = @view(H[:, nh])
        sm += ((wuf' * hif_hold - wuf' * hjf_hold) > 0)
    end
    return sm / biter.nusers
end

auc_outsamp2(biter::AbstractBPRIter, B::BPRResult) = auc_outsamp2(biter, B.W, B.H)

function grid_search(biterorig::AbstractBPRIter; sample_count=1,
                     ks=Integer.(linspace(10, 100, 3)),
                     λws=-linspace(0.001, 0.1, 3),
                     λhps=-linspace(0.001, 0.1, 3),
                     λhns=-linspace(0.001, 0.1, 3),
                     αs=linspace(0.001, 0.1, 3),
                     tol=1e-5, loop_size=4096, max_iters=0, min_iters=1,
                     min_auc=0.0)
    iterover = repeat(reshape(collect(Iterators.product(ks, λws, λhps, λhns, αs)),
                              :), inner=[sample_count])
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

grid_search(data::AbstractArray{<:Real, 2}; kwargs...) = grid_search(BPRIter(data); kwargs...)

end #module

