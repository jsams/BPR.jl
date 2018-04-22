include("BPR.jl")
import Main.BPR
using BenchmarkTools
using SparseArrays

function plain_iters(B::BPR.AbstractBPRIter; num_iters=4096)
    sm = 0
    @inbounds for _ in 1:num_iters
        user, p, n = BPR.draw_upn_tup(B)
        sm += user + p + n
    end
    return sm
end

N = 2500
P = 3200
D = 0.06
X = sprand(P, N, D);
@time bd = BPR.BPRIterDense(X)
#@time bs = BPR.BPRIterSparse(X)
@time bb = BPR.BPRIterBits(X)

Base.summarysize(bd)
#Base.summarysize(bs)
Base.summarysize(bb)
Base.summarysize(bb) / Base.summarysize(bd)
#Base.summarysize(bb) / Base.summarysize(bs)

@time plain_iters(bd)
#@time plain_iters(bs)
@time plain_iters(bb)

dbench = @benchmark plain_iters($(bd))
#sbench = @benchmark plain_iters($(bs))
bbench = @benchmark plain_iters($(bb))

