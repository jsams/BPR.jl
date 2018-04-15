#include("BPR.jl")
import BPR

N = 2500
P = 3200
D = 0.06
X = sprand(N, P, D);
@time bd = BPR.BPRIterDense(X)
@time bs = BPR.BPRIterSparse(X)

function plain_iters(B::BPR.AbstractBPRIter)
    sm = 0
    @inbounds for user in B.users
        (user, p, n), _ = next(B, nothing)
        sm += user + p + n
    end
end

@time plain_iters(bd)
@time plain_iters(bs)
