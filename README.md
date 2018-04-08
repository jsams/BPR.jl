# BPR.jl
Implementation of [Rendle et. al 2009 Bayesian Personalized Ranking for Matrix
Factorization](https://dl.acm.org/citation.cfm?id=1795167).

```julia
Pkg.clone("https://github.com/jsams/BPR.jl.git")
using BPR
# generate some data. The values are unimportant, only zero versus > 0
X = sprand(3000, 4000, 0.05)
# by creating an iterator from the data, can re-use it for other runs
biter = BPR.BPR_iter(X)
@time bpr = BPR.bpr(biter, 10, 0.01, 0.01, 0.01, 0.01; tol=0.01, max_iters=10)
# but could also run straight from the matrix
@time bpr = BPR.bpr(X, 10, 0.01, 0.01, 0.01, 0.01; tol=0.01, max_iters=10)

# did it converge
bpr.converged
# what tolerance was achieved
bpr.value
# what was the BPR-OPT criterion in the last run
bpr.bpr_opt
# how well do we predict in a hold out sample
bpr.auc_outsample 
# matrix of predicted rankings from
bpr.W * bpr.H'
```
