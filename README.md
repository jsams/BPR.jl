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
bpr.W' * bpr.H
```

To figure out hyperparamters (number of dimensions, regularizations, and
learning rate), a handy function, `grid_search` can help you with that. It
takes the data and a vector for each parameter to search over, and constructs
the grid of all those points, running the algorithm for each grid point
`sample_count` times. It constructs a new hold out sample for each run. It
returns a DataFrame with the convergence properties and run settings as the
columns minus the resulting parameterization. It is built to run in parallel,
so starting julia with `-p#` will run in parallel on `#` separate processes.

```
griddf = grid_search(X, sample_count=2; max_iters=100)
```

Up to you to analyze griddf as to whether you need to refine the grid search or
select the optimal hyperparamters.

# TODO
 * would be best not to store negative examples per user for memory reasons,
   but how to efficiently choose among them without storing them? Sets aren't
   THAT fast.  benchmarks.jl suggests ~1.5s for 2500 iterations versus 0.02s. Not
   acceptable.
 * uniform sampling is maybe not quite uniform, read paper to uniform over what.
 * fix progress meters (e.g. min_iters, auc)...maybe just kill insample auc anyway
 * add loop size to result (whoops!)

