# data properties in convenient format with iterator protocol
abstract type  AbstractBPRIter end

Base.string(bpr::AbstractBPRIter) = "$(bpr.nusers) x $(bpr.nprods) $(typeof(bpr))"
Base.show(io::Base.IO, bpr::AbstractBPRIter) = print(io, string(bpr))


Base.start(bpr::AbstractBPRIter) = nothing

Base.done(bpr::AbstractBPRIter, state) = false

Base.iteratorsize(bpr::AbstractBPRIter) = Base.IsInfinite()

