# data properties in convenient format with iterator protocol
abstract type  AbstractBPRIter end

Base.string(bpr::AbstractBPRIter) = "$(bpr.nprods) products x $(bpr.nusers) users $(typeof(bpr))"
Base.show(io::Base.IO, bpr::AbstractBPRIter) = print(io, string(bpr))

@inline function draw_upn_tup(bpr::AbstractBPRIter)
    user = rand(bpr.users)
    pp, np = draw_posneg(user, bpr)
    return (user, pp, np)
end

# interface requires:
# 1) users field on which rand() can draw from quickly. suggestion: Integer UnitRange
# 2) draw_posneg(user, B) to draw a random (positive, negative) tuple for a given user
# 3) draw_holdout(user, B) to get the (positive, negative) holdout tuple for a given user
# 4) create instance from a product x user array
# 5) create a new instance with new holdouts with an instance of itself
#
# with these draw_upn_tup(B) to draw a random (user, positive, negative) tuple
# will work just based on the abstract type

