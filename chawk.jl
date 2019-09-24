#=
chawk:
- Julia version: 1.2.0
- Author: ding
- Date: 2019-09-24
=#

"""
context-sensitive Hawkes process model
1. context
    feature_dim: number of features.
2. multivariation
    event_type_num: number of event type.
3. sparsity (TODO)
    L1: L1-norm for matrix alpha's sparsity
"""
mutable struct chawk
    feature_dim::Int
    event_type_num::Int
    mu::Array
    alpha::Array

    function chawk(feature_dim::Int, event_type_num::Int)
        mu = rand(Float64, event_type_num, feature_dim)
        alpha = rand(Float64, event_type_num, event_type_num)
        return new(feature_dim, event_type_num, mu, alpha)
    end
end

