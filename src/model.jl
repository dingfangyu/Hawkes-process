"""
context(feature)-sensitive Hawkes process model
with multiple samples and event types
"""
mutable struct Hawkes
    event_types_num::Int
    features_num::Int
    mu::Array
    alpha::Array
    beta::Float64

    function Hawkes(
        ;
        event_types_num=1,
        features_num=0
    )
        etn = event_types_num
        fn = features_num

        mu = rand(Float64, etn, 1 + fn) # 2-dim Array
        alpha = rand(Float64, etn, etn) # 2-dim Array
        beta = rand(Float64)
        return new(etn, fn, mu, alpha, beta)
    end
end
