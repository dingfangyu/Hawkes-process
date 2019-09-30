"""
context(feature)-sensitive Hawkes process model
with multiple samples and event types
"""
mutable struct Hawkes
    event_types_num::Int
    features_num::Int
    mu::Array
    alpha::Array

    function Hawkes(
        ;
        event_types_num::Int=1,
        features_num::Int=0
    )
        etn = event_types_num
        fn = features_num

        mu = rand(Float64, etn, 1 + fn) # 2-dim Array
        alpha = rand(Float64, etn, etn) # 2-dim Array
        return new(etn, fn, mu, alpha)
    end
end
