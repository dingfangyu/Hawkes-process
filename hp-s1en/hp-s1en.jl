mutable struct hp_s1en
    event_num::Int
    mu::Array
    alpha::Array

    function hp_s1en(event_num::Int)
        mu = rand(Float64, event_num) # 1-dim Array
        alpha = rand(Float64, event_num, event_num) # 2-dim Array
        return new(event_num, mu, alpha)
    end
end # struct


"""
loss: -log-likelihood
- data: {(t_i, e_i)}, t_i :: Float, e_i :: Int
"""
function loss(model::hp_s1en, t::Array, e::Array)
    n::Int = length(t)
    T::Float64 = t[end]

    l::Float64 = 0.0
    for i = 1:n
        ag_sum::Float64 = 0.0
        for j = 1:i - 1
            ag_sum += 
        end

        l += log(
            model.mu[e[i]] + ag_sum
        )
    end

end # function
