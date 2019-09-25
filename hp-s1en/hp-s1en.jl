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
kernel function
- exp kernel
"""
const lambda = 0.1
g(t::Float64) = lambda * exp(-lambda * t)
G(t::Float64) = 1 - exp(-lambda * t)


"""
loss: -log-likelihood
- data: {(t_i, e_i)}, t_i :: Float, e_i :: Int
"""
function loss(model::hp_s1en, t::Array, e::Array)
    n::Int = length(t)
    T0::Float64 = t[1]
    T::Float64 = t[end]

    # lambda
    l::Float64 = 0.0
    for i = 1:n
        ag_sum::Float64 = 0.0
        for j = 1:i - 1
            ag_sum += model.alpha[e[i], e[j]] * g(t[i] - t[j])
        end

        l += log(
            model.mu[e[i]] + ag_sum
        )
    end


    # mu
    l -= (T - T0) * sum(model.mu)


    # alpha
    aG_sum::Float64 = 0.
    n_e::Int = maximum(e)
    for ev = 1:n_e
        for j = 1:n
            aG_sum += model.alpha[ev, e[j]] * G(T - t[j])
        end
    end

    l -= aG_sum

    return -l
end # function


"""
train
"""
function train!(
    model::hp_s1en,
    t::Array,
    e::Array,
    lo_his::Array
    ;
    iterations = 100
    )
    println("begin training")
    n::Int = length(t)
    n_e::Int = maximum(e)

    for iter = 1:iterations
        # p
        p::Array{Float64,2} = zeros(n, n)

        for i = 1:n
            den::Float64 = 0.0
            den += model.mu[e[i]]
            for j = 1:i - 1
                den += model.alpha[e[i], e[j]] * g(t[i] - t[j])
            end

            for j = 1:i - 1
                p[i, j] = model.alpha[e[i], e[j]] * g(t[i] - t[j]) / den
            end

            p[i, i] = model.mu[e[i]] / den
        end

        # mu
        for ev = 1:n_e
            sum_p_diag::Float64 = 0.0
            for i = 1:n
                if e[i] == ev
                    sum_p_diag += p[i, i]
                end
            end

            model.mu[ev] = sum_p_diag / (T - T0)
        end


        # alpha[u, v]
        for u = 1:n_e
            for v = 1:n_e
                num::Float64 = 0.0
                for i = 1:n
                    for j = 1:i - 1
                        if e[i] == u && e[j] == v
                            num += p[i, j]
                        end
                    end
                end

                den::Float64 = 0.0
                for j = 1:n
                    if e[j] == v
                        den += G(T - t[j])
                    end
                end

                model.alpha[u, v] = num / den
            end
        end

        # push
        push!(lo_his, loss(model, t, e))
    end
end # function
