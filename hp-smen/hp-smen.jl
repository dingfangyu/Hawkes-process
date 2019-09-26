mutable struct hp_smen
    event_num::Int
    mu::Array
    alpha::Array

    function hp_smen(event_num)
        mu = rand(Float64, event_num) # 1-dim Array
        alpha = rand(Float64, event_num, event_num) # 2-dim Array
        return new(event_num, mu, alpha)
    end # function
end


"""
kernel function
- exp kernel
"""
const lambda = 0.04
g(t::Float64) = lambda * exp(-lambda * t)
G(t::Float64) = 1 - exp(-lambda * t)


"""
negative log-likelihood
 - data: Array{Any, 1}
         [[t_s1, e_s1], [t_s2, e_s2], ...[t_sm, e_sm]]
         t_si: Array{Float64, 1}
         e_si: Array{Int, 1}
"""
function loss(model::hp_smen, data::Array)
    samples_num::Int = length(data)

    l::Float64 = 0.0

    for s = 1:samples_num
        t::Array{Float64, 1} = data[s][1]
        e::Array{Int, 1} = data[s][2]

        n::Int = length(t)
        T0::Float64 = t[1]
        T::Float64 = t[end]

        # lambda
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

    end


    return -l
end # function


"""
train
"""
function train!(
    model::hp_smen,
    data::Array,
    lo_his::Array
    ;
    iterations::Int = 100)
    println("begin training")

    samples_num::Int = length(data)

    for iter = 1:iterations
        # get p: psij
        # p = Array{Array{Float64, 2}}()
        p = []
        for s = 1:samples_num
            t::Array{Float64, 1} = data[s][1]
            e::Array{Int, 1} = data[s][2]

            n::Int = length(t)
            ps::Array{Float64, 2} = zeros(n, n)

            for i = 1:n
                den::Float64 = 0.0
                den += model.mu[e[i]]
                for j = 1:i - 1
                    den += model.alpha[e[i], e[j]] * g(t[i] - t[j])
                end

                for j = 1:i - 1
                    ps[i, j] = model.alpha[e[i], e[j]] * g(t[i] - t[j]) / den
                end

                ps[i, i] = model.mu[e[i]] / den
            end

            push!(p, ps)
        end

        # update mu
        for ev in 1:model.event_num
            mu_num::Float64 = 0.0
            mu_den::Float64 = 0.0
            for s = 1:samples_num
                t::Array{Float64, 1} = data[s][1]
                e::Array{Float64, 1} = data[s][2]
                n::Int = length(t)
                T0::Float64 = t[1]
                T::Float64 = t[end]

                for i = 1:n
                    if e[i] == ev
                        mu_num += p[s][i, i]
                    end
                end

                mu_den += T - T0
            end

            model.mu[ev] = mu_num / mu_den
        end

        # update alpha
        for u = 1:model.event_num
            for v = 1:model.event_num
                al_num::Float64 = 0.0
                al_den::Float64 = 0.0

                for s = 1:samples_num
                    t::Array{Float64, 1} = data[s][1]
                    e::Array{Float64, 1} = data[s][2]
                    n::Int = length(t)
                    T0::Float64 = t[1]
                    T::Float64 = t[end]

                    for i = 1:n
                        for j = 1:i - 1
                            if e[i] == u && e[j] == v
                                al_num += p[s][i, j]
                            end
                        end
                    end

                    for j = 1:n
                        if e[j] == v
                            al_den += G(T - t[j])
                        end
                    end

                end

                model.alpha[u, v] = al_num / al_den
            end
        end

        # push
        push!(lo_his, loss(model, data))
    end
end
