"""
naive Hawkes process:
- 1 kind of event
- 1 sample
"""
mutable struct naive_hp
    mu::Float64
    alpha::Float64

    function naive_hp()
        mu::Float64 = rand(Float64) / 10
        alpha::Float64 = rand(Float64) / 10
        return new(mu, alpha)
    end # function
end # mutable struct


"""
kernel function
- exp kernel
"""
const lambda = 0.01
g(t::Float64) = lambda * exp(-lambda * t)
G(t::Float64) = 1 - exp(-lambda * t)


"""
negative log_likelihood
- data: time series
"""
function loss(model::naive_hp, data::Array)
    time_begin, time_end = data[1], data[end]

    log_likelihood::Float64 = 0.0

    # sum log(lambda)
    for (i, t) in enumerate(data)
        g_sum::Float64 = 0.0
        for j = 1:i-1
            g_sum += g(t - data[j])
        end

        log_likelihood += log(model.mu + model.alpha * g_sum)
    end # for

    # mu * T
    log_likelihood -= model.mu * (time_end - time_begin)

    # sum alpha * G
    G_sum::Float64 = 0.0
    for t in data
        G_sum += G(time_end - t)
    end
    log_likelihood -= model.alpha * G_sum

    return -log_likelihood
end # function


"""
loss upperbound (is tight)
 - using Jensen's inequality
"""
function loss_ub(model::naive_hp, data::Array)
    time_begin, time_end = data[1], data[end]

    # set p
    n::Int = length(data)
    p::Array{Float64,2} = zeros(n, n)

    # get p
    for i = 1:n
        g_sum::Float64 = 0.0
        for j = 1:i-1
            g_sum += g(data[i] - data[j])
        end
        den = model.mu + model.alpha * g_sum

        for j = 1:i-1
            p[i, j] = model.alpha * g(data[i] - data[j]) / den
            # print(p[i, j], '\t')
        end # for

        p[i, i] = model.mu / den
        # println(p[i, i])
    end # for

    # println(p[1, 1])

    log_likelihood::Float64 = 0.0
    # sum log(lambda)
    for i = 1:n
        for j = 1:i-1
            log_likelihood += p[i, j] * log(
                model.alpha * g(data[i] - data[j]) / p[i, j]
            )
        end

        log_likelihood += p[i, i] * log(model.mu / p[i, i])
    end

    # mu * T
    log_likelihood -= model.mu * (time_end - time_begin)

    # sum alpha * G
    G_sum::Float64 = 0.0
    for t in data
        G_sum += G(time_end - t)
    end
    log_likelihood -= model.alpha * G_sum

    return -log_likelihood
end # function



"""
training
"""
function train!(
    model::naive_hp,
    data::Array,
    mu_history::Array,
    alpha_history::Array,
    loss_hisory::Array,
    loss_ub_hisory::Array;
    iterations::Int=10
    )
    println("start training")

    time_begin, time_end = data[1], data[end]
    n::Int = length(data)

    for iter = 1:iterations
        # set p
        p::Array{Float64,2} = zeros(n, n)

        # get p
        for i = 1:n
            g_sum::Float64 = 0.0
            for j = 1:i-1
                g_sum += g(data[i] - data[j])
            end
            den = model.mu + model.alpha * g_sum

            for j = 1:i-1
                p[i, j] = model.alpha * g(data[i] - data[j]) / den
                # print(p[i, j], '\t')
            end # for

            p[i, i] = model.mu / den
            # println(p[i, i])
        end # for


        # Based on p, update mu and alpha
        # mu
        sum_diag::Float64 = 0.
        for i = 1:n
            sum_diag += p[i, i]
        end
        # println("sum_diag: ", sum_diag)
        model.mu = sum_diag / (time_end - time_begin)


        # alpha
        G_sum::Float64 = 0.
        for i = 1:n
            G_sum += G(time_end - data[i])
        end
        model.alpha = (sum(p) - sum_diag) / G_sum

        # log
        push!(mu_history, model.mu)
        push!(alpha_history, model.alpha)
        push!(loss_hisory, loss(model, data))
        push!(loss_ub_hisory, loss_ub(model, data))
    end
end
