#=
train:
- Julia version: 1.2.0
- Author: ding
- Date: 2019-09-24
=#

include("chawk.jl")

"""
train data
    1. data
        an array of `n` different event sequences::Array{Tuple}
    2. training method
        ADMM
"""
function train!(model::chawk, data::Array)
    println(model.mu)
end

"""
kernel function
"""
const lambda = 10
g(x) = lambda * exp(-lambda * x)
G(x) = 1. - exp(-lambda * x)

"""
1-dim vector's multiplication
"""
function mul(x, y)
   sum(x .* y)
end

"""
loss function (model, data):
    1. negative log-likelihood of total probability of observing the sequences in the time span.
    2. L1-norm (TODO)
    3. L2-norm (TODO)
"""
function loss(model::chawk, data::Array)::Float64
    # log-lilklihood:
    log_likelihood = 0.

    # for each sequence(2-dim Array) i
    for (i, sequence) in enumerate(data)
        time, type, feature = sequence[:, 1], sequence[:, 2], sequence[:, 3:3 + model.feature_dim - 1]
        ni = length(time)
        time_begin, time_end = time[1], time[end]

        # log λ
        for j in 1:ni
            ag_sum = 0.
            for k in 1:j - 1
                ag_sum += model.alpha[Int(type[j]), Int(type[k])] *
                            g(time[j] - time[k])
            end

            log_likelihood += log(
                                mul(model.mu[Int(type[j]), :], feature[j, :]) +
                                ag_sum
                                )
        end

        # ∫ μ
        ft_quad = zeros(model.feature_dim)
        for j in 1:ni - 1
            ft_quad += feature[j, :] * (time[j + 1] - time[j])
        end

        for j in 1:model.event_type_num
            log_likelihood -= mul(model.mu[j, :], ft_quad)
        end

        # ∫ α g
        for ty in 1:model.event_type_num
            for j in 1:ni - 1
                log_likelihood -= model.alpha[ty, Int(type[j])] * G(time_end - time[j])
            end
        end
    end

    return -log_likelihood
end


"""
loss_lowerbound (model, data):
    log-likelihood's lower boundary
"""
function loss_lowerbound(model::chawk, data::Array)

end

model = chawk(2,5)

data = [[1. 1 [1. 1.]; [2. 1 [1. 1.]]]]

# train!(model, data)

println(loss(model, data))

# println(data[1][:, 1])
