"""
negative log-likelihood
 - data: Array{Tuple, 1}
         [(t_s1, e_s1, f_s1),
          (t_s2, e_s2, f_s2),
          ...,
          (t_sm, e_sm, f_sm)]

         t_si: Array{Float64, 1}    # happen times of sample i
         e_si: Array{Int, 1}        # event types of sample i
         f_si: Array{Float64, 2}    # features of sample i
"""
function loss(model::Hawkes, data::Array)::Float64
    samples_num::Int = length(data)

    l::Float64 = 0.0

    for s = 1:samples_num
        # shadow copy
        t::Array{Float64, 1},
        e::Array{Int, 1},
        f::Array{Float64, 2} = data[s]

        # events num
        n::Int = length(t)

        # time span
        T0::Float64 = t[1] # maybe 0, you can set it.
        T::Float64 = t[end]

        # log(intensity) items
        for i = 1:n
            ag_sum::Float64 = 0.0
            for j = 1:i - 1
                ag_sum += model.alpha[e[i], e[j]] * g(t[i] - t[j])
            end

            l += log(
                model.mu[e[i], :]' * f[i, :] # linear combination
                + ag_sum
            )
        end

        # mu items
        ft_sum = zeros(Float64, 1, model.features_num + 1)
        for i = 1:n - 1
            ft_sum += f[i, :] * (t[i + 1] - t[i])
        end
        l -= sum(ft_sum' * sum(model.mu, dims=1)) # item

        # alpha items
        aG_sum::Float64 = 0.0
        for ev = 1:model.event_types_num
            for j = 1:n
                aG_sum += model.alpha[ev, e[j]] * G(T - t[j])
            end
        end

        l -= aG_sum
    end

    return -l
end # function
