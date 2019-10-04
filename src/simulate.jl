using Random


"""
intensity
"""
function intensity(
        model::Hawkes,
        history_data::Tuple,
        t_now::Float64,
        event_type::Int,
    )

    ag_sum = 0.0
    t, e, f, T0, T = history_data
    n = findlast(x -> x < t_now, t)
    for i = 1:n
        ag_sum += model.alpha[event_type, e[i]] * g(t_now - t[i])
    end

    return model.mu[event_type, :]' * f[end, :] + ag_sum # f
end

"""
sum intensity
"""
function sum_intensity(model::Hawkes, history_data::Tuple, t_now::Float64)
    return sum(intensity(model, history_data, t_now, ev) for ev in 1:model.event_types_num)
end

"""
exp distribution
"""
function exp_distrib(l::Float64)
    return -log(1 - rand(Float64)) / l
end


"""
select
 - sum(a) == 1
"""
function random_select(a::Array)
    u = rand(Float64)
    s = 0.0
    for (i, x) in enumerate(a)
        s += x
        if u < s
            return i
        end
    end
end


"""
prediction
 - Ogata's thinning algorithm
"""
function predict(
        model::Hawkes,
        history_data::Tuple,
        time_range::Float64
    )
    println("begin prediction")

    t, e, f, T0, T = history_data

    t_now = T
    while t_now <= T + time_range
        intensity_sum = sum_intensity(model, history_data, t_now)
        delta_t = exp_distrib(intensity_sum)
        intensity_delta_sum = sum_intensity(model, history_data, t_now + delta_t)
        u = rand(Float64)

        if t_now + delta_t < T + time_range && u < intensity_delta_sum / intensity_sum
            ti = t_now + delta_t
            push!(t, ti)
            push!(e, random_select(
                        [intensity(model, history_data, ti, ev) / sum_intensity(model, history_data, ti)
                         for ev in 1:model.event_types_num]
                             ))
        end

        t_now += delta_t
    end
end
