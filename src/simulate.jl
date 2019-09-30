"""
intensity
"""
function intensity(
        model::Hawkes,
        history_data::Tuple{Array{Float64, 1}, Array{Int, 1}, Array{Float64, 2}},
        t_now::Float64,
        event_type::Int,
    )

    ag_sum = 0.0
    t, e, f = history_data
    n = length(t)
    for i = 1:n
        ag_sum += model.alpha[event_type, e[i]] * g(t_now - t[i])
    end

    return model.mu[event_type, :]' * f[end, :] + ag_sum
end


"""
prediction
"""
const dt = 1.0
function predict(
        model::Hawkes,
        history_data::Tuple{Array{Float64, 1}, Array{Int, 1}, Array{Float64, 2}},
        time_range::Float64
    )
    println("begin prediction")

    t, e, f = history_data

    T = t[end]

    for t_now in T:dt:T + time_range
        for ev in 1:model.event_types_num
            prob::Float64 = rand(Float64)

            if intensity(model,
                    history_data,
                    t_now,
                    ev
                ) > prob * dt

                push!(t, t_now)
                push!(e, ev)
                # push!(f, f[end]) #
            end
        end
    end
end
