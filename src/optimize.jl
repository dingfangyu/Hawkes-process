"""
get_pq:
    p, q: ADMM auxiliary matrix
"""
function get_pq(
        model::Hawkes,
        data_s::Tuple
    )::Tuple

    t, e, f, T0, T = data_s
    # sequence length
    n = length(t)

    # result
    p = zeros(n, n)
    q = zeros(n, model.features_num + 1)
    for i = 1:n
        den = 0.0
        den += sum(model.mu[e[i], :]' * f[i, :]) # 1x1 mat sum to scalar
        for j = 1:i - 1
            den += model.alpha[e[i], e[j]] * g(t[i] - t[j], model)
        end

        for j = 1:i - 1
            p[i, j] = model.alpha[e[i], e[j]] * g(t[i] - t[j], model) / den
        end

        for k = 1:model.features_num + 1
            q[i, k] = model.mu[e[i], k] * f[i, k] / den
        end
    end

    return (p, q)
end


"""
training by ADMM
"""
function train!(
    model::Hawkes,
    data::Array,
    lo_his::Array
    ;
    iterations::Int = 100)
    println("begin training")

    samples_num::Int = length(data)

    for iter = 1:iterations
        # get p and q
        p = Array{Array{Float64, 2}, 1}()
        q = Array{Array{Float64, 2}, 1}()
        for s = 1:samples_num
            ps, qs = get_pq(model, data[s])
            push!(p, ps)
            push!(q, qs)
        end

        # update mu[ev, k]
        for ev in 1:model.event_types_num
            for k in 1:model.features_num + 1
                mu_num = 0.0
                mu_den = 0.0
                for s = 1:samples_num
                    t, e, f, T0, T = data[s]
                    n = length(t)
                    for i = 1:n
                        if e[i] == ev
                            mu_num += q[s][i, k]
                        end
                    end
                    for i = 1:n - 1
                        mu_den += f[i, k] * (t[i + 1] - t[i])
                    end
                end
                model.mu[ev, k] = mu_num / (mu_den + 1e-5)
            end
        end

        # update alpha[u, v]
        for u = 1:model.event_types_num
            for v = 1:model.event_types_num
                al_num = 0.0
                al_den = 0.0
                for s = 1:samples_num
                    t, e, f, T0, T = data[s]
                    n = length(t)
                    for i = 1:n
                        for j = 1:i - 1
                            if e[i] == u && e[j] == v
                                al_num += p[s][i, j]
                            end
                        end
                    end
                    for j = 1:n
                        if e[j] == v
                            al_den += G(T - t[j], model)
                        end
                    end

                end
                model.alpha[u, v] = al_num / (al_den + 1e-5)
            end
        end

        # update beta
        beta_num = 0.0
        beta_den = 0.0
        for s in 1:samples_num
            t, e, f, T0, T = data[s]
            n = length(t)
            for i = 1:n
                for j = 1:i - 1
                    beta_num += p[s][i, j]
                    beta_den += p[s][i, j] * (t[i] - t[j])
                end
            end
        end
        model.beta = beta_num / (beta_den + 1e-5)


        # push
        println("iteration: ", iter, ", loss: ", loss(model, data))
    end
end
