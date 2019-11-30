using DelimitedFiles

"""
A file corresponds to a sample,
recording a sequence of (t_i, e_i, f_i),
e_i refers to the event type of the i-th event.
"""
function load_data(files::Array{String, 1})
    data = Array{Tuple, 1}()

    for file in files
        raw_data = readdlm(file)

        t = raw_data[1:end - 1, 1]
        T = raw_data[end, 2]
        n = length(t)

        e = Array{Int, 1}()
        for (i, ev) in enumerate(raw_data[1:end - 1, 2])
            if ev == -1
                push!(e, 2)
            else
                push!(e, 1)
            end
        end

        if size(raw_data)[2] > 2
            f = raw_data[1:end - 1, 3:end]
            f = hcat(ones(Float64, length(t)), f)
        else
            f = hcat(ones(Float64, length(t)))
        end

        push!(data, (t, e, f, 0.0, T))
    end

    return data
end
