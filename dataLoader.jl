using DelimitedFiles


function load_data(files::Array{String, 1})
    data = Array{Array, 1}()

    for file in files
        raw_data = readdlm(file)

        t = raw_data[:, 1]
        e = Array{Int, 1}()
        for (i, ev) in enumerate(raw_data[:, 2])
            if ev == -1
                push!(e, 2)
            else
                push!(e, 1)
            end
        end

        push!(data, Array{Any}[t, e])
    end

    return data
end
