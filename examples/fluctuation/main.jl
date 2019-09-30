include("../../src/hp.jl")
include("dataLoader.jl")


# data
files = ["examples/fluctuation/eventData/" *
         string(i) * "event.txt"
         for i in 0:19]
data = load_data(files)

# 0.9 size of data
train_data =  Array{Tuple, 1}()
for d in data
    t, e, f = d
    n = length(t)
    push!(train_data, (t[1:Int(round(0.9 * n))],
                       e[1:Int(round(0.9 * n))],
                       f[1:Int(round(0.9 * n)), :])
                       )
end


# model
model = Hawkes(event_types_num=2, features_num=0)


# train
loss_his = Array{Float64, 1}()
train!(model, data, loss_his; iterations=100)


# plot
using PyCall
@pyimport matplotlib.pyplot as plt

# plt.imshow(model.alpha)
# plt.show()
# plt.imshow(model.mu)
# plt.show()

# predict
# plt.hist(data[1][1], 200)
# plt.show()

# using CSV``

for i in 1:20
    plt.subplot(211)
    plt.hist(data[i][1], 200)
    predict(model, train_data[i], data[i][1][end] - train_data[i][1][end])
    plt.subplot(212)
    plt.hist(train_data[i][1], 200)
    plt.show()
    # println(data[i][1])
    # println(train_data[i][1])

    # output
    f = open("examples/fluctuation/predData/" * string(i - 1) * "pred.txt", "a")
    for (t, e) in zip(train_data[i][1], train_data[i][2])
        writedlm(f, [t e])
    end
    close(f)
end
