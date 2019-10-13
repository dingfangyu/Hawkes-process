include("../../src/hp.jl")
include("dataLoader.jl")

# data
files = ["examples/fluctuation/M4_20/" *
         string(i) * "event.txt"
         for i in 0:19]
data = load_data(files)

# 0.9 size of data
const proportion = 0.95

train_data = Array{Tuple, 1}()

for d in data
    t, e, f, T0, T = d

    train_n = findlast(x -> (x < proportion * T), t)

    push!(train_data, (t[1:train_n],
                       e[1:train_n],
                       f[1:train_n, :],
                       0.0,
                       proportion * T
                       )
         )
end


# model
model = Hawkes(event_types_num=2, features_num=0)


# train
loss_his = Array{Float64, 1}()
train!(model, data, loss_his; iterations=50)


# plot
using PyCall
@pyimport matplotlib.pyplot as plt

println(model.beta)
plt.imshow(model.alpha)
plt.show()
plt.imshow(model.mu)
plt.show()

# predict
for i in 1:20
    predict(model, train_data[i], data[i][5] - train_data[i][5])

    plt.subplot(211)
    plt.hist(data[i][1], bins=200, range=(0.0, data[i][5]))
    plt.axvline(x=train_data[i][5], ymin=0, color="orange")

    plt.subplot(212)
    plt.hist(train_data[i][1], bins=200, range=(0.0, data[i][5]))
    plt.axvline(x=train_data[i][5], ymin=0, color="orange")

    plt.savefig("examples/fluctuation/predData/plot" * string(i - 1) * ".jpg")
    plt.close()
    # plt.show()

    # output
    f = open("examples/fluctuation/predData/" * string(i - 1) * "pred.txt", "w")
    for (t, e) in zip(train_data[i][1], train_data[i][2])
        writedlm(f, [t e])
    end
    close(f)
end
