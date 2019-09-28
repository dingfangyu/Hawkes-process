include("Hawkes.jl")
include("dataLoader.jl")


# data
files = ["event/0event.txt"]
data = load_data(files)


# model
model = Hawkes(event_types_num=2)


# train
loss_his = Array{Float64, 1}()
train!(model, data, loss_his; iterations=100)


# plot
using PyCall
@pyimport matplotlib.pyplot as plt


# predict
plt.hist(data[1][1], 200)
plt.show()


predict(model, data[1], 10000.0)


plt.hist(data[1][1], 2000)
plt.show()
