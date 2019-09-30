include("../../src/hp.jl")
include("dataLoader.jl")


# data
files = ["examples/fluctuation/event/" *
         string(i) * "event.txt"
         for i in 0:19]
data = load_data(files)


# model
model = Hawkes(event_types_num=2, features_num=0)


# train
loss_his = Array{Float64, 1}()
train!(model, data, loss_his; iterations=100)


# plot
using PyCall
@pyimport matplotlib.pyplot as plt

plt.imshow(model.alpha)
plt.show()
plt.imshow(model.mu)
plt.show()

# predict
plt.hist(data[1][1], 200)
plt.show()


# predict(model, data[1], 10000.0)
#
#
# plt.hist(data[1][1], 2000)
# plt.show()
