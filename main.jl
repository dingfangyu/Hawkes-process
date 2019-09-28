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
# println(model.mu)
# println(model.alpha)


using PyCall
@pyimport matplotlib.pyplot as plt


# plt.plot(loss_his, label="loss")
# plt.legend()
# plt.show()
#
# plt.imshow(model.alpha)
# plt.show()


# predict
plt.plot(data[1][1])
plt.show()


predict(model, data[1], 1000.0)


plt.plot(data[1][1])
plt.show()
