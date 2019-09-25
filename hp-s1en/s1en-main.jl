include("hp-s1en.jl")

# read data
using DelimitedFiles
raw_data = readdlm("naive-data.txt")
data = raw_data[:, 1]


# train
model = hp_s1en()
loss_his = [loss(model, t, e)]
train!(model, t, e, loss_his; iterations=300)


# plot
using PyCall
@pyimport matplotlib.pyplot as plt

plt.plot(loss_his[2:end], label="loss")
plt.legend()
plt.show()

plt.plot(model.mu)
plt.show()
plt.plot(model.alpha)
plt.show()
