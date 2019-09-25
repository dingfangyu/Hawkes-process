include("naive-hp.jl")


# read data
using DelimitedFiles
raw_data = readdlm("naive-data.txt")
data = raw_data[:, 1]


# train
model = naive_hp()
mu_his = [model.mu]
al_his = [model.alpha]
loss_his = [loss(model, data)]
loss_ub_his = [loss_ub(model, data)]
train!(model, data, mu_his, al_his, loss_his; iterations=300)


# plot
using PyCall
@pyimport matplotlib.pyplot as plt

plt.plot(mu_his, label="mu")
plt.plot(al_his, label="alpha")
plt.legend()
plt.show()

plt.plot(loss_his[2:end], label="loss")
plt.plot(loss_ub_his[2:end], label="loss_ub")
plt.legend()
plt.show()