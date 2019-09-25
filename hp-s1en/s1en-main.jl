include("hp-s1en.jl")

# read data
using DelimitedFiles
# raw_data = readdlm("s1en-data.txt")
raw_data = readdlm("./event/0event.txt")

t = raw_data[:, 1]
e = Array{Int, 1}()
for (i, ev) in enumerate(raw_data[:, 2])
    if ev == -1
        push!(e, 2)
    else
        push!(e, 1)
    end
end

println(e)


# train
model = hp_s1en(2)
loss_his = [loss(model, t, e)]
train!(model, t, e, loss_his; iterations=300)


# plot
using PyCall
@pyimport matplotlib.pyplot as plt

plt.plot(loss_his[2:end], label="loss")
plt.legend()
plt.show()

# plt.plot(model.mu, 'o')
# plt.show()
plt.imshow(model.alpha)
plt.show()

println(model.mu)
println(model.alpha)
