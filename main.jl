include("Hawkes.jl")

# read data
using DelimitedFiles

data = Array{Array, 1}()
# for file = 15:15
#     raw_data = readdlm("./event/" * string(file) * "event.txt")
#
#     t = raw_data[:, 1]
#     e = Array{Int, 1}()
#     for (i, ev) in enumerate(raw_data[:, 2])
#         if ev == -1
#             push!(e, 2)
#         else
#             push!(e, 1)
#         end
#     end
#
#     push!(data, [t, e])
# end
#
# # train
model = Hawkes(1)

t = readdlm("s1e1-data.txt")[:, 1]
# println(t)
e = ones(Int, length(t))
# println(e)
push!(data, [t, e])

loss_his = [loss(model, data)]
train!(model, data, loss_his; iterations=300)


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
