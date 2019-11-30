include("../../src/hp.jl")


using PyCall
@pyimport matplotlib.pyplot as plt

tpp_path = "./examples/disease-network/tpp.csv"
disease_path = "./examples/disease-network/disease.csv"

using DataFrames, CSV

tpp_df = CSV.read(tpp_path)
disease_df = CSV.read(disease_path)

# train_data: [
#   time_series,
#   event_sequence,
#   feature_sequence,
#   T0,
#   T
# ]
train_data = Array{Tuple, 1}()

subjects = unique(tpp_df[!, :subject_id])
for subject in subjects
    tpp_s = tpp_df[tpp_df.subject_id .== subject, :]

    t = tpp_s[:, :age]
    e = tpp_s[:, :primary_id] .+ 1
    f = hcat(ones(Float64, length(t)))
    T0 = 0.0
    T = t[end] + 0.01
    #
    # println(t)
    # println(e)
    # println(f)
    # break

    push!(train_data, (t, e, f, T0, T))
end


# model
model = Hawkes(event_types_num=32, features_num=0)

println(model.beta)
plt.imshow(model.alpha)
plt.show()
plt.imshow(model.mu)
plt.show()

# train
loss_his = Array{Float64, 1}()
train!(model, train_data, loss_his; iterations=10)

println(model.beta)
plt.imshow(model.alpha)
plt.show()
plt.imshow(model.mu)
plt.show()
