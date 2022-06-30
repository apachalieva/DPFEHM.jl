using ChainRulesCore: length
import DPFEHM
import GaussianRandomFields
import DifferentiableBackwardEuler
import Optim
import Random
import Zygote
import ChainRulesCore
import BSON

using Distributed
using JLD2
using Statistics: mean, std

batch_size = 32 # 32, 64, 128, 256
learning_rate = 1e-4 # 1e-3, 3e-4, 1e-4

println(batch_size)
println(learning_rate)

global losses_train = Float64[]
global losses_test = Float64[]
global rmses_train = Float64[]
global rmses_test = Float64[]
global train_time = Float64[]

@everywhere begin
    using Flux
    using ChainRulesCore
    #using CUDA
    using Random
    using GaussianRandomFields
    using DPFEHM

    n = 51
    ns = [n, n]
    steadyhead = 0e0
    sidelength = 200
    thickness  = 1.0
    mins = [-sidelength, -sidelength] #meters
    maxs = [sidelength, sidelength] #meters
    num_eigenvectors = 200
    sigma = 1.0
    lambda = 50
    cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
    x_pts = range(mins[1], maxs[1]; length=ns[1])
    y_pts = range(mins[2], maxs[2]; length=ns[2])
    mean_value = fill(log(1e-4),(n,n))
    grf = GaussianRandomFields.GaussianRandomField(mean_value, cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)

    pressure_target  = 0.0

    coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid2d(mins, maxs, ns, thickness)
    dirichletnodes = Int[]
    dirichleths = zeros(size(coords, 2))
    for i = 1:size(coords, 2)
        if abs(coords[1, i]) == sidelength || abs(coords[2, i]) == sidelength
            push!(dirichletnodes, i)
            dirichleths[i] = steadyhead
        end
    end

    monitoring_well_node = 781
    @assert coords[:, monitoring_well_node] == [-80, -80]
    injection_extraction_nodes = [1041, 1821]
    @assert coords[:, injection_extraction_nodes[1]] == [-40, -40]
    @assert coords[:, injection_extraction_nodes[2]] == [80, 80]

    function getQs(Qs::Vector, is::Vector)
        sum(Qs .* ((collect(1:size(coords, 2)) .== i) for i in is))
    end

    function solve_numerical(Qs, T)
        logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))
        Qs = getQs(Qs, injection_extraction_nodes)
        @assert length(T) == length(Qs)
        Ks_neighbors = logKs2Ks_neighbors(T)
        h_gw = DPFEHM.groundwater_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
        @assert coords[:, monitoring_well_node] == [-80, -80]#make sure we are looking at the right node
        return h_gw[monitoring_well_node] - steadyhead
    end
end
#=
#not sure where this architecture came from
model = Chain(
    Conv((3, 3), 1=>8, pad=(1,1), relu),
    x -> maxpool(x, (2,2)),
    Conv((3, 3), 8=>16, pad=(1,1), relu),
    x -> maxpool(x, (2,2)),
    Conv((3, 3), 16=>8, pad=(1,1), relu),
    x -> maxpool(x, (2,2)),
    flatten,
    Dense(288, 1),
) |> f64
=#

#like LeNet5
model = Chain(Conv((5, 5), 1=>6, relu),
              MaxPool((2, 2)),
              Conv((5, 5), 6=>16, relu),
              MaxPool((2, 2)),
              flatten,
              Dense(1296, 120, relu),
              Dense(120, 84, relu),
              Dense(84, 1)) |> f64

#=
#like vgg16 model
model = Chain(Conv((3, 3), 1 => 64, relu, pad=(1, 1), stride=(1, 1)),
             BatchNorm(64),
             Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
             BatchNorm(64),
             MaxPool((2,2)),
             Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
             BatchNorm(128),
             Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
             BatchNorm(128),
             MaxPool((2,2)),
             Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
             BatchNorm(256),
             Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
             BatchNorm(256),
             Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
             BatchNorm(256),
             MaxPool((2,2)),
             Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
             BatchNorm(512),
             Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
             BatchNorm(512),
             Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
             BatchNorm(512),
             MaxPool((2,2)),
             Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
             BatchNorm(512),
             Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
             BatchNorm(512),
             Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
             BatchNorm(512),
             MaxPool((2,2)),
             flatten,
             Dense(512, 4096, relu),
             Dropout(0.5),
             Dense(4096, 4096, relu),
             Dropout(0.5),
             Dense(4096, 1)) |> gpu
=#

# Make neural network parameters trackable by Flux
θ = params(model)

#gpumodel(x) = cpu(model(gpu(x)))

function loss(x)
    Ts = reshape(hcat(map(y->y[1], x)...), size(x[1][1], 1), size(x[1][1], 2), 1, length(x))
    targets = map(y->y[2], x)
    #Q1 = gpumodel(Ts)
    Q1 = model(Ts)
    Qs = map(Q->[Q, Qinj], Q1)
    loss = sum(pmap(i->solve_numerical(Qs[i], Ts[:, :, 1, i]) - targets[i], 1:size(Ts, 4)).^2)
    return loss
end

opt = ADAM(learning_rate)

# Injection rate
Qinj = 0.031688 # [m^3/s] (1 MMT water/yr)

# Training epochs
epochs = 1:10000

# Calculate distance between extraction and injection wells
# batch 1:1 for one batch size
data_train_batch = [[(GaussianRandomFields.sample(grf), pressure_target) for i = 1:batch_size] for v in 1:16]
data_test = [[(GaussianRandomFields.sample(grf), pressure_target) for i = 1:batch_size] for i = 1:16]

println("The training has started..")
loss_train = sum(map(x->loss(x), data_train_batch))
rmse_train = sqrt(loss_train/(batch_size*length(data_train_batch)))
loss_test = sum(map(x->loss(x), data_test))
rmse_test = sqrt(loss_test/(batch_size * length(data_test)))
println(string("epoch: 0 train rmse: ", rmse_train, " test rmse: ", rmse_test))
# Save convergence metrics
push!(losses_test, loss_test)
push!(rmses_test, rmse_test)
for epoch in epochs
    data_train_batch = [[(GaussianRandomFields.sample(grf), pressure_target) for i = 1:batch_size] for v in 1:16]
    tt = @elapsed Flux.train!(loss, θ, data_train_batch, opt)
    push!(train_time, tt)
    loss_train = sum(map(x->loss(x), data_train_batch))
    rmse_train = sqrt(loss_train/(batch_size*length(data_train_batch)))
    loss_test = sum(map(x->loss(x), data_test))
    rmse_test = sqrt(loss_test/(batch_size * length(data_test)))
    # Terminal output
    println(string("epoch: ", epoch, " time: ", tt, " train rmse: ", rmse_train, " test rmse: ", rmse_test))
    # Save convergence metrics
    push!(losses_train, loss_train)
    push!(rmses_train, rmse_train)
    push!(losses_test, loss_test)
    push!(rmses_test, rmse_test)
end
println("The training has finished!")

#@save string("loss_data_10000.jld2") epochs losses_train  rmses_train  losses_test  rmses_test  train_time

@BSON.save "model_bigrun_$(batch_size)_$(learning_rate)_10k.bson" model epochs losses_train  rmses_train  losses_test  rmses_test  train_time

println("The data has beens saved!")
