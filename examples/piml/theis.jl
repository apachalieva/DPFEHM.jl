using ChainRulesCore: length
import DPFEHM
import GaussianRandomFields
import DifferentiableBackwardEuler
import Optim
import Random
import Zygote
import ChainRulesCore

using JLD2
using Statistics: mean, std

global losses_test = Float64[]
global rmses_test = Float64[]
global train_time = Float64[]

@everywhere begin
    using Flux
    using ChainRulesCore
    using Random
    using GaussianRandomFields
    #using Distributed
    #using Random
    using DPFEHM

    Random.seed!(0)
    #push!(LOAD_PATH,"/home/apachalieva/Projects/subsurface_flow/git/DPFEHM.jl_fork/src")
    #push!(LOAD_PATH,"/home/apachalieva/Projects/subsuface_flow/git/DPFEHM.jl_fork/examples/piml/")

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
    grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)

    pressure_target  = 1.0

    coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid2d(mins, maxs, ns, thickness)
    dirichletnodes = Int[]
    dirichleths = zeros(size(coords, 2))
    for i = 1:size(coords, 2)
        if abs(coords[1, i]) == sidelength || abs(coords[2, i]) == sidelength
            push!(dirichletnodes, i)
            dirichleths[i] = steadyhead
        end
    end

    function getQs(Qs::Vector, rs::Vector)
        return sum(map(i->getQs(Qs[i], rs[i]), 1:length(Qs)))
    end

    function getQs(Q::Number, r::Number)#this splits the Q across all the nodes that are as close as possible to r (should be at least 4)
        dists = map(i->(sqrt(sum(coords[:, i] .^ 2)) - r) ^ 2, 1:size(coords, 2))
        mindist = minimum(dists)
        goodnodes = (dists .≈ mindist)
        Qs = Q * goodnodes / sum(goodnodes)
        return Qs
    end

    function solve_numerical(Qs, T, rs)
        logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))
        Qs = getQs(Qs, rs)
        @assert length(T) == length(Qs)

        Ks_neighbors = logKs2Ks_neighbors(T)
        h_gw = DPFEHM.groundwater_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
        goodnode = div(size(coords, 2), 2) + 1
        @assert coords[:, goodnode] == [0, 0]#make sure we are looking at the right "good node"
        return h_gw[goodnode, end] - steadyhead
    end
end

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

# Make neural network parameters trackable by Flux
θ = params(model)

function loss(x)
    Ts = reshape(hcat(map(y->y[1], x)...), size(x[1][1], 1), size(x[1][1], 2), 1, length(x))
    targets = map(y->y[2], x)

    #@show size(Ts)
    #@show targets
   
    Q1 = model(Ts)
    Qs = map(Q->[Q, Qinj], Q1)

    # i and 1 might be swapped
    loss = sum(pmap(i->solve_numerical(Qs[i], Ts[:, :, 1, i], rs) - targets[i], 1:size(Ts, 4)).^2)
    return loss
end

function cb2()
    # Terminal output
    loss_train = sum(map(x->loss(x),data_train_batch))
    loss_test = sum(map(x->loss(x),data_test))
    println(string("callback: train loss: ", loss_train, " test loss: ", loss_test))
end

cb = Flux.throttle(cb2, 10)
opt = ADAM()

# Set ranges for P, T
# Fix pressure target for now. It should be removed, but when I removed it with MultiTheis, the PIML framework didn't work as well. Moral of the story, NN's are squirrelly things that don't always make sense. 
PRES = (0.0,0.0) # Pressure target range [m] 
TRANS = (-2.0,0.0) # log10 transmissivity [m^2/s]
# Assume that reservoir thickness is 100m (S = S_s*b)
STOR = (-4.0,-1.0) # log10 storativity

# Location to monitor pressure at
mon_well_crds = [[-75.0,-75.0]]
# Location of injection and extraction wells
well_crds = [[-50.0,-50.0],[50.0,50.0]]
# Injection rate
Qinj = 0.031688 # [m^3/s] (1 MMT water/yr)
# Training epochs
epochs = 1:100
# Calculate distance between extraction and injection wells
rs = [sqrt((sum((cs-mon_well_crds[1]).^2))) for cs in well_crds]

# Input/output normalization function
lerp(x, lo, hi) = x*(hi-lo)+lo

# Random T, S, and P generator
sample() = [10^lerp(rand(), TRANS...), 10^lerp(rand(), STOR...), lerp(rand(), PRES...)]

# batch 1:1 for one batch size
data_train_batch = [[(GaussianRandomFields.sample(grf), pressure_target) for i = 1:2] for v in 1:1000]
data_test = [[(GaussianRandomFields.sample(grf), pressure_target)] for i = 1:100]

println("The training has started..")
for epoch in epochs
    tt = @elapsed Flux.train!(loss, θ, data_train_batch, opt)#, cb = cb)
    push!(train_time, tt)
    loss_test = sum(map(x->loss(x), data_test))
    rmse_test = sqrt(loss_test/length(data_test))
    # Terminal output
    println(string("epoch: ", epoch," test rmse: ", rmse_test))
    # Save convergence metrics
    push!(losses_test, loss_test)
    push!(rmses_test, rmse_test)
end
println("The training has finished!")

@save string("loss_data_2000.jld2") epochs losses_test  rmses_test train_time
println("The data has beens saved!")
