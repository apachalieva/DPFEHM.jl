using Flux
using Statistics: mean, std
using Random
using Distributed
using ChainRulesCore
#import MultiTheisDP.MultiTheis
#using PyPlot
#ioff()
#using PyCall
#gridspec = pyimport("matplotlib.gridspec")
#using LaTeXStrings
using JLD2
#include("plot_train.jl")
#include("theislike.jl")
push!(LOAD_PATH,"../../src")
push!(LOAD_PATH,"./")

########################################################
# User input
n = 51
ns = [n, n]
steadyhead = 0e0
sidelength = 200
thickness  = 1.0
mins = [-sidelength, -sidelength] #meters
maxs = [sidelength, sidelength] #meters
sigma = 1.0
lambda = 50
num_eigenvectors = 200
coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid2d(mins, maxs, ns, thickness)
dirichletnodes = Int[]
dirichleths = zeros(size(coords, 2))
for i = 1:size(coords, 2)
  if abs(coords[1, i]) == sidelength || abs(coords[2, i]) == sidelength
  push!(dirichletnodes, i)
  dirichleths[i] = steadyhead
  end
end

# Location to monitor pressure at
@everywhere mon_well_crds = [[-75.0,-75.0]]
# Location of injection and extraction wells
@everywhere well_crds = [[-50.0,-50.0],[50.0,50.0]]
# Injection rate
@everywhere Qinj = 0.031688 # [m^3/s] (1 MMT water/yr)
# Time to check pressure
#t = 30.0*86400.0 # 30 days converted to seconds
# Create xy grid for plotting pressures
#xs = ys = -100.0:1.0:100.0
# Training epochs
epochs = 1:100
########################################################

#@everywhere begin
#  
#end
# Calculate distance between extraction and injection wells
@everywhere rs = [sqrt((sum((cs-mon_well_crds[1]).^2))) for cs in well_crds]

# Input/output normalization function
lerp(x, lo, hi) = x*(hi-lo)+lo

@everywhere global nevals = 0
# Wrapper around MultiTheis solve returning pressure head at a point
@everywhere function head(T::Array{Float64,2},Q1::Float64,Qinj::Float64,rs::Array{Float64,1};train=false)
	Qs = [Q1,Qinj]
	if train
		global nevals = nevals + 1
	end
  num_res = TheisLike.solve_numerical(Qs, T, rs)
end

# Set random seed for repeatability
@everywhere Random.seed!(0)

@everywhere model = Chain(
    Conv((3, 3), 1=>8, pad=(1,1), relu), 
    x -> maxpool(x, (2,2)),

    Conv((3, 3), 8=>16, pad=(1,1), relu),
    x -> maxpool(x, (2,2)),

    Conv((3, 3), 16=>8, pad=(1,1), relu),
    x -> maxpool(x, (2,2)),

    flatten,
    Dense(288, 1),
) |> f64


#print(model)
# Make neural network parameters trackable by Flux
@everywhere θ = params(model)

# Neural network function (T, S, target)->(Q1,Q2)
# 'target' is the target pressure head at the monitoring location
@everywhere function aim(T)
 
  T = Flux.unsqueeze(T, 3)  
  T = Flux.unsqueeze(T, 3)  
  Q1 = model(T)

  # Make sure that Q1 and Q2 are always negative for extraction
  Q1 = Q1[1]
  #Q2 = -softplus(Q2)
  Q1
end

# Set ranges for P, T
# Fix pressure target for now. It should be removed, but when I removed it with MultiTheis, the PIML framework didn't work as well. Moral of the story, NN's are squirrelly things that don't always make sense. 
PRES = (0.0,0.0) # Pressure target range [m] 
TRANS = (-2.0,0.0) # log10 transmissivity [m^2/s]
# Assume that reservoir thickness is 100m (S = S_s*b)
STOR = (-4.0,-1.0) # log10 storativity

# Random T, S, and P generator
sample() = [10^lerp(rand(), TRANS...), 10^lerp(rand(), STOR...), lerp(rand(), PRES...)]

# Physics model vs NN 
@everywhere function mydiff(T, target; train=false)
    @show myid()
    (head(T, aim(T), Qinj, rs, train=train) - target)
end
@everywhere function loss(T, target; train=true)
	mydiff(T, target, train=train)^2
end

#loss(x; train=true) = sum(pmap(x->loss(x[1], x[2]; train=train), 1:nworkers(), x) )
#mydiff(x; train=true) = sum(map(x->mydiff(x[1], x[2]; train=train), 1:nworkers(), x) 

function ChainRulesCore.rrule(::typeof(pmap), f, x)
  results_and_backs = pmap(i->Zygote.pullback(f, x[i]), 1:length(x))  
  results = map(x->x[1], results_and_backs)
  backs = map(x->x[2], results_and_backs)
  return results, delta->pmap(i->backs[i](delta[i]), 1:length(delta))
end

loss(x; train=true) = sum(pmap(x->loss(x[1], x[2]; train=train), x) )
mydiff(x; train=true) = sum(pmap(x->mydiff(x[1], x[2]; train=train), x) )


function cb2()
	# Terminal output
	loss_train = sum(map(x->loss(x,train=false),data_train_batch))
	loss_test = sum(map(x->loss(x,train=false),data_test))
	println(string("callback: train loss: ", loss_train, " test loss: ", loss_test))
end

global losses_train = Float64[]
global losses_test = Float64[]
global rmses_train = Float64[]
global rmses_test = Float64[]
global meandiffs_train = Float64[]
global stddiffs_train = Float64[]
global meandiffs_test = Float64[]
global stddiffs_test = Float64[]
global train_time = Float64[]

cb = Flux.throttle(cb2, 10)

opt = ADAM()
