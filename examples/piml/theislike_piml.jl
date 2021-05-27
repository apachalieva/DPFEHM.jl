import DPFEHM
@everywhere import GaussianRandomFields
import DifferentiableBackwardEuler
import Optim
#import PyPlot
import Random
import Zygote
import ChainRulesCore

@everywhere using Flux
@everywhere using ChainRulesCore
using Statistics: mean, std
using Random
using JLD2
#using TheisLike

@everywhere begin
    Random.seed!(0)

    include("theislike_piml_setup.jl")
    push!(LOAD_PATH,"/home/apachalieva/Projects/subsurface_flow/git/DPFEHM.jl_fork/src")
    push!(LOAD_PATH,"/home/apachalieva/Projects/subsuface_flow/git/DPFEHM.jl_fork/examples/piml/")

    function ChainRulesCore.rrule(::typeof(pmap), f, x)
        results_and_backs = pmap(i->Zygote.pullback(f, x[i]), 1:length(x))  
        results = map(x->x[1], results_and_backs)
        backs = map(x->x[2], results_and_backs)
        return results, delta->pmap(i->backs[i](delta[i]), 1:length(delta))
    end
    n = 51
    ns = [n, n]
    steadyhead = 0e0
    sidelength = 200
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
end

# batch 1:1 for one batch size
data_train_batch = [[(GaussianRandomFields.sample(grf), pressure_target) for i = 1:2] for v in 1:1000]
data_test = [[(GaussianRandomFields.sample(grf), pressure_target)] for i = 1:100]

println("The training has started..")

# Place to save stuff
#folder_name = "AD_results_new"
for epoch in epochs
    #println("Epoch ", epoch)
    #tt = @elapsed Flux.train!(loss, θ, data_train_batch, opt)
    tt = @elapsed Flux.train!(loss, θ, data_train_batch, opt, cb = cb)
    push!(train_time, tt)
    loss_train = sum(map(x->loss(x,train=false), data_train_batch))
    loss_test = sum(map(x->loss(x,train=false), data_test))
    rmse_train = sqrt(loss_train/length(data_train_batch))
    rmse_test = sqrt(loss_test/length(data_test))
    diffs_train = map(x->mydiff(x,train=false), data_train_batch)
    diffs_test = map(x->mydiff(x,train=false), data_test)
    meandiff_train = mean(diffs_train)
    meandiff_test = mean(diffs_test)
    stddiff_train = std(diffs_train)
    stddiff_test = std(diffs_test)
    # Terminal output
    println(string("epoch: ", epoch, " train rmse: ", rmse_train, " test rmse: ", rmse_test))
    # Save convergence metrics
    push!(losses_train, loss_train)
    push!(losses_test, loss_test)
    push!(rmses_train, rmse_train)
    push!(rmses_test, rmse_test)
    push!(meandiffs_train, meandiff_train)
    push!(meandiffs_test, meandiff_test)
    push!(stddiffs_train, stddiff_train)
    push!(stddiffs_test, stddiff_test)
    #plot_maps(epoch,display=false,folder_name=folder_name);
    #run(`convert $[string(folder_name,"/images/iters",lpad(epoch,4,"0"),".png"), string(folder_name,"/images/convergence",lpad(epoch,4,"0"),".png")] -append $[string(folder_name,"/images/combine",lpad(epoch,4,"0"),".png")]`)
end
println("The training has finished!")
#plot_losses(folder_name=folder_name, combine=true)

@save string("loss_data_2000.jld2") meandiffs_train stddiffs_train epochs losses_test losses_train meandiffs_test stddiffs_test rmses_train rmses_test train_time
println("The data has beens saved!")
#@save string(folder_name,"/data/loss_data.jld2") meandiffs_train stddiffs_train epochs losses_test losses_train meandiffs_test stddiffs_test rmses_train rmses_test train_time θ well_crds mon_well_crds

#run(`ffmpeg -y -r 5 -f image2 -s 1920x1080 -i $folder_name/images/combine%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p $folder_name/multitheisdp_1well.mp4`)
