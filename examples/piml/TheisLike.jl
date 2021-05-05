import Zygote

module TheisLike
	#push!(LOAD_PATH,"/Users/dharp/source/DPFEHM.jl/src")
	import GaussianRandomFields
	using GaussianRandomFields
	using Random
	import DifferentiableBackwardEuler
	import DPFEHM

	include("theislike_piml_setup.jl")

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
