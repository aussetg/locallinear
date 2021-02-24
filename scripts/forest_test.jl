using ClusterManagers
using Distributed

if gethostname() == "M1"
    Distributed.addprocs(32)
elseif gethostname() == "tsicluster0"
    ClusterManagers.addprocs_sge(10, qsub_env="-ac JULIA_NUM_THREADS=8")
end

using DataFrames
using CSV
using MLBase
using StatsBase
using StatsModels
using MAT
using BSON
using BSON: @save, @load
using Dates
using ProgressMeter
@everywhere using DecisionTree

include(joinpath(@__DIR__, "edr.jl"))

n_subfeatures=-1; n_trees=100; partial_sampling=0.7; max_depth=30
min_samples_leaf=5; min_samples_split=2; min_purity_increase=0.0

function crossval(X, Y)
    n = size(X, 1)
    results = DataFrame(method=[], loss=[])
    @showprogress for idx in Kfold(n, 50) |> collect
        X_train = X[idx, :]
        Y_train = Y[idx]
    
        dt = fit(ZScoreTransform, X_train, dims=1)
        X_train = StatsBase.transform(dt, X_train)
    
        X_test = X[setdiff(1:n, idx), :]
        X_test = StatsBase.transform(dt, X_test)
        Y_test = Y[setdiff(1:n, idx)]
    
        forest = build_forest(Y_train, X_train,
        n_subfeatures,
        n_trees,
        partial_sampling,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase)
        # We will set subsampling to linit ourselves to 2000 examples for gradients approx
        subsampling = min(5000 / size(X_train, 1), 1)
    
        gradient = DecisionTree.treeregressor.KNNGradientEstimator(X_train, Y_train; subsampling=subsampling, distributed=true)
    
        forest_grad = build_forest(Y_train, X_train,
        n_subfeatures,
        n_trees,
        partial_sampling,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase; gradient = gradient)
    
        res = ( apply_forest(forest, X_test) .- Y_test) .^ 2 |> mean
        res_grad = ( apply_forest(forest_grad, X_test) .- Y_test) .^ 2 |> mean
        results = vcat(results, DataFrame(method=[:standard, :gradient], loss=[res, res_grad]))
    end
    return results
end

function swissroll()
    X, Y = make_locally_sparse_swiss_roll(2000; noise=0.1)
    return X, Y
end

function wisconsin()
    file = joinpath(@__DIR__, "../data/breast-cancer-wisconsin.csv")
    filepath = joinpath(@__DIR__, file)
    df = CSV.read(filepath; silencewarnings=true)
    binarize(x) = x == "M" ? 1 : 0
    df.target = binarize.(df[:, :diagnosis])
    dfx = select(df, Not([:Column33, :id, :target, :diagnosis]))
    dfy = select(df, :target)
    return Array(dfx), Vector{Float64}(Array(dfy) |> vec)
end

function heart()
    file = joinpath(@__DIR__, "../data/heart-disease-uci.csv")
    filepath = joinpath(@__DIR__, file)
    df = CSV.read(filepath; silencewarnings=true)
    dfx = select(df, Not([:target]))
    dfy = select(df, :target)
    return Array(dfx), Vector{Float64}(Array(dfy) |> vec)
end

function diamonds()
    file = joinpath(@__DIR__, "../data/diamonds.csv")
    filepath = joinpath(@__DIR__, file)
    df = CSV.read(filepath; silencewarnings=true)
    dfx = modelmatrix(@formula(price ~ carat + cut + color + clarity + depth + table + x + y + z), df)
    dfy = select(df, :price)
    return Array(dfx), Vector{Float64}(Array(dfy) |> vec)
end

function gasoline()
    file = joinpath(@__DIR__, "../data/gasoline.mat")
    filepath = joinpath(@__DIR__, file)
    vars = matread(filepath)
    dfx = vars["NIR"]
    dfy = vars["octane"]
    return Array(dfx), Vector{Float64}(Array(dfy) |> vec)
end

function sdss()
    file = joinpath(@__DIR__, "../data/Skyserver_SQL2_27_2018 6_51_39 PM.csv")
    filepath = joinpath(@__DIR__, file)
    df = CSV.read(filepath; silencewarnings=true)
    binarize(x) = x == "GALAXY" ? 1 : 0
    df.target = binarize.(df[:, :class])
    dfx = df[!, [:ra, :dec, :u, :g, :r, :i, :z, :redshift]]
    dfy = df[!, :target]
    return Array(dfx), Vector{Float64}(Array(dfy) |> vec) 
end

function main()
    @showprogress for data in [sdss]#[wisconsin, heart, diamonds, gasoline, swissroll, sdss]
        X, Y = data()
        results = crossval(X, Y) 
        @save "results_$(string(data))_$(now()).bson" results
    end
end

main()