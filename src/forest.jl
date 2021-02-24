import Base: length, convert, promote_rule, show, start, next, done
using Statistics: mean
using StatsBase: sample, shuffle, weights
using Random: randperm
using ProgressMeter
using MultivariateStats
using NearestNeighbors
using LinearAlgebra: svd

include(joinpath(@__DIR__, "locallinear.jl"))

function softmax(xs::AbstractVector)
    max_ = maximum(xs)
    exp_ = exp.(xs .- max_)
    exp_ ./ sum(exp_)
end

expt(x::AbstractVector, t) = max.((1. .+ (1. - t) .* x), 0.).^(1. / (1. - t))

function softmax(xs::AbstractVector, t)
    max_ = maximum(xs)
    exp_ = expt(xs .- max_, t)
    exp_ ./ sum(exp_)
end

rescale(xs::AbstractVector) = xs ./ sum(xs)

struct Leaf
    pred::Float64
    size::Int
end

abstract type Node end

struct SNode <: Node
    featid::Int
    featval::Float64
    left::Union{Leaf, Node}
    right::Union{Leaf, Node}
end

# With Projection
struct PNode <: Node
    featid::Int
    featval::Float64
    left::Union{Leaf, Node}
    right::Union{Leaf, Node}
    Π::AbstractMatrix
end

const Tree = Union{Node, Leaf}

predict(tree::Leaf, x::AbstractVector) = tree.pred
function predict(tree::SNode, x::AbstractVector) 
    if x[tree.featid] <= tree.featval
        return predict(tree.left, x) 
    else
        return predict(tree.right, x) 
    end
end

function predict(tree::PNode, x::AbstractVector) 
    if (tree.Π * x)[tree.featid] <= tree.featval
        return predict(tree.left, x) 
    else
        return predict(tree.right, x) 
    end
end

function predict(tree::Tree, x::AbstractMatrix)
    preds = Vector{Float64}(undef, size(x, 1))
    Threads.@threads for i in 1:size(x, 1)
        preds[i] =  predict(tree, x[i, :])
    end
    return preds
end

function predict(forest::Vector{Tree}, x::AbstractArray)
    preds = mean(predict.(forest, Ref(x)))
end

function l2split(x::AbstractVector, y::AbstractVector; fast = false)
    temp = sort(x) |> unique
    candidates = (temp[2:end] .+ temp[1:end-1])./2
    n = length(candidates)
    candidates = fast ? shuffle(candidates)[1:floor(Int, sqrt(n))] : candidates
    ncandidates = length(candidates)
    losses = Vector{Float64}(undef, ncandidates)
    if !isempty(losses)
        Threads.@threads for i in 1:ncandidates
            left = findall(k -> x[k] < candidates[i], 1:n)
            right = findall(k -> x[k] >= candidates[i], 1:n)
            losses[i] = sum((y[left] .- mean(y[left])).^2) + sum((y[right] .- mean(y[right])).^2)
        end
        k = argmin(losses)
        return candidates[k], losses[k]
    else
        # Cannot split
        return temp[1], sum((y .- mean(y)).^2)
    end
end

abstract type GradientEstimator end

struct DummyGradient <: GradientEstimator
end
function (m::DummyGradient)(x::AbstractVector)
    throw("Please provide a gradient estimator of type GradientEstimator") 
end

function (m::GradientEstimator)(x::AbstractVector)
    return zeros(eltype(x), length(x))
end

function (m::GradientEstimator)(x::AbstractArray)
    return m.(eachrow(x))
end

struct KNNGradientEstimator <: GradientEstimator
    gradients::AbstractMatrix
    kdtree::KDTree
end

function KNNGradientEstimator(x::AbstractMatrix, y::AbstractVector; subsampling=0.5, K=nothing, λ=nothing)
    n, d = size(x)
    selected = randperm(n)[1:min(ceil(Int, subsampling*n), n)]
    K = ceil(Int, (subsampling*n)^(4 / (4+d)))
    lola = LocalLasso(x[selected, :]', y[selected], [K], exp.(LinRange(-8, 1, 10)))
    gradients = Matrix{Float64}(undef, d, n)
    # This is too slow. Gradient subsampling ?
    Threads.@threads for i in 1:length(selected)
        gradients[:, i] = predict(lola, x[selected[i], :])[2]
    end
    return KNNGradientEstimator(gradients, lola.kdtree)
end

function (m::KNNGradientEstimator)(x::AbstractVector)
    nn = knn(m.kdtree, x, 1)[1]
    return m.gradients[:, nn]
end

struct SimpleGradientEstimator <: GradientEstimator
    gradients::Dict{AbstractVector, AbstractVector}
end

function SimpleGradientEstimator(x::AbstractMatrix, y::AbstractVector; K=nothing, λ=nothing)
    n, d = size(x)
    if isnothing(K) && isnothing(λ)
        lola = LocalLasso(x', y)
    elseif isnothing(K)
        K = ceil(Int, n^(4 / (4+d)))
        ks = max.(ceil.(Int, LinRange(0.2 * K, min(n, 5.0 * K), 50)), 10) |> unique
        lola = LocalLasso(x', y, ks, [λ])
    elseif isnothing(λ)
        lola = LocalLasso(x', y, [K], exp.(LinRange(-8, 1, 10)))
    else
        lola = LocalLasso(x', y, K, λ)
    end
    gradients = Dict{AbstractVector, AbstractVector}()
    # This is too slow. Gradient subsampling ?
    for i in eachrow(x)
        gradients[i] = predict(lola, i)[2]
    end
    return SimpleGradientEstimator(gradients)
end

(m::SimpleGradientEstimator)(x::AbstractVector) = m.gradients[x]


function grow_tree(x::AbstractMatrix, y::AbstractVector; sampling = :Equal, fast = false, max_depth = 5, gradient = DummyGradient(), fw=x -> softmax(x, 3.))
    n, d = size(x)
    if max_depth <= 1 || n <= 5
        return Leaf(mean(y), n)
    else
        if sampling == :Gradient
            if isnothing(gradient)
                gradient = KNNGradientEstimator(x, y)
            end
            mgrad = mean(map(l -> abs.(l), gradient(x)))
            # if any(mgrad .≈ 0.)
            #     w = mgrad .+ eps(Float64) |> weights
            # else
            #     w = mgrad |> weights
            # end
            w = (mgrad.+0.001) |> vec |> fw |> weights
            nfeats = floor(Int, sqrt(d))
            if count(w .≈ 0.) > d - nfeats
                w = ones(eltype(w), length(w)) |> weights
            end
            feats = sample(1:d, w, nfeats; replace=false)
            splits = Vector{Tuple{Float64, Float64}}(undef, nfeats)
            Threads.@threads for i in 1:nfeats
                splits[i] = l2split(x[:, feats[i]], y)
            end
        elseif sampling == :Projection
            if isnothing(gradient)
                gradient = KNNGradientEstimator(x, y)
            end
            gradients = gradient(x)
            #M = sum(gradients .* transpose.(gradients))
            #pca = fit(PCA, M)
            #Π = projection(pca)
            #w = weights(principalvars(pca))
            SVD = hcat(gradients...)' |> svd
            Π = SVD.V
            w = (SVD.S.+0.001) |> vec |> fw |> weights
            d = length(w)
            nfeats = floor(Int, sqrt(d))
            splits = Vector{Tuple{Float64, Float64}}(undef, nfeats)
            feats = sample(1:d, w, nfeats; replace=false)
            Πx = x*Π
            Threads.@threads for i in 1:nfeats
                splits[i] = l2split(Πx[:, feats[i]], y)
            end
        else
            nfeats = floor(Int, sqrt(d))
            splits = Vector{Tuple{Float64, Float64}}(undef, nfeats)
            feats = randperm(d)[1:nfeats]
            Threads.@threads for i in 1:nfeats
                splits[i] = l2split(x[:, feats[i]], y)
            end
        end
        id = sortperm(splits, by=(x -> x[2]))[1]
        featid = feats[id]
        featval, loss = splits[id]
        left = findall(k -> x[k, featid] < featval, 1:n)
        right = findall(k -> x[k, featid] >= featval, 1:n)
        if length(left) <= 5 || length(right) <= 5
            return grow_tree(x, y, sampling = sampling, fast = fast, max_depth = max_depth-1, gradient = gradient, fw=fw)
        elseif sampling == :Projection
            return PNode(featid,
                        featval,
                        grow_tree(x[left, :], y[left], sampling = sampling, fast = fast, max_depth = max_depth-1, gradient = gradient, fw=fw),
                        grow_tree(x[right, :], y[right], sampling = sampling, fast = fast, max_depth = max_depth-1, gradient = gradient, fw=fw),
                        Π
                    )
        else
            return SNode(featid,
                        featval,
                        grow_tree(x[left, :], y[left], sampling = sampling, fast = fast, max_depth = max_depth-1, gradient = gradient, fw=fw),
                        grow_tree(x[right, :], y[right], sampling = sampling, fast = fast, max_depth = max_depth-1, gradient = gradient, fw=fw)
                    )
        end
    end
end

function grow_forest(x::AbstractMatrix, y::AbstractVector; ntrees=100, sampling = :Equal, fast = false, max_depth = 5, gradient = DummyGradient(), show_progress=true, fw=x -> softmax(x, 3.))
    if show_progress
        p = Progress(ntrees)
    end
    forest = Vector{Tree}(undef, ntrees)
    n, d = size(x)
    Threads.@threads for i in 1:ntrees
        idx = sample(1:n, n, replace=true) |> unique
        forest[i] = grow_tree(x[idx, :], y[idx]; sampling = sampling, fast = fast, max_depth = max_depth, gradient = gradient, fw=fw)
        if show_progress
            next!(p)
        end
    end
    return forest
end