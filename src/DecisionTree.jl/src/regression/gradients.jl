using Random
using ParallelDataTransfer
using Distributed
include("locallinear.jl")

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

function KNNGradientEstimator(x::AbstractMatrix, y::AbstractVector; subsampling=0.5, K=nothing, λ=nothing, distributed=false)
    n, d = size(x)
    selected = Random.randperm(n)[1:min(ceil(Int, subsampling*n), n)]
    if isnothing(K) && isnothing(λ)
        lola = LocalLasso(x[selected, :]', y[selected])
    elseif isnothing(K)
        K = ceil(Int, n^(4 / (4+d)))
        ks = max.(ceil.(Int, LinRange(0.2 * K, min(n, 5.0 * K), 50)), 10) |> unique
        lola = LocalLasso(x', y, ks, [λ])
    elseif isnothing(λ)
        lola = LocalLasso(x', y, [K], exp.(LinRange(-8, 1, 10)))
    else
        lola = LocalLasso(x', y, K, λ)
    end
    if distributed
        wp = CachingPool(workers())

        # This is too slow. Gradient subsampling ?

        gradients = pmap(i -> predict(lola, x[i, :])[2], wp, selected)
        gradients = reduce(hcat, gradients)
    else
        gradients = Matrix{Float64}(undef, d, n)
        # This is too slow. Gradient subsampling ?
        for i in 1:length(selected)
            gradients[:, i] = predict(lola, x[selected[i], :])[2]
        end
    end
    return KNNGradientEstimator(gradients, KDTree(x[selected, :]', Euclidean()))
end

function (m::KNNGradientEstimator)(x::AbstractVector)
    nn = knn(m.kdtree, x, 1)[1]
    return m.gradients[:, nn]
end
struct SimpleGradientEstimator <: GradientEstimator
    gradients::Dict{AbstractVector, AbstractVector}
end

function SimpleGradientEstimator(x::AbstractMatrix, y::AbstractVector; K=nothing, λ=nothing, distributed=false)
    n, d = size(x)
    if distributed
        @eval @everywhere using ParallelDataTransfer
        @passobj 1 workers() x
        @passobj 1 workers() y
        @everywhere n, d = size(x)
        if isnothing(K) && isnothing(λ)
            @everywhere lola = LocalLasso(x', y)
        elseif isnothing(K)
            @everywhere K = ceil(Int, n^(4 / (4+d)))
            @everywhere ks = max.(ceil.(Int, LinRange(0.2 * K, min(n, 5.0 * K), 50)), 10) |> unique
            @everywhere lola = LocalLasso(x', y, ks, [λ])
        elseif isnothing(λ)
            @everywhere lola = LocalLasso(x', y, [K], exp.(LinRange(-8, 1, 10)))
        else
            @everywhere lola = LocalLasso(x', y, K, λ)
        end
        # This is too slow. Gradient subsampling ?
        gradients = pmap(i -> predict(lola, x[i, :])[2], 1:n)
        gradients = zip([x[i, :] for i in 1:n], gradients)
    else
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
    end
    return SimpleGradientEstimator(gradients)
end

(m::SimpleGradientEstimator)(x::AbstractVector) = m.gradients[x]