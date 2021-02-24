# With KNN
using NearestNeighbors
using GLMNet

wlinreg(x, y, w) = (x * Diagonal(w))' \ (Diagonal(w) * reshape(y, (size(y, 1), 1)))

struct LocalLasso{T <: Real, L <: Real}
    X :: AbstractMatrix{T}
    Y :: AbstractVector{L}
    kdtree :: KDTree
    K :: Vector{Int}
    λ :: Vector{Float64}
    function LocalLasso(X::AbstractMatrix{T}, Y::AbstractVector{L}) where {T <: Real, L <: Real}
        d, N = size(X)
        K = ceil(Int, N^(4 / (4+d)))
        ks = max.(ceil.(Int, LinRange(0.2 * K, min(N, 5.0 * K), 50)), 10) |> unique
        new{T, L}(X, Y, KDTree(X, Euclidean()), ks, exp.(LinRange(-8, 1, 10)))
    end
    LocalLasso(X::AbstractMatrix{T}, Y::AbstractVector{L}, K::Int, λ::Float64) where {T <: Real, L <: Real} = new{T, L}(X, Y, KDTree(X, Euclidean()), [K], [λ])
    LocalLasso(X::AbstractMatrix{T}, Y::AbstractVector{L}, K::Vector{Int}, λ::Vector{Float64}) where {T <: Real, L <: Real} = new{T, L}(X, Y, KDTree(X, Euclidean()), K, λ)
end

struct LocalLassoPlus{T <: Real, L <: Real}
    lola :: LocalLasso{T, L}
    LocalLassoPlus(X::AbstractMatrix{T}, Y::AbstractVector{L}) where {T <: Real, L <: Real} = new{T, L}(LocalLasso(X, Y))
    LocalLassoPlus(X::AbstractMatrix{T}, Y::AbstractVector{L}, k, λ) where {T <: Real, L <: Real} = new{T, L}(LocalLasso(X, Y, k, λ))
end


struct LocalConstant{T <: Real, L <: Real}
    X :: AbstractMatrix{T}
    Y :: AbstractVector{L}
    kdtree :: KDTree
    K :: Vector{Int}
    function LocalConstant(X::AbstractMatrix{T}, Y::AbstractVector{L}) where {T <: Real, L <: Real}
        d, N = size(X)
        K = ceil(Int, N^(2 / (2+d)))
        ks = max.(ceil.(Int, LinRange(0.2 * K, min(N, 5.0 * K), 50)), 10) |> unique
        new{T, L}(X, Y, KDTree(X, Euclidean()), ks)
    end
    LocalConstant(X::AbstractMatrix{T}, Y::AbstractVector{L}, K::Int) where {T <: Real, L <: Real} = new{T, L}(X, Y, KDTree(X, Euclidean()), [K])
    LocalConstant(X::AbstractMatrix{T}, Y::AbstractVector{L}, K::Vector{Int}) where {T <: Real, L <: Real} = new{T, L}(X, Y, KDTree(X, Euclidean()), K)
end

struct LocalLinear{T <: Real, L <: Real}
    X :: Matrix{T}
    Y :: Vector{L}
    kdtree :: KDTree
    K :: Vector{Int}
    function LocalLinear(X::AbstractMatrix{T}, Y::AbstractVector{L}) where {T <: Real, L <: Real}
        d, N = size(X)
        K = ceil(Int, N^(4 / (4+d)))
        ks = max.(ceil.(Int, LinRange(0.2 * K, min(N, 5.0 * K), 50)), 10) |> unique
        new{T, L}(X, Y, KDTree(X, Euclidean()), ks)
    end
    LocalLinear(X::AbstractMatrix{T}, Y::AbstractVector{L}, K::Int) where {T <: Real, L <: Real} = new{T, L}(X, Y, KDTree(X, Euclidean()), [K])
    LocalLinear(X::AbstractMatrix{T}, Y::AbstractVector{L}, K::Vector{Int}) where {T <: Real, L <: Real} = new{T, L}(X, Y, KDTree(X, Euclidean()), K)
end

function cv_local(X, Y, x, f, ks, lambdas, kdtree::KDTree)
    # Check if CV is needed
    if length(ks) > 1 || length(lambdas) > 1
        # Select 100 nearest neighbours of x
        nns, _ = knn(kdtree, x, ks[1])
        # We will do a LoO on Ks
        results = Dict{Tuple{Int, Float64}, Float64}()
        for k in ks, λ in lambdas
            errors = zeros(length(nns))
            # KNN is not thread safe so let's do it beforehand
            idxs = [filter!(x -> x != nn, knn(kdtree, X[:, nn], min(length(Y), k+1))[1]) for nn in nns]
            # HOT LOOP. Optimize
            Threads.@threads for i in 1:length(idxs)
                a, _ = f((X .- X[:, nns[i]])[:, idxs[i]], Y[idxs[i]], λ)
                errors[i] = (Y[nns[i]] - a)^2
            end
            results[(k, λ)] = sum(errors) / length(errors)
        end
        # Select the best 
        (k, λ) = sort(results |> collect, by=x -> x[2])[1][1]
    else
        (k, λ) = ks[1], lambdas[1]
    end
    # Train 
    idxs, _ = knn(kdtree, x, k)
    a, b = f((X .- x)[:, idxs], Y[idxs], λ) 
    return a, b, k, λ
end

function fit_lll(X, Y, λ)
    cv = glmnet(X', Y; intercept=true, lambda=[λ])
    return cv.a0[1], Array{Float32, 2}(cv.betas) |> vec
end

function fit_ll(X, Y, λ)
    res = wlinreg(vcat(ones((1, size(X, 2))), X), Y, ones(size(Y)))
    return res[1], res[2:end]
end

function fit_lc(X, Y, λ)
    res = wlinreg(ones((1, size(X, 2))), Y, ones(size(Y)))
    return res[1], nothing
end

# [TODO] Reprogram the CV to be LoO
function predict(lola::LocalLasso, x)
    d, n = size(lola.X)
    K = n^(4 / (4+d))
    ks = min.(max.(ceil.(Int, K .* [0.5,  1., 2., 5., 10., 20., 50.]), 5), n) |> unique
    # We first get a list of potential lambdas ([TODO] how does glmnet build it ?)
    nns, _ = knn(lola.kdtree, x, ks[end], true)
    results = Dict{Tuple{Int, Float64}, Float64}()
    lambda = glmnet((lola.X .- x)[:, nns]', lola.Y[nns]).lambda
    for k in ks
        losses = zeros(Float64, length(lambda))
        for nn in nns[1:20] # [TODO] We need to check we at least have 20...
            glm = glmnet((lola.X .- lola.X[:, nn])[:, [i for i in nns if i != nn]]', lola.Y[[i for i in nns if i != nn]], lambda=lambda)
            losses += (glm.a0 .- lola.Y[nn]).^2 ./ length(20)
        end
        m = argmin(losses)
        results[(k, lambda[m])] = losses[m]
    end
    (k, λ) = sort(results |> collect, by=x -> x[2])[1][1]
    idxs, _ = knn(lola.kdtree, x, k)
    glm = glmnet((lola.X .- x)[:, idxs]', lola.Y[idxs], lambda=[λ])
    return glm.a0[1], glm.betas[:, 1], k, λ
end

function predict(d::LocalLassoPlus, x; k = nothing)
    a, b, k, λ = isnothing(k) ? cv_local(d.lola.X, d.lola.Y, x, fit_lll, d.lola.K, d.lola.λ, d.lola.kdtree) : cv_local(d.lola.X, d.lola.Y, x, fit_lll, k, d.lola.λ, d.lola.kdtree)
    dropped = isapprox.(0., b; atol=10e-4)
    if count(dropped) > 0 && count(dropped) < length(x)
        next = LocalLassoPlus(d.lola.X[.!dropped, :], d.lola.Y, d.lola.K, d.lola.λ)
        a, b, k, λ = predict(next, x[.!dropped]; k = k)
        full_b  = zeros(length(x))
        full_b[.!dropped] .= b
        return a, full_b, k, λ
    else
        return a, b, k, λ
    end
end

predict(d::LocalLinear, x; k = nothing) = isnothing(k) ? cv_local(d.X, d.Y, x, fit_ll, d.K, [0.], d.kdtree) : cv_local(d.X, d.Y, x, fit_lll, k, [0.], d.kdtree)

predict(d::LocalConstant, x; k = nothing) = isnothing(k) ? cv_local(d.X, d.Y, x, fit_lc, d.K, [0.], d.kdtree) : cv_local(d.X, d.Y, x, fit_lll, k, [0.], d.kdtree)
