# Implement Wang Zeroth Order
using GLMNet
using Convex, SCS
import LinearAlgebra: dot, norm
import Zygote: gradient

function wang(f, x, n, δ, λ; debiased = true, doublepenalty = false)
    # Alg 1
    d = size(x, 1)
    z = rand((-1, 1), (d, n))
    y = f.(eachcol(x .+ δ .* z)) ./ δ
    # Solve 3
    # Why does intercept = false model doesn't work ?
    if doublepenalty
        X = [z; ones(1, n)]
        lasso = glmnet(X', y; intercept = false, lambda=[λ])
        β = Array{Float32, 2}(lasso.betas) |> vec
        g, μ = β[1:(end-1)], β[end]
    else
        lasso = glmnet(z', y; intercept = true, lambda=[λ])
        g = Array{Float32, 2}(lasso.betas) |> vec
        μ = lasso.a0[1]
    end
    # Debiasing 
    if debiased
        g .= g.+ 1. / n * z * (y - z' * g .- μ)
    end
    return g, μ
end

function mirrordescent(f, ∇f, x_0, B, T, n, ψ, η)
    # Note that we do not debiase as in the original paper for a more comparable method to ours. Debiasing can also be applied to our method.
    xs = [x_0]
    Tp = floor(T/(2*n))
    ∇ψ(x) = gradient(ψ, x)[1]
    for t in 2:Tp
        x = xs[end]
        lola = LocalLasso(X, Y)
        _, g, _, _ = predict(lola, x)
        # Solve the strongly convex problem
        xnext = Variable(size(x, 1))
        problem = minimize(η*dot(g, xnext) + ψ(xnext) - ψ(x) - dot(∇ψ(x), xnext - x), norm(xnext, 1) <= B)
        solve!(problem, () -> SCS.Optimizer(verbose=false))
        push!(xs, xnext.value |> vec)
    end
    return xs
end

mirrordescent_wang(f, x_0, B, T, n, ψ, η, δ, λ; debiased = true, doublepenalty = false) = mirrordescent(f, (f, x, n) -> wang(f, x, n, δ, λ; debiased = debiased, doublepenalty = doublepenalty), x_0, B, T, n, ψ, η)

ψ(x) = 1. / 2. * square(norm(x, 2))
ψ(x::T) where T <: Real = 1. / 2. * sum(x .^ 2)
ψ(x::Vector{T}) where T <: Real = 1. / 2. * sum(x .^ 2)