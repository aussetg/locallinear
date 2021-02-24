using DrWatson
@quickactivate "locallinear"
using Distributions
using LinearAlgebra: I, dot
using Zygote: gradient
using PGFPlotsX
using ProgressMeter
using StatsBase
using StatsModels
using Random
using TickTock
include(srcdir("DecisionTree.jl", "src/regression/locallinear.jl"))
include(srcdir("wang.jl"))

f_d(x) = sum(100 * (x[2:end] - x[1:(end-1)].^2).^2 + (x[1:(end-1)] .- 1.0).^2) # + randn()
f_s(x) = sum((x[2:20] - x[1:(20-1)].^2).^2 + (x[1:(20-1)] .- 1.0).^2) #+ 0.1 * randn()

tf_d(x) = sum(100 * (x[2:end] - x[1:(end-1)].^2).^2 + (x[1:(end-1)] .- 1.0).^2) 
tf_s(x) = sum((x[2:20] - x[1:(20-1)].^2).^2 + (x[1:(20-1)] .- 1.0).^2)

f = f_s
tf = tf_s

d = 50

mutable struct Nesterov
    eta::Float64
    rho::Float64
    velocity::Vector{Float64}
end
  
Nesterov(η = 0.001, ρ = 0.9) = Nesterov(η, ρ, Vector{Float64}())

function apply!(o::Nesterov, Δ)
    η, ρ = o.eta, o.rho
    v = isempty(o.velocity) ? zero(Δ) : o.velocity
    d = @. ρ^2 * v - (1+ρ) * η * Δ
    o.velocity = ρ*v - η*Δ
    Δ = -d
end

opt = Nesterov(0.0001, 0.9)
x = 2 .* ones(d)
X = rand(MvNormal(x, 0.01 * I), 50)
Y = f.(eachcol(X))
xs = [x]
times_ours = [0.]
tick()
for i in 1:100
    lola = LocalLasso(X, Y)
    _, Δ, _, _ = predict(lola, xs[end])
    Δ = apply!(opt, clamp.(Δ, -1000.f0, 1000.f0))

    temp = rand(MvNormal(xs[end] - Δ, 0.01 * I), 50)
#=     selected = abs.(Δ) .> 0.
    dirs = rand(MvNormal(zeros(count(selected)), 0.01 * I), 50)
    
    temp = repeat(xs[end] - Δ, outer=(1, 50))
    temp[selected, :] .+= dirs =#

    ftemp = f.(eachcol(temp))
    
    m = argmin(ftemp)
    x = temp[:, m]
    push!(xs, x)
    global Y = vcat(ftemp, Y)
    global X = hcat(temp, X)
    push!(times_ours, peektimer())
end
tock()

opt_3 = Nesterov(0.0001, 0.9)
x = 2 .* ones(d)
xs_fan = [x]
times_fan = [0.]
tick()
for i in 1:100
    X = rand(MvNormal(xs_fan[end], 0.01 * I), 50)
    Y = f.(eachcol(X))
    lola = LocalLasso(X, Y, 50, 0.)
    _, Δ, _, _ = predict(lola, xs[end])
    Δ = apply!(opt_3, clamp.(Δ, -1000.f0, 1000.f0))
    push!(xs_fan, xs_fan[end] - Δ)
    push!(times_fan, peektimer())
end
tock()

opt_2 = Nesterov(0.0001, 0.9)
x = 2 .* ones(d)
xs_grad = [x]
times_grad = [0.]
tick()
for i in 1:100
    Δ = gradient(f, xs_grad[end])[1]
    Δ = apply!(opt_2, clamp.(Δ, -1000.f0, 1000.f0))
    push!(xs_grad, xs_grad[end] - Δ)
    push!(times_grad, peektimer())
end
tock()

using Optim
sol = optimize(f, x, NelderMead(), Optim.Options(iterations=5000, store_trace=true))

temp = map(x -> x.value, sol.trace)[1:50:end]
nm = zeros(101)
nm[1:length(temp)] .= temp
nm[(length(temp)+1):end] .= temp[end]

opt = Nesterov(0.0001, 0.9)
x = 2 .* ones(d)
xs_wang = [x]
times_wang = [0.]
tick()
for i in 1:100000
    Δ, μ = wang(f, xs_wang[end], 50, 0.01, 1.; debiased=true, doublepenalty=false)
    Δ = apply!(opt, clamp.(Δ, -1000.f0, 1000.f0))
    push!(xs_wang, xs_wang[end]-Δ)
    push!(times_wang, peektimer())
end
tock()

xs_md_wang = mirrordescent_wang(f, 2 .* ones(d), 1, 5000, 25, ψ, 0.001, 0.01, 1.; debiased = true, doublepenalty = false)

xs_md_ours = [2 .* ones(d)]
X = rand(MvNormal(x, 0.001 * I), 50)
Y = f.(eachcol(X))
Tp = floor(5000/(2*25))
η = 0.001
B = 1
∇ψ(x) = gradient(ψ, x)[1]
for t in 2:Tp
    x = xs_md_ours[end]
    lola = LocalLasso(X, Y)
    _, g, _, _ = predict(lola, x)
    # Solve the strongly convex problem
    xnext = Variable(size(x, 1))
    problem = minimize(η*dot(g, xnext) + ψ(xnext) - ψ(x) - dot(∇ψ(x), xnext - x), norm(xnext, 1) <= B)
    solve!(problem, () -> SCS.Optimizer(verbose=false))
    push!(xs_md_ours, xnext.value |> vec)
    temp = rand(MvNormal(xnext.value |> vec, 0.001 * I), 50)
    ftemp = f.(eachcol(temp))
    global Y = vcat(ftemp, Y)
    global X = hcat(temp, X)
end

xs_md_fan = [2 .* ones(d)]
X = rand(MvNormal(x, 0.001 * I), 50)
Y = f.(eachcol(X))
Tp = floor(5000/(2*25))
η = 0.001
B = 1
∇ψ(x) = gradient(ψ, x)[1]
for t in 2:Tp
    x = xs_md_fan[end]
    X = rand(MvNormal(x, 0.001 * I), 50)
    Y = f.(eachcol(X))
    lola = LocalLasso(X, Y, 50, 0.)
    _, g, _, _ = predict(lola, x)
    # Solve the strongly convex problem
    xnext = Variable(size(x, 1))
    problem = minimize(η*dot(g, xnext) + ψ(xnext) - ψ(x) - dot(∇ψ(x), xnext - x), norm(xnext, 1) <= B)
    solve!(problem, () -> SCS.Optimizer(verbose=false))
    push!(xs_md_fan, xnext.value |> vec)
end

rosenbrock = @pgf TikzPicture(
    PGFPlotsX.Axis(
            {
            xlabel=raw"\# function evaluations",
            ylabel=raw"$f(x)$",
            width="8cm",
            height="6cm"
            },
            Legend(["Ours", "Wang (2018)", "Fan (1992)"]),
            PlotInc({ no_marks },
                Table(; x = 0:50:5000, y = tf.(xs))
            ),
            # PlotInc({ no_marks },
            #     Table(; x = 0:100, y = nm)
            #     ),
            PlotInc({ no_marks },
                Table(; x = 0:50:5000, y = tf.(xs_wang))
            ),
            PlotInc({ no_marks },
                Table(; x = 0:50:5000, y = tf.(xs_fan))
            ),
            )
        )

rosenbrock_md = @pgf TikzPicture(
    PGFPlotsX.Axis(
            {
            xlabel=raw"\# function evaluations",
            ylabel=raw"$f(x)$",
            ymax=21,
            width="8cm",
            height="6cm"
            },
            Legend(["Ours + MD", "Wang (2018) + MD", "Fan (1992) + MD"]),
            PlotInc({ no_marks },
                Table(; x = 1:50:5000, y = tf.(xs_md_ours))
                ),
            PlotInc({ no_marks },
                Table(; x = 1:50:5000, y = tf.(xs_md_wang))
            ),
            PlotInc({ no_marks },
                Table(; x = 1:50:5000, y = tf.(xs_md_fan))
            )
            )
        )

rosenbrock_time = @pgf TikzPicture(
    PGFPlotsX.Axis(
            {
            xlabel=raw"Time (s)",
            ylabel=raw"$f(x)$",
            width="8cm",
            height="6cm"
            },
            Legend(["Ours", "Wang (2018)", "Fan (1992)"]),
            PlotInc({ no_marks },
                Table(; x = times_ours, y = tf.(xs))
            ),
            # PlotInc({ no_marks },
            #     Table(; x = 0:100, y = nm)
            #     ),
            PlotInc({ no_marks },
                Table(; x = times_wang, y = tf.(xs_wang))
            ),
            PlotInc({ no_marks },
                Table(; x = times_fan, y = tf.(xs_fan))
            ),
            )
        )

pgfsave("rosenbrock_$(d)+wang_nesterov.tikz", rosenbrock)
pgfsave("rosenbrock_$(d)+wang_md.tikz", rosenbrock_md)
pgfsave("rosenbrock_time_$(d)+wang_nesterov.tikz", rosenbrock_time)


#### ML Opt ###

using DataFrames
using CSV
using StatsBase
using StatsModels
using LinearAlgebra: norm

file = joinpath(@__DIR__, "data/adult.csv")
filepath = joinpath(@__DIR__, file)

df = CSV.File(filepath; header=1) |> DataFrame

X = modelmatrix(@formula(income ~ 1 + workclass + education + maritalstatus + occupation + relationship + race + gender + nativecountry + age + fnlwgt + educationalnum + capitalgain + capitalloss + hoursperweek), df)
Y = df[!, 15]
binarize(x) = x == "<=50K" ? 1 : 0
Y = binarize.(Y)

perm = randperm(size(X, 1))
X = X[perm, :]
scaler = StatsBase.fit(ZScoreTransform, X, dims=1)
X = StatsBase.transform(scaler, X)
X[:, 1] .= 1.
Y = BitArray(Y[perm])

logit(x::AbstractVector, θ::AbstractVector) = 1. ./ (1. .+ exp.(-x'θ))
logit(x::AbstractArray, θ::AbstractVector) = 1. ./ (1. .+ exp.(-θ'*X')) |> vec

function logloss(X, Y, θ)
    p = logit(X, θ)
    -sum(log, p[Y]) - sum(log, 1. .- p[.!Y])
end

θ = zeros(size(X, 2))
opt = Nesterov(0.0001, 0.3)
Θ = rand(MvNormal(θ, .001 * I), 50)'
alllosses = logloss.(Ref(X), Ref(Y), eachrow(Θ))
θs = reshape(θ, length(θ), 1)
losses = [logloss(X, Y, θ)]
times = [0.]
tick()
for i in 1:100
    lola = LocalLasso(Θ', alllosses)
    _, Δ, _, _ = predict(lola, θs[:, end])
    Δ = apply!(opt, clamp.(Δ, -1000.f0, 1000.f0))

    temp = rand(MvNormal(θs[:, end] - Δ, 0.001 * I), 50)'
#=     selected = abs.(Δ) .> 0.
    dirs = rand(MvNormal(zeros(count(selected)), 0.01 * I), 50)
    
    temp = repeat(xs[end] - Δ, outer=(1, 50))
    temp[selected, :] .+= dirs =#

    ftemp = logloss.(Ref(X), Ref(Y), eachrow(temp))
    
    m = argmin(ftemp)
    θ = temp[m, :]
    push!(losses, ftemp[m])
    θs = [θs θ]
    push!(times, peektimer())
    global alllosses = vcat(alllosses, ftemp)
    global Θ = vcat(Θ, temp)
end
tock()

opt_2 = Nesterov(0.0001, 0.3)
θ = θs[:, 1]
θs_grad = [θ]
times_grad = [0.]
tick()
for i in 1:100
    Δ = gradient(θ -> logloss(X, Y, θ), θs_grad[end])[1] |> vec
    Δ = apply!(opt_2, clamp.(Δ, -1000.f0, 1000.f0))
    push!(θs_grad, θs_grad[end] - Δ)
    push!(times_grad, peektimer())
end
tock()

using Optim
times_nm = [0.]
sol = optimize(θ -> logloss(X, Y, θ), θs[:, 1], NelderMead(), Optim.Options(iterations=5000, store_trace=true))

temp = map(x -> x.value, sol.trace)[1:50:end]
nm = zeros(101)
nm[1:length(temp)] .= temp
nm[(length(temp)+1):end] .= temp[end]

nm2 = map(x -> x.value, sol.trace)
times_nm = map(x -> x.metadata["time"], sol.trace)

logisticreg = @pgf TikzPicture(
        Axis(
            {
            xlabel=raw"\# iterations",
            ylabel=raw"Log-Likelihood"
            },
            Legend(["Estimated gradients", "True gradients", "Nelder-Mead"]),
            PlotInc({ no_marks },
                Table(; x = 0:100, y = logloss.(Ref(X), Ref(Y), eachcol(θs)))
                ),
            PlotInc({ no_marks },
                Table(; x = 0:100, y = logloss.(Ref(X), Ref(Y), θs_grad))
                ),
            PlotInc({ no_marks },
                Table(; x = 0:100, y = nm)
                ),
            )
        )

logisticreg2 = @pgf TikzPicture(
    Axis(
        {
        xlabel=raw"Time (s)",
        ylabel=raw"Log-Likelihood"
        },
        Legend(["Estimated gradients", "True gradients", "Nelder-Mead"]),
        PlotInc({ no_marks },
            Table(; x = times, y = losses)
            ),
        PlotInc({ no_marks },
            Table(; x = times_grad, y = logloss.(Ref(X), Ref(Y), θs_grad))
            ),
        PlotInc({ no_marks },
            Table(; x = times_nm, y = nm2)
            ),
        )
    )

logisticreg = @pgf TikzPicture(
                Axis(
                    {
                    xlabel=raw"\# iterations",
                    ylabel=raw"Log-Likelihood"
                    },
                    Legend(["Estimated gradients", "True gradients"]),
                    PlotInc({ no_marks },
                        Table(; x = 0:100, y = logloss.(Ref(X), Ref(Y), θs))
                        ),
                    PlotInc({ no_marks },
                        Table(; x = 0:100, y = logloss.(Ref(X), Ref(Y), θs_grad))
                        )
                    )
                )

pgfsave("logisticreg.tikz", logisticreg)
pgfsave("logisticreg_time.tikz", logisticreg2)

### Hyper Opt ###

using ScikitLearn 
using PyCall
using StatsBase
using StatsModels
using Random
using DataFrames: DataFrame, missing
using CSV

ScikitLearn.Skcore.import_sklearn() = PyCall.pyimport_conda("sklearn", "scikit-learn")
@sk_import svm: SVC
@sk_import preprocessing: (LabelBinarizer, StandardScaler, OneHotEncoder)

file = joinpath(@__DIR__, "../data/adult.txt")
filepath = joinpath(@__DIR__, file)
#df = CSV.read(filepath; silencewarnings=true, header=false)
#X = Array{Float64}(df[1:4])
#Y = Vector{Bool}(df[5])

df = CSV.read(filepath; silencewarnings=true)

X = modelmatrix(@formula(income ~ 1 + workclass + education + maritalstatus + occupation + relationship + race + gender + nativecountry + age + fnlwgt + educationalnum + capitalgain + capitalloss + hoursperweek), df)
Y = df[15]
binarize(x) = x == "<=50K" ? 1 : 0
Y = binarize.(Y)

perm = randperm(size(X, 1))
X = X[perm, :]
Y = Y[perm]

X_train, Y_train = X[1:2000, :], Y[1:2000]
X_test, Y_test = X[2001:4001, :], Y[2001:4001]

#X_train, Y_train = X[1:1000, :], Y[1:1000]
#X_test, Y_test = X[1001:end, :], Y[1001:end]

function loss(θ)
    logC, logω1, logω2, coef0, loggamma = θ
    C = exp(logC)
    ω1, ω2 = exp(logω1), exp(logω2)
    class_weight=Dict(zip(0:1, [ω1, ω2]))
    gamma = exp(loggamma)
    svc = ScikitLearn.fit!(SVC(C=C, coef0=coef0, gamma=gamma, class_weight=class_weight, probability=true), X_train, Y_train)
    p = svc.predict_proba(X_test)[:, 2]
    -sum(log, p[BitArray(Y_test)]) - sum(log, 1. .- p[.!BitArray(Y_test)])
end

θ = [0., 0.,0., 0., 0.]

opt = Nesterov(0.0001, 0.9)
Θ = rand(MvNormal(θ, 4.0*I), 50)
losses = loss.(eachcol(Θ))
θs = [θ]
@showprogress for i in 1:20
    lola = LocalLasso(Θ, losses)
    _, Δ, _, _ = predict(lola, θs[end])
    Δ = apply!(opt, clamp.(Δ, -1000.f0, 1000.f0))

    temp = rand(MvNormal(θs[end] - Δ, 4.0*I), 50)
#=     selected = abs.(Δ) .> 0.
    dirs = rand(MvNormal(zeros(count(selected)), 0.01 * I), 50)
    
    temp = repeat(xs[end] - Δ, outer=(1, 50))
    temp[selected, :] .+= dirs =#

    ftemp = loss.(eachcol(temp))
    
    m = argmin(ftemp)
    θ = temp[:, m]
    push!(θs, θ)
    global losses = vcat(losses, ftemp)
    global Θ = hcat(Θ, temp)
end

hyperparam = @pgf TikzPicture(
        Axis(
            {
            xlabel=raw"\# iterations",
            ylabel=raw"Test Loss"
            },
            Legend(["Estimated gradients"]),
            PlotInc({ no_marks },
                Table(; x = 0:20, y = loss.(θs))
                )
            )
        )

pgfsave("hyperparam.tikz", hyperparam)

### Sci Opt ###