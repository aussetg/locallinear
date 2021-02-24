using Plots
using DataFrames
using CSV
using StatsBase
using Statistics
using BSON
using BSON: @save, @load
using Dates

@load joinpath(@__DIR__, "../results/results_diamonds_2020-05-13T09:30:00.08.bson") results
results_diamonds = results
@load joinpath(@__DIR__, "../results/results_gasoline_2020-05-13T09:30:42.477.bson") results
results_gasoline = results
@load joinpath(@__DIR__, "../results/results_heart_2020-05-13T08:02:40.534.bson") results
results_heart = results
@load joinpath(@__DIR__, "../results/results_wisconsin_2020-05-13T08:02:27.985.bson") results
results_wisconsin = results
@load joinpath(@__DIR__, "../results/results_swissroll_2020-05-13T09:35:19.758.bson") results
results_swissroll = results

by(results_diamonds, :method, :loss => mean, :loss => var, :loss => x -> 1.96 * var(x)/sqrt(50))
by(results_gasoline, :method, :loss => mean, :loss => var, :loss => x -> 1.96 * var(x)/sqrt(50))
by(results_heart, :method, :loss => mean, :loss => var, :loss => x -> 1.96 * var(x)/sqrt(50))
by(results_wisconsin, :method, :loss => mean, :loss => var, :loss => x -> 1.96 * var(x)/sqrt(50))
by(results_swissroll, :method, :loss => mean, :loss => var, :loss => x -> 1.96 * var(x)/sqrt(50))

X, Y = wisconsin()
size(X)
X, Y = heart()
size(X)
X, Y = diamonds()
size(X)
X, Y = gasoline()
size(X)