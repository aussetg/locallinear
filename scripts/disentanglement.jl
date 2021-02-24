using Distributed
using DrWatson
@quickactivate "locallinear"
using Base.Filesystem: readdir
using FileIO
using ImageShow
using Images
using Random: shuffle!
using Flux
using Flux: throttle, params
using Base.Iterators: partition
using Images: channelview
using CuArrays
using ImageTransformations
using Random: shuffle
using Distributions
using LinearAlgebra: I, Diagonal, norm, dot
using Flux, MLDatasets
using Zygote
using Random
using CSV
using ProgressMeter
using BSON
using BSON: @load, @save
using CuArrays
using LinearAlgebra: norm
using PGFPlotsX
using DataFrames

include(srcdir("queuepool.jl"))
include(srcdir("utils.jl"))
include(srcdir("DecisionTree.jl", "src/regression/locallinear.jl"))

const dir = srcdir("exp_raw/CACD2000/")
struct AsyncImageLoader
    dataset_root :: String
    filenames :: Vector{String}
    batch_size :: Int
    data_loader :: Function

    queue_pool::QueuePool
    device :: Function

    function AsyncImageLoader(path, num_workers::Int, batch_size::Int, data_loader :: Function; queue_size=128, device = gpu)
        
        images = filter(x -> endswith(x, ".jpg"), readdir(dir))

        @info("Adding $(num_workers) new data workers...")
        queue_pool = QueuePool(num_workers, data_loader, quote
                # The workers need to be able to load images and preprocess them via Metalhead
                using DrWatson
                @quickactivate "locallinear"
                using Flux, Images
                using Images: channelview
                using ImageTransformations
                using BSON
                using BSON: @load, @save
                using ProgressMeter
                include(srcdir("DecisionTree.jl", "src/regression/locallinear.jl"))
                include(srcdir("utils.jl"))
            end, 
            queue_size)

        return new(path, images, batch_size, data_loader, queue_pool, device)
    end
end

Base.length(id::AsyncImageLoader) = div(length(id.filenames), id.batch_size)

mutable struct AsyncImageLoaderState
    batch_idx::Int
    job_offset::Int
    
    function AsyncImageLoaderState(id::AsyncImageLoader)
        @info("Creating IIS with $(length(id.filenames)) images")

        # Build permutation for this iteration
        permutation = shuffle(1:length(id.filenames))

        # Push first job, save value to get job_offset (we know that all jobs
        # within this iteration will be consequtive, so we only save the offset
        # of the first one, and can use that to determine the job ids of every
        # subsequent job:
        filename = joinpath(id.dataset_root, id.filenames[permutation[1]])
        job_offset = push_job!(id.queue_pool, filename)

        # Next, push every other job
        for pidx in permutation[2:end]
            filename = joinpath(id.dataset_root, id.filenames[pidx])
            push_job!(id.queue_pool, filename)
        end
        return new(
            0,
            job_offset,
        )
    end
end

function Base.iterate(id::AsyncImageLoader, state=AsyncImageLoaderState(id))
    # If we're at the end of this epoch, give up the ghost
    if state.batch_idx >= length(id)
        return nothing
    end

    # Otherwise, wait for the next batch worth of jobs to finish on our queue pool
    next_batch_job_ids = state.job_offset .+ (0:(id.batch_size-1)) .+ id.batch_size*state.batch_idx
    # Next, wait for the currently-being-worked-on batch to be done.
    pairs = fetch_result.(Ref(id.queue_pool), next_batch_job_ids)
    state.batch_idx += 1

    # Collate X's and Y's into big tensors:
    X = cat((p[1] for p in pairs)...; dims=4) |> id.device
    Y = cat((p[2] for p in pairs)...; dims=1) |> id.device

    # Return the fruit of our labor
    return (X, Y), state
end

const BATCH_SIZE = 128

dataset = AsyncImageLoader(dir, 16, BATCH_SIZE, load_image; queue_size=512, device = gpu)

using Zygote: @adjoint
@adjoint CuArrays.randn(dims::Int) = CuArrays.randn(dims), Δ -> (nothing,nothing)

@load joinpath(@__DIR__, "../models/vae_bce.bson") enc μ logσ f
enc_t, μ_t, logσ_t = enc |> gpu, μ |> gpu, logσ |> gpu

function make_latents()
    @showprogress for β in Float32.(collect(LinRange(1., 10., 20)))[1:10]
        @load joinpath(@__DIR__, "../models/vae_bce_beta_$(β).bson") enc μ logσ f
        Flux.loadparams!(enc_t, params(enc) |> gpu)
        Flux.loadparams!(μ_t, params(μ) |> gpu)
        Flux.loadparams!(logσ_t, params(logσ) |> gpu)

        g(X) = (h = enc_t(X); (μ_t(h), logσ_t(h)))

        latents = Vector{Matrix{Float32}}()
        ages = Vector{Int}()

        for (x, y) in dataset
            latent = vcat(g(x)...) |> cpu
            latents = hcat(latents, latent)
            push!(latents, latent)
            append!(ages, y |> cpu)
        end
        bson(joinpath(@__DIR__, "../models/latents_$(β)_512.bson"), latents=latents[1:512, :])
        bson(joinpath(@__DIR__, "../models/latents_$(β)_1024.bson"), latents=latents)
        bson(joinpath(@__DIR__, "../models/ages_$(β).bson"), ages=ages)
    end
end

@everywhere function __make_gradients(d, β; ks=nothing, lambda=nothing)
    @load joinpath(@__DIR__, "../models/ages_$(β).bson") ages
    @load joinpath(@__DIR__, "../models/latents_$(β)_$(d).bson") latents
    lola = LocalLasso(latents, ages)
    idx = 1
    gradients = zeros(Float32, (d, 1000))
    #p = Progress(100, 1)
    for l in eachcol(latents)
        if idx > 1000
            break
        else
            try
                pred = predict(lola, l; ks=ks, lambda=lambda)
                @show pred[3] pred[4]
                gradients[:, idx] .= pred[2]
            catch e
                continue
            end
        end
        idx += 1
        #next!(p)
    end
    bson(joinpath(@__DIR__, "../models/gradients_$(β)_$(d).bson"), gradients=gradients)
end

function make_gradients(d; ks=nothing, lambda=nothing)
    @showprogress pmap(β -> __make_gradients(d, β; ks=ks, lambda=lambda), Float32.(collect(LinRange(1., 10., 20)))[1:10])
end

cosine(a, b) = dot(a, b) / (norm(a) * norm(b))

function metric(a)
    idxs = sortperm(abs.(a); rev=true)
    n = length(a)
    cosine(a, [i == idxs[1] ? sign(a[idxs[1]]) : 0 for i in 1:n])
end

function metric2(a)
    absa = abs.(a)
    n = length(a)
    w = absa ./ sum(absa)
    sum([w[k] * cosine(a, [i == k ? sign(a[k]) : 0 for i = 1:n]) for k = 1:n if a[k] != 0])
end

function metric3(a)
    absa = abs.(a)
    n = length(a)
    w = absa ./ sum(absa)
    minimum([cosine(a, [i == k ? sign(a[k]) : 0 for i = 1:n]) for k = 1:n if a[k] != 0])
end

function make_plots(d)
    dir = joinpath(@__DIR__, "../models")
    files = filter(x -> startswith(x, "gradients_") && endswith(x, "$(d).bson"), readdir(dir))
    sort!(files; by=x -> parse(Float64, split(x, "_")[2]))
    βs = Vector{Float64}()
    metrics = Vector{Float64}()
    for file in files[1:10]
        β = parse(Float64, split(file, "_")[2])
        push!(βs, β)
        @load joinpath(dir, file) gradients
        ∇ = mean(abs, gradients; dims=2) |> vec
        push!(metrics, metric3(∇))
    end

    results = DataFrame(β = [], metric = [])
    for file in files
        β = parse(Float64, split(file, "_")[2])
        push!(βs, β)
        @load joinpath(dir, file) gradients
        for ∇ in eachcol(gradients)
            push!(results, [β, metric(∇)])
        end
    end
    results = results[.!isnan.(results.metric), :]

    disentangle = @pgf TikzPicture(
        PGFPlotsX.Axis(
            {
            xlabel=raw"$\beta$",
            ylabel=raw"Age entanglement"
            },
            #Legend([""]),
            PlotInc({ no_marks },
                Table(; x = βs, y = metrics)
                )
            )
        )

        disentangle = @pgf TikzPicture(
        PGFPlotsX.Axis(
            {scatter,
            xlabel=raw"$\beta$",
            ylabel=raw"Age disentanglement metric"
            },
            #Legend([""]),
            Plot(
                { scatter,
                  "only marks"
                },
                Table(; x = results.β, y = results.metric)
                )
            )
        )
 

    pgfsave("disentangle.tikz", disentangle)

end

function main()
    #make_latents()
    make_gradients(512)
    #make_plots(1024)
end

main()