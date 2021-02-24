####################################### On a manifold #######################################
using Distributed
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
using LinearAlgebra: I, Diagonal, norm
using Flux, MLDatasets
using Zygote
using Random
using CSV
using ProgressMeter
using BSON
using BSON: @load, @save
using CuArrays
using LinearAlgebra: norm

include(joinpath(@__DIR__, "queuepool.jl"))
include(joinpath(@__DIR__, "utils.jl"))

const dir = joinpath(@__DIR__, "../CACD2000/")

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
                using Flux, Images
                using Images: channelview
                using ImageTransformations
                include(joinpath(@__DIR__, "utils.jl"))
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
    return X, state
end

mutable struct OneCycle
    steps::Int64
    clip::Float64
    amplification::Float64
    state::IdDict
end
  
OneCycle(steps = 1000, clip = 1e-4, amplification = 10.) = OneCycle(steps, clip, amplification, IdDict())
  
function apply!(o::OneCycle, x, Δ)
    n = get!(o.state, x, 1)
    Δ .*= n <= o.steps / 2 ? 1 + 2 * n / o.steps * (o.amplification-1) : o.amplification + n / o.steps * (1 - o.amplification)
    o.state[x] = n + 1
    return Δ
end

const BATCH_SIZE = 128

dataset = AsyncImageLoader(dir, 16, BATCH_SIZE, load_image; queue_size=512, device = gpu)

const LATENT_SIZE = 512

NUM_EPOCHS = 50
training_steps = 0

# Stupid fix
mish(x::AbstractArray) = x .* tanh.(softplus.(x))
mutable struct EvoNormS0
    v
    γ
    β
end

EvoNormS0() = EvoNormS0(randn(Float32), randn(Float32), randn(Float32))
EvoNormS0(shape) = EvoNormS0(randn(Float32, (shape...,1)), randn(Float32, (shape...,1)), randn(Float32))

# Overload call, so the object can be used as a function
function (m::EvoNormS0)(x)
    H, W, C, N = size(x)
    T = eltype(x)
    μ = mean(x, dims = 4)
    σ² = sum((x .- μ) .^ 2, dims = 4)
    ϵ = 1e-5
    x .* NNlib.sigmoid.(m.v .* x) ./ (σ² .+ ϵ) .* m.γ .+ m.β 
end

Flux.@functor EvoNormS0

enc = Chain(Conv((4, 4), 3 => 32, stride = 2, pad = 1, leakyrelu),
#    EvoNormS0(),
	BatchNorm(32),
    Conv((4, 4), 32 => 64, stride = 2, pad = 1, leakyrelu),
#    EvoNormS0(),
	BatchNorm(64),
    Conv((4, 4), 64 => 128, stride = 2, pad = 1, leakyrelu),
#    EvoNormS0(),
	BatchNorm(128),
    Conv((4, 4), 128 => 256, stride = 2, pad = 1, leakyrelu),
#    EvoNormS0(),
    BatchNorm(256),
    Conv((4, 4), 256 => 512, stride = 2, pad = 1, leakyrelu),
#    EvoNormS0(),
    BatchNorm(512),
    Conv((4, 4), 512 => 512, stride = 2, pad = 1, leakyrelu),
#    EvoNormS0(),
    BatchNorm(512),
	x -> reshape(x, :, size(x, 4))) |> gpu
    
μ, logσ = Dense(2048, LATENT_SIZE) |> gpu, Dense(2048, LATENT_SIZE) |> gpu
g(X) = (h = enc(X); (μ(h), logσ(h)))

# Need an adjoint for CuArrays.randn
using Zygote: @adjoint
@adjoint CuArrays.randn(dims::Int) = CuArrays.randn(dims), Δ -> (nothing,nothing)

z(μ, logσ) = μ .+ exp.(logσ) .* CuArrays.randn(LATENT_SIZE)

function upsample(x; ratio=(2, 2, 1, 1))
    (h, w, c, n) = size(x)
    y = similar(x, (ratio[1], 1, ratio[2], 1, 1, 1))
    fill!(y, 1)
    z = reshape(x, (1, h, 1, w, c, n))  .* y
    reshape(z, size(x) .* ratio)
end

@adjoint upsample(x; ratio=(2, 2, 1, 1)) = upsample(x; ratio=(2, 2, 1, 1)), 
Δ -> (MeanPool((ratio[1], ratio[2]))(Δ), )

f = Chain(
    Dense(LATENT_SIZE, 2048, swish),
    x -> reshape(x, 2, 2, 512, :),
    ConvTranspose((4, 4), 512 => 256, leakyrelu; stride = 2, pad = 1),
#    EvoNormS0(),
	BatchNorm(256),
    ConvTranspose((4, 4), 256 => 128, leakyrelu; stride = 2, pad = 1),
#    EvoNormS0(),
	BatchNorm(128),
    ConvTranspose((4, 4), 128 => 64, leakyrelu; stride = 2, pad = 1),
#    EvoNormS0(),
	BatchNorm(64),
    ConvTranspose((4, 4), 64 => 32, leakyrelu, stride = 2, pad = 1),
#    EvoNormS0(),
    BatchNorm(32),
    ConvTranspose((4, 4), 32 => 16, leakyrelu; stride = 2, pad = 1),
#    EvoNormS0(),
    BatchNorm(16),
    ConvTranspose((4, 4), 16 => 8, leakyrelu; stride = 2, pad = 1),
#    EvoNormS0(),
    BatchNorm(8),
    ConvTranspose((1, 1), 8 => 3; stride = 1, pad = 0),
#    EvoNormS0(),
    BatchNorm(8),
    x -> sigmoid.(x)
    ) |> gpu

#= f = Chain(
    Dense(LATENT_SIZE, 2048, selu),
    x -> reshape(x, 2, 2, 512, :),
    upsample,
    Conv((3, 3), 512 => 256, leakyrelu; stride = 1, pad=1),
    BatchNorm(256),
    upsample,
    ConvTranspose((3, 3), 256 => 128, leakyrelu; stride = 1, pad=1),
    BatchNorm(128),
    upsample,
    ConvTranspose((3, 3), 128 => 64, leakyrelu; stride = 1, pad=1),
    BatchNorm(64),
    upsample,
    ConvTranspose((3, 3), 64 => 32, leakyrelu, stride = 1, pad=1),
    BatchNorm(32),
    upsample,
    ConvTranspose((3, 3), 32 => 16, leakyrelu; stride = 1, pad=1),
    BatchNorm(16),
    upsample,
    ConvTranspose((3, 3), 16 => 8, leakyrelu; stride = 1, pad=1),
    BatchNorm(8),
    ConvTranspose((3, 3), 8 => 3; stride = 1, pad=1),
    BatchNorm(8),
    x -> sigmoid.(x)
    ) |> gpu =#
    
# KL-divergence between approximation posterior and N(0, 1) prior.
kl_q_p(μ, logσ) = 0.5f0 * sum(exp.(2f0 .* logσ) + μ.^2 .- 1f0 .- (2 .* logσ))

# GPU Normal logpdf
#lπ = 1.8378770664093453 |> Float32 # log(2pi) |> Float32
#lpdf(x, μ) = - 0.5f0 * lπ .- 0.5f0 .* (x .- μ).^2
lpdf(y, x) = y .* log.(x .+ eps(Float32)) + (1f0 .- y) .* log.(1f0 .- x .+ eps(Float32))

# logp(x|z) - conditional probability of data given latents.
logp_x_z(x, z) = sum(lpdf(x, f(z)))

# Monte Carlo estimator of mean ELBO using M samples.
L̄(X, β) = ((μ̂, logσ̂) = g(X); (logp_x_z(X, z(μ̂, logσ̂)) - β .* kl_q_p(μ̂, logσ̂)) .* 1.f0 / BATCH_SIZE)

# L2 regularization
penalty() = sum(norm, params(enc, μ, logσ, f))
# Sample from the learned model.
modelsample() = f(z(CuArrays.zeros(LATENT_SIZE), CuArrays.zeros(LATENT_SIZE))) |> cpu

@show isfile(joinpath(@__DIR__, "../models/vae_bce.bson"))
if isfile(joinpath(@__DIR__, "../models/vae_bce.bson"))
    @info "Loading Pre-Trained"
    @load joinpath(@__DIR__, "../models/vae_bce.bson") enc μ logσ f
    enc, μ, logσ, f = enc |> gpu, μ |> gpu, logσ |> gpu, f |> gpu
else
    @info "Pre-training"
    CuArrays.allowscalar(false)
    opt = RADAM(0.00001)

    #Flux.Optimise.@epochs 10 Flux.train!(loss, ps, dataset, opt, cb=evalcb)
    Flux.Optimise.@epochs 20 Flux.train!(x -> -L̄(x, 1.f0), ps, dataset, opt)
    bson(joinpath(@__DIR__, "../dmodels/vae_bce.bson"), enc=enc |> cpu, μ=μ |> cpu, logσ=logσ |> cpu, f=f |> cpu)
end

ps = params(enc, μ, logσ, f)

################################# Learn Parameters ##############################

function main()
    @showprogress for β in Float32.(collect(LinRange(1., 10., 20)))
        @info "Training"
        CuArrays.allowscalar(false)
        opt = RADAM(0.00001)

        #Flux.Optimise.@epochs 10 Flux.train!(loss, ps, dataset, opt, cb=evalcb)
        Flux.Optimise.@epochs 5 Flux.train!(x -> -L̄(x, β), ps, dataset, opt)
        bson(joinpath(@__DIR__, "../models/vae_bce_beta_$(β).bson"), enc=enc |> cpu, μ=μ |> cpu, logσ=logσ |> cpu, f=f |> cpu)
    end
end

main()