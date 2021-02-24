latents = hcat([b |> gpu |> enc |> μ |> cpu for b in all_batches]...)
bson("latents.bson", latents=latents)

################################# Display Images ###############################
using BSON
using Images, ImageView

model = BSON.load("models/vae_bce.bson")
enc, μ, logσ, f = model[:enc] |> gpu, model[:μ] |> gpu, model[:logσ] |> gpu, model[:f] |> gpu

arr = load_image("17_Evan_Rachel_Wood_0001.jpg")[1]
arr = reshape(arr, (128, 128, 3, 1))

function display_image(arr)
    tmp = dropdims(arr; dims=4)
    tmp = permutedims(tmp, (3, 1, 2))
    colorview(RGB, tmp)
end

function save_image(arr; title="encoded.jpg")
    tmp = display_image(arr)
    save(title, tmp)
end

arr |> display_image
arr |> gpu |> enc |> μ |> f |> cpu |> display_image
arr |> gpu |> enc |> μ |> f |> cpu |> save_image

using NearestNeighbors

latents = BSON.load("models/latents.bson")[:latents]

ages = vcat(pmap(b -> load_batch(b)[2], batches)...)
kdtree = KDTree(X)


arr |> display_image
arr |> gpu |> enc |> μ |> f |> cpu |> display_image

latent = dropdims(arr |> gpu |> enc |> μ |> cpu; dims=2)

include("src/locallinear.jl")

lll = LocalLasso(latents, ages)

age, ∇age = lll(latent, ages)

latent |> gpu |> f |> cpu |> display_image
latent+0.2∇age |> gpu |> f |> cpu |> display_image
latent+0.2∇age |> gpu |> f |> cpu |> save_image