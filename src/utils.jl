unsqueeze(xs, dim) = reshape(xs, (size(xs)[1:dim-1]..., 1, size(xs)[dim:end]...))
stack(xs, dim) = cat(unsqueeze.(xs, dim)..., dims=dim)
unstack(xs, dim) = [copy(selectdim(xs, dim, i)) for i in 1:size(xs, dim)]

function load_image(im)
    arr = imresize(load(im), (128, 128))
    arr = RGB.(arr) |> channelview
    arr = permutedims(arr, [2, 3, 1])
    file = split(im, "/")[end]
    age = parse(Int, split(file, "_")[1])
    return Float32.(arr), age
end

unzip(arr) = map(first, arr), map(last, arr)

function load_batch(batch)
    X, Y = load_image.(batch) |> unzip
    X = stack(X, 4)
    return X, Y
end