function make_swiss_roll(n_samples; noise=0.0)
    t = 1.5 * π .+ 3. * π * rand(Float64, n_samples)
    x = t .* cos.(t)
    y = t .* sin.(t)
    z = 10 * rand(n_samples)

    X = [x y z]
    X += noise .* randn(n_samples, 3)

    return X, t
end

function make_locally_sparse_swiss_roll(n_samples; noise=0.0)
    t = 1.5 * π .+ 3. * π * rand(Float64, n_samples)
    x = t .* cos.(t)
    y = t .* sin.(t)
    z = 10 * rand(Float64, n_samples)

    X = [x y z]
    f = z .+ max.((9. .- t), 0.) .* x .+ max.((t .- 9.), 0.) .* y
    X += noise .* randn(n_samples, 3)

    return X, f
end

function make_trefoil(n_samples; noise=0.0)
    t = 10. * rand(Float64, n_samples)
    x = sin.(t) .+ 2. * sin.(2. * t)
    y = cos.(t) .- 2. * cos.(2. * t)
    x = -sin.(3. * t)
end