"""
Multilayer Perceptron
Cost function : Cross-Entropy + L2 regularization
Activation function : ReLU (or Sigmoïd)
Output function : Softmax
Labels structure : One-hot encoding
Weights initilization : Normal(0,1/sqrt(sizes[l]))
"""

mutable struct Network
    num_layers::Int64
    sizes::Vector{Int64}
    weights::Array{Matrix{Float64}}
    index::Array
    a::Array{Matrix{Float64}}
    δ::Matrix{Float64}
    λ::Float64
    function Network(sizes::Vector{Int64}, λ::Float64, n_total::Int64, seed::Int64)
        N = new()
        N.num_layers = length(sizes)
        N.sizes = copy(sizes)
        if seed > 0
            mrg_gen = MRG32k3aGen([seed for i = 1:6])
            mrg = next_stream(mrg_gen)
            N.weights = Array{Matrix{Float64}}(undef, N.num_layers-1)
        else
            N.weights = [zeros(j, i+1) for (i,j) in zip(sizes[1:N.num_layers-1], sizes[2:N.num_layers])]
        end
        N.index = Array{UnitRange{Int64}}(undef, N.num_layers-1)
        for k = 1:N.num_layers-1
            if seed > 0
                next_substream!(mrg)
                N.weights[k] = quantile.(Normal(0.0, 1/sqrt(sizes[k])), [rand(mrg) for i=1:sizes[k+1], j=1:sizes[k]+1])
            end
            i = sizes[k]
            j = sizes[k+1]
            N.index[k] = (k==1 ? 0 : N.index[k-1][end])+1:(k==1 ? 0 : N.index[k-1][end])+j*(i+1)
        end
        N.a = [ones(j + (i==length(sizes) ? 0 : 1), n_total) for (i,j) in enumerate(sizes)]
        N.δ = zeros(maximum(sizes), n_total)
        N.λ = λ
        return N
    end
end

function cross_entropy!(N::Network, Y::AbstractArray, data::AbstractArray)
    res = Series(Mean(), Variance())
    cov = Mean()
    n = size(Y, 2)
    for i = 1:n
        y = @view Y[:, i]
        j = argmax(y)
        error = -log(N.a[end][j, i])
        fit!(res, error)
        fit!(cov, error*data[i])
        data[i] = error
    end
    avg, var = value(res)
    L2 = @views N.λ/(2*n)*sum(sum(N.weights[i][:, 1:end-1].^2) for i = 1:N.num_layers-1)
    return avg + L2, var, value(cov)
end

function cross_entropy!(N::Network, Y::AbstractArray)
    #res = Series(Mean(), Variance())
    res = Mean()
    n = size(Y, 2)
    for i = 1:n
        y = @view Y[:, i]
        j = argmax(y)
        error = -log(N.a[end][j, i])
        fit!(res, error)
    end
    #avg, var = value(res)
    L2 = @views N.λ/(2*n)*sum(sum(N.weights[i][:, 1:end-1].^2) for i = 1:N.num_layers-1)
    #return avg + L2, var
    return value(res)
end 

function feedforward!(N::Network, β::AbstractArray, X::AbstractArray)
    for i = 1:N.num_layers-1
        N.weights[i][:,:] = @views reshape(β[N.index[i]], size(N.weights[i]))
    end
    n = size(X, 2)
    N.a[1][1:end-1, 1:n] = copy(X) 
    for l = 1:N.num_layers-2
        N.a[l+1][1:end-1, 1:n] = @views ReLU(N.weights[l]*N.a[l][:, 1:n])
    end
    N.a[end][:, 1:n] = @views softmax(N.weights[end]*N.a[end-1][:, 1:n])
end

function backpropagation!(N::Network, β::AbstractArray, X::AbstractArray, Y::AbstractArray, gradient::Vector{Float64})
    n = size(Y, 2)
    feedforward!(N, β, X)
    N.δ[1:N.sizes[end], 1:n] = @view N.a[end][:, 1:n]
    for i = 1:n
        y = @view Y[:, i]
        j = argmax(y)
        N.δ[j, i] -= 1
    end
    gradient[N.index[end]] = @views vec([(N.δ[1:N.sizes[end], 1:n]*(N.a[end-1][1:end-1, 1:n])'/n + N.λ/n*N.weights[end][:, 1:end-1]) mean(eachcol(N.δ[1:N.sizes[end], 1:n]))])
    for l = (N.num_layers-2):-1:1
        N.δ[1:N.sizes[l+1], 1:n] = @views ((N.weights[l+1][:, 1:end-1])'*N.δ[1:N.sizes[l+2], 1:n]).*map(z -> z==0 ? 0 : 1, N.a[l+1][1:end-1, 1:n])
        gradient[N.index[l]] = @views vec([(N.δ[1:N.sizes[l+1], 1:n]*(N.a[l][1:end-1, 1:n])'/n + N.λ/n*N.weights[l][:, 1:end-1]) mean(eachcol(N.δ[1:N.sizes[l+1], 1:n]))])
    end
end

function backpropagation!(N::Network, β::AbstractArray, X::AbstractArray, Y::AbstractArray, score::Array{Vector{Float64}})
    n = size(Y, 2)
    feedforward!(N, β, X)
    N.δ[1:N.sizes[end], 1:n] = @view N.a[end][:, 1:n]
    for i = 1:n
        y = @view Y[:, i]
        j = argmax(y)
        N.δ[j, i] -= 1
        score[i][N.index[end]] = @views vec([(N.δ[1:N.sizes[end], i]*(N.a[end-1][1:end-1, i])' + N.λ/n*N.weights[end][:, 1:end-1]) N.δ[1:N.sizes[end], i]])
        for l = (N.num_layers-2):-1:1
            N.δ[1:N.sizes[l+1], i] = @views ((N.weights[l+1][:, 1:end-1])'*N.δ[1:N.sizes[l+2], i]).*map(z -> z==0 ? 0 : 1, N.a[l+1][1:end-1, i])
            score[i][N.index[l]] = @views vec([(N.δ[1:N.sizes[l+1], i]*(N.a[l][1:end-1, i])' + N.λ/n*N.weights[l][:, 1:end-1]) N.δ[1:N.sizes[l+1], i]])
        end
    end
end

function backpropagation!(N::Network, β::AbstractArray, X::AbstractArray, Y::AbstractArray, hessian::Matrix)
    hessian[:, :] = zeros(N.index[end][end], N.index[end][end])
    n = size(Y, 2)
    feedforward!(N, β, X)
    N.δ[1:N.sizes[end], 1:n] = @view N.a[end][:, 1:n]
    for i = 1:n
        score = zeros(N.index[end][end])
        y = @view Y[:, i]
        j = argmax(y)
        N.δ[j, i] -= 1
        score[N.index[end]] = @views vec([(N.δ[1:N.sizes[end], i]*(N.a[end-1][1:end-1, i])' + N.λ/n*N.weights[end][:, 1:end-1]) N.δ[1:N.sizes[end], i]])
        for l = (N.num_layers-2):-1:1
            N.δ[1:N.sizes[l+1], i] = @views ((N.weights[l+1][:, 1:end-1])'*N.δ[1:N.sizes[l+2], i]).*map(z -> z==0 ? 0 : 1, N.a[l+1][1:end-1, i])
            score[N.index[l]] = @views vec([(N.δ[1:N.sizes[l+1], i]*(N.a[l][1:end-1, i])' + N.λ/n*N.weights[l][:, 1:end-1]) N.δ[1:N.sizes[l+1], i]])
        end
        hessian[:, :] = @views hessian[:, :] + score*score'
    end
    hessian[:, :] = @views hessian[:, :]/n
end

sigmoid(inp::AbstractArray) = map(z -> 1.0/(1.0+exp(-z)), inp)

ReLU(inp::AbstractArray) = map(z -> max(z,0), inp)

function softmax(inp::AbstractArray)
    out = similar(inp)
    for (i,v) in enumerate(eachcol(inp))
        m = maximum(v)
        out[:, i] = map(z -> exp(z-m), v)
        out[:, i] = out[:, i]/sum(out[:, i])
    end
    return out
end
        
        
    