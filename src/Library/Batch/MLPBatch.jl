mutable struct BatchMLP <: Batch
    features::Matrix
    labels::Matrix
    train_x::SubArray
    train_y::SubArray
    test_x::SubArray
    test_y::SubArray
    valid_x::SubArray
    valid_y::SubArray
    weights::Array
    function BatchMLP(features, labels, weights)
        b = new()
        b.features = features
        b.labels = labels
        b.weights = weights
        return b
    end
end

function iterate(b::BatchMLP, state::Int64 = 1)
    if state > size(b.train_y, 2)
        return nothing
    else
        return LM_Individual(b.train_x[:, state], argmax(b.train_y[:, state])-1, b.weights[state]), state+1
    end
end