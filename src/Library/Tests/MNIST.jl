function one_hot(set::AbstractArray, nb_out::Int64)
    new_set = zeros(nb_out, length(set))
    for i in 1:length(set)
         new_set[set[i]+1, i] = 1
    end
    return new_set
end

function evaluation(β::AbstractArray, X::AbstractArray, Y::AbstractArray, N::Network)
""" Evaluates the percentage of success of β over the set """
    feedforward!(N, β, X)
    s = 0
    n = size(Y, 2)
    for i = 1:n
        if argmax(N.a[end][:, i]) == argmax(Y[:, i])
            s += 1
        end
    end
    return 100*s/n
end

function Test_MNIST(optim::String = "RA", layers::Array = [], init_seed::Int64 = 1, set_seed::Int64 = 0,
        criterion::Int64 = 2, verbose::Bool = false, vargs ...)
"""
vargs[1] = N0::Int64 (if optim == RA)
vargs[2] = coeff::Float64 (if optim == RA)
vargs[3] = eps::Float64 (if optim == RA)
vargs[4] = sample_coeff::Float64 (if optim == RA)
"""
    
    (train_x, train_y) = MNIST.traindata()
    (test_x,  test_y)  = MNIST.testdata()
    
    ntest = size(test_y, 1)
    ntrain = criterion==3 ? size(train_y, 1)-ntest : size(train_y, 1)
    nvalid = criterion==3 ? ntest : 0
    ntot = ntrain + nvalid + ntest
    sizes = size(layers, 1) != 0 ? [784; layers; 10] : [784, 10]
    N = Network(sizes, 0.0, ntot, init_seed)
    
    b = BatchMLP([MNIST.convert2features(train_x) MNIST.convert2features(test_x)], [one_hot(train_y, 10) one_hot(test_y, 10)], ones(Int64, ntot))
    mnist = MLP(b, N.index[end][end])
    
    if set_seed > 0
        mrg_gen = MRG32k3aGen([set_seed for i = 1:6])
        mrg = next_stream(mrg_gen)
        perm = [Int(ceil(ntot*rand(mrg))) for i = 1:ntot]
        list = [true for i = 1:ntot]
        for (ind, k) in enumerate(perm)
            if list[k]
                list[k] = false
            else
                i = k==ntot ? k : k+1
                j = k==1 ? k : k-1
                while (i<ntot || j>1) && !list[i] && !list[j]
                    i = i==ntot ? i : i+1
                    j = j==1 ? j : j-1
                end
                perm[ind] = list[i] ? i : j
                list[perm[ind]] = false
            end
        end
    else
        perm = [i for i = 1:ntot]
    end
    
    mnist.batch.train_x = @view mnist.batch.features[:, perm[1:ntrain]]
    mnist.batch.train_y = @view mnist.batch.labels[:, perm[1:ntrain]]
    mnist.batch.valid_x = @view mnist.batch.features[:, perm[ntrain+1:ntrain+nvalid]]
    mnist.batch.valid_y = @view mnist.batch.labels[:, perm[ntrain+1:ntrain+nvalid]]
    mnist.batch.test_x = @view mnist.batch.features[:, perm[ntrain+nvalid+1:ntot]]
    mnist.batch.test_y = @view mnist.batch.labels[:, perm[ntrain+nvalid+1:ntot]]
    
    mnist.n_train = ntrain
    mnist.n_valid = nvalid
    mnist.α = 0.05
    mnist.ϵ = 0
    mnist.sample_size = ntrain
    mnist.subsample = mnist.sample_size
    mnist.q_student = quantile(TDist(mnist.sample_size-1), 1-mnist.α)
    
    function tTest(state::GERALDINE.BTRState)
    """ Test if the difference of the two last value of the loss function is less or equal than ϵ*f_old """
        if state.iter != 0 && mnist.f_old != state.fx
            σ2 = mnist.var + mnist.old_var - 2*mnist.cov
            tstat = (mnist.f_old*(1 - mnist.ϵ) - state.fx)/sqrt(σ2/mnist.sample_size)
            if tstat <= mnist.q_student
                return true
            end
        end
        mnist.f_old = state.fx
        return false
    end
    
    function validation(state::GERALDINE.BTRState)
    """ Test if the current value of the loss function over the validation set is greater than the previous one """
        ftest = mnist.f_valid(state.x)
        if state.iter != 0 && mnist.f_old < ftest
            return true
        end
        mnist.f_old = ftest
        return false
    end
    
    function f(β::AbstractArray)
        trainx = @view mnist.batch.train_x[:, 1:mnist.sample_size]
        trainy = @view mnist.batch.train_y[:, 1:mnist.sample_size]
        feedforward!(N, β, trainx)
        fvalue, vari, cova = cross_entropy!(N, trainy, mnist.data)
        mnist.old_var = mnist.var
        mnist.var = vari
        mnist.cov = cova - mnist.f_old*fvalue
        return fvalue
    end
    
    function f_valid(β::AbstractArray)
        feedforward!(N, β, mnist.batch.valid_x)
        return cross_entropy!(N, mnist.batch.valid_y)
    end
    
    function f_train(β::AbstractArray)
        trainx = @view mnist.batch.train_x[:, 1:mnist.sample_size]
        trainy = @view mnist.batch.train_y[:, 1:mnist.sample_size]
        feedforward!(N, β, trainx)
        return cross_entropy!(N, trainy)
    end
    
    function g_score!(β::AbstractArray, gradient::Vector{Float64}, score::Array{Vector{Float64}})
        trainx = @view mnist.batch.train_x[:, 1:mnist.sample_size]
        trainy = @view mnist.batch.train_y[:, 1:mnist.sample_size]        
        backpropagation!(N, β, trainx, trainy, gradient)
        trainx = @view mnist.batch.train_x[:, 1:mnist.subsample]
        trainy = @view mnist.batch.train_y[:, 1:mnist.subsample]
        backpropagation!(N, β, trainx, trainy, score)
    end
    
    function g!(β::AbstractArray, gradient::Vector{Float64})
        trainx = @view mnist.batch.train_x[:, 1:mnist.sample_size]
        trainy = @view mnist.batch.train_y[:, 1:mnist.sample_size]        
        backpropagation!(N, β, trainx, trainy, gradient)
    end
    
    function bhhh!(β::AbstractArray, hessian::Matrix)
        trainx = @view mnist.batch.train_x[:, 1:mnist.sample_size]
        trainy = @view mnist.batch.train_y[:, 1:mnist.sample_size]        
        backpropagation!(N, β, trainx, trainy, hessian)
    end
        
    mnist.f = f
    mnist.f_valid = f_valid
    mnist.f_train = f_train
    mnist.tTest = tTest
    mnist.validation = validation
    β0 = Array{Float64}(undef, mnist.dim)
    for i in 1:N.num_layers-1
       β0[N.index[i]] = vec(N.weights[i])
    end
        
    if optim == "RA"
        if criterion == 1
            error("criterion 1 is not available for RA")
        end
        mnist.ϵ = vargs[3]
        mnist.score! = g_score!
        param, acc = solve_BTR_RA(mnist, β0, vargs[1], vargs[2], sample_coeff = vargs[4], criterion = criterion, verbose = verbose)
    elseif optim == "HOPS"
        mnist.score! = g_score!
        state, acc = solve_BTR_HOPS(mnist, x0 = β0, verbose = verbose, criterion = criterion)
        param = state.x
    elseif optim == "BFGS"
        mnist.∇f! = g!
        state, acc = solve_BTR_BFGS(mnist, x0 = β0, verbose = verbose, criterion = criterion)
        param = state.x
    elseif optim == "BHHH"
        mnist.∇f! = g!
        mnist.bhhh! = bhhh!
        state, acc = solve_BTR_BHHH(mnist, x0 = β0, verbose = verbose, criterion = criterion)
        param = state.x
    else
        error("'", optim,"' does not exist")
        return
    end
    
    return param, acc, N, mnist
end