using GERALDINE, RDST, Distributions, ForwardDiff, OnlineStats

mutable struct LS
    n::Int64
    x::Array
    y::Vector
    βs::Vector
    U::Vector
    R::Array{Function}
    f::Function
    g!::Function
    g_score!::Function
    H!::Function
    TH!::Function
    function LS(βs::Vector, fct::Function, n::Int64, p::Int64, σ::Float64)
        model = new()
        model.n = n
        model.βs = copy(βs)
        mrg_gen = MRG32k3aGen([1,1,1,1,1,1])
        mrg = next_stream(mrg_gen)
        model.x = ones(n, p)
        for i in 1:p
            model.x[:,i] = [rand(mrg) for k in 1:n]
        end
        next_substream!(mrg)
        model.U = quantile.(Normal(0.0, σ), [rand(mrg) for k in 1:n])
        model.y = Array{Float64}(undef, n)
        model.R = Array{Function}(undef, n)
        for i in 1:n
            model.y[i] = fct(model.βs, model.x[i,:]) + model.U[i]
            model.R[i] = β -> model.y[i] - fct(β, model.x[i,:])
        end
        f(β) = 0.5/n*sum([model.R[i](β)^2 for i in 1:n])
        model.f = f
        function g!(β::Vector, storage::Vector)
            storage[:] = zeros(p+1)
            for i in 1:n 
                storage[:] += model.R[i](β)*ForwardDiff.gradient(model.R[i], β)
            end
            storage[:] *= 1/n
        end
        model.g! = g!
        function g_score!(β::Vector, storage::Vector, score::Array)
            storage[:] = zeros(p+1)
            for i in 1:n
                score[i] = zeros(p+1)
                score[i] = ForwardDiff.gradient(model.R[i], β)
                storage[:] += model.R[i](β)*score[i]
            end
            storage[:] *= 1/n
        end
        model.g_score! = g_score!
        function H!(β::Vector, storage::Matrix)
            storage[:] = zeros(p+1, p+1)
            for i in 1:n
                s = ForwardDiff.gradient(model.R[i], β)
                storage[:,:] += s*s'
            end
            storage[:, :] *= 1/n
        end
        model.H! = H!
        function TH!(β::Vector, storage::Matrix)
            storage[:, :] = ForwardDiff.hessian(model.f, β)
        end      
        model.TH! = TH!
        return model
    end
end

import Base.println
function println(m::LS)
    println("β = $(m.βs')")
end

function model(p::Int64)
    βs = [i/100 for i in 1:p]
    β0 = [0.0 for i in 1:p]

    fct(β, x) = β[1] + sum([exp(β[i+1]*x[i]) for i in 1:p-1])
    
    m = LS(βs, fct, 20*p, p-1, 0.01)
    
    return β0, m
end

function Tests_LS()
    
    (β01, m1) = model(25)
    (β02, m2) = model(50)
    (β03, m3) = model(100)
    
    list_m = [(m1,β01), (m2,β02), (m3,β03)]
    
    epsilon = 1e-4
    max_iter = 1000

    res, elapsed_time, b_alloc, gctime, memallocs = @timed OPTIM_btr_BFGS(m1.f, m1.g!, β01, verbose = false)
    res, elapsed_time, b_alloc, gctime, memallocs = @timed OPTIM_btr_TH(m1.f, m1.g!, m1.H!, β01, verbose = false)
    res, elapsed_time, b_alloc, gctime, memallocs = @timed OPTIM_btr_HOPS(m1.f, m1.g_score!, β01, ones(m1.n), verbose = false)
    
    avg_time = [[Mean() for j=1:3] for i=1:3]
    avg_alloc = [[Mean() for j=1:3] for i=1:3]

    println("\n\nStart\n\n")
    for k=1:10
    for (i,(m,β0)) in enumerate(list_m)
        res, elapsed_time, b_alloc, gctime, memallocs = @timed OPTIM_btr_BFGS(m.f, m.g!, β0, verbose = false, nmax = max_iter, epsilon = epsilon)
        fit!(avg_time[i][1], elapsed_time)
        fit!(avg_alloc[i][1], b_alloc)
        res, elapsed_time, b_alloc, gctime, memallocs = @timed OPTIM_btr_TH(m.f, m.g!, m.H!, β0, verbose = false, nmax = max_iter, epsilon = epsilon)
        fit!(avg_time[i][2], elapsed_time)
        fit!(avg_alloc[i][2], b_alloc)
        res, elapsed_time, b_alloc, gctime, memallocs = @timed OPTIM_btr_HOPS(m.f, m.g_score!, β0, ones(m.n), verbose = false, nmax = max_iter, epsilon = epsilon)
        fit!(avg_time[i][3], elapsed_time)
        fit!(avg_alloc[i][3], b_alloc)
        println("\nNext\n")
    end
    end
    println("time = ", avg_time)
    println("\nalloc = ", avg_alloc)
end

Tests_LS()
println("\nDone")