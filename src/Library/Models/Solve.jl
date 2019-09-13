function solve_BTR_RA(m::Models, β0::AbstractArray, N0::Int64, coeff::Float64;
        sample_coeff::Float64 = 1.0, criterion::Int64 = 2, verbose::Bool = false)
    f_values::Array = []
    nb_iter::Int64 = 0
    m.sample_size = N0
    m.subsample = Int64(round(sample_coeff*m.sample_size))
    m.q_student = quantile(TDist(m.sample_size-1), 1-m.α)
    while m.sample_size <= m.n_train
        if verbose
            println("\n$(nb_iter+=1) : N = $(m.sample_size)\n")
        end
        if m.sample_size < m.n_train || criterion!=3
            if m.sample_size == m.n_train
                m.ϵ = 0.0
            end
            state, acc = OPTIM_btr_HOPS(m.f, m.score!, β0, m.batch.weights[1:m.subsample], verbose = verbose, epsilon = 1e-20, tTest = m.tTest)
        else
             state, acc = OPTIM_btr_HOPS(m.f_train, m.score!, β0, m.batch.weights[1:m.subsample], verbose = verbose, epsilon = 1e-20, tTest = m.validation)        
        end
        β0 = copy(state.x)
        f_values = copy([f_values; acc])
        m.sample_size = Int64(round(m.sample_size*coeff))
        m.subsample = Int64(round(sample_coeff*m.sample_size))
        m.q_student = quantile(TDist(m.sample_size-1), 1-m.α)
    end
    return β0, f_values
end

function solve_BTR_HOPS(m::Models; x0::AbstractArray = zeros(m.dim), verbose::Bool = false, nmax::Int64 = 1000, criterion::Int64 = 1)
    w = Int64[]
    for ind in m.batch
        push!(w, ind.n_sim)
    end
    if criterion == 1
        return OPTIM_btr_HOPS(m.f, m.score!, x0, w, verbose = verbose, nmax = nmax)
    end
    if criterion == 2
        return OPTIM_btr_HOPS(m.f, m.score!, x0, w, verbose = verbose, nmax = nmax, epsilon = 1e-20, tTest = m.tTest)
    end
    if criterion == 3
        return OPTIM_btr_HOPS(m.f, m.score!, x0, w, verbose = verbose, nmax = nmax, epsilon = 1e-20, tTest = m.validation)
    end 
end

function solve_BTR_BHHH(m::Models; x0::AbstractArray = zeros(m.dim), verbose::Bool = false, nmax::Int64 = 1000, criterion::Int64 = 1)
    if criterion == 1
        return OPTIM_btr_TH(m.f, m.∇f!, m.bhhh!, x0, verbose = verbose, nmax = nmax)
    end
    if criterion == 2
        return OPTIM_btr_TH(m.f, m.∇f!, m.bhhh!, x0, verbose = verbose, nmax = nmax, epsilon = 1e-20, tTest = m.tTest)
    end
    if criterion == 3
        return OPTIM_btr_TH(m.f, m.∇f!, m.bhhh!, x0, verbose = verbose, nmax = nmax, epsilon = 1e-20, tTest = m.validation)
    end
end

function solve_BTR_BFGS(m::Models; x0::AbstractArray = zeros(m.dim), verbose::Bool = false, nmax::Int64 = 1000, criterion::Int64 = 1)
    if criterion == 1
        return OPTIM_btr_BFGS(m.f, m.∇f!, x0, verbose = verbose, nmax = nmax)
    end
    if criterion == 2
        return OPTIM_btr_BFGS(m.f, m.∇f!, x0, verbose = verbose, nmax = nmax, epsilon = 1e-20, tTest = m.tTest)
    end
    if criterion == 3
        return OPTIM_btr_BFGS(m.f, m.∇f!, x0, verbose = verbose, nmax = nmax, epsilon = 1e-20, tTest = m.validation)
    end
end

function solve_BTR_TH(m::Models; x0::AbstractArray = zeros(m.dim), verbose::Bool = false, nmax::Int64 = 1000)
    return OPTIM_btr_TH(m.f, m.∇f!, m.Hf!, x0, verbose = verbose, nmax = nmax)
end

function solve_RSAG(lm::Models; verbose::Bool = false, nmax::Int64 = 1000)
    return OPTIM_AGRESSIVE_RSAG(lm.f, lm.∇f!, lm.batch, lm.f::Function; x0 = zeros(m.dim), L = 1.0, nmax = 500, 
        ϵ = 1e-4, verbose = false, n_test = 500, n_optim = 100)
end

function solve_AG(m::Models; verbose::Bool = false, nmax::Int64 = 1000)
    return OPTIM_AGRESSIVE_AG(m.f, m.∇f!, x0 = zeros(m.dim), L = 1.0, nmax = nmax, 
        ϵ = 1e-4, verbose = verbose)
end

function solve_BFGS(m::Models; verbose::Bool = false, nmax::Int64 = 1000)
    return OPTIM_BFGS(m.f, m.∇f!, x0 = zeros(m.dim), nmax = nmax, 
        ϵ = 1e-4, verbose = verbose)
end

"""
chose between :

  'solve_BTR_BFGS'

  'solve_BTR_TH'

  'solve_RSAG'
  
  'solve_AG'
  
  'solve_BFGS'
"""
solve = solve_BTR_BFGS
