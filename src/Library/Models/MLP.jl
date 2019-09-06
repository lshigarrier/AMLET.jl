mutable struct MLP <: Models
    batch::Batch
    dim::Int64
    n_train::Int64
    n_valid::Int64
    sample_size::Int64
    subsample::Int64
    q_student::Float64
    α::Float64
    ϵ::Float64

    f_old::Float64
    old_var::Float64
    var::Float64
    cov::Float64
    data::Array{Float64}
    
    f::Function 
    f_valid::Function
    f_train::Function
    score!::Function             
    ∇f!::Function             
    bhhh!::Function
    Hf!::Function
    tTest::Function
    validation::Function

    function MLP(b::Batch, dim::Int64)
        m = new()
        m.dim = dim
        m.batch = b
        n_total = sum(b.weights)
        m.data = zeros(Int(n_total))
        m.var = 0
        m.f_old = 0
        return m
    end
end
