

mutable struct SubBatch <: Batch
    batch::Batch
    size::Int64
    n_ind::Int64
    state_next
    function SubBatch(batch, n::Int64, n_max::Int64)
        n = new()
        n.batch = batch
        n.size = n
        return n
    end
end

function iterate(b::SubBatch, state::Int64 = 1)
    if state > b.size
        
        return nothing
    else
        tmp = iterate(b.batch, b.state_next)
        if tmp == nothing
            ind, st = iterate(b.batch)
            b.state_next = st
            return ind, state +1
        else
            ind, st = tmp
            b.state_next = st
            return ind, state +1
        end
    end
end