module HillFunctionsJLD2Ext

using HillFunctions
using JLD2

export JLD2Writer, open_jld2_writer, load_step_jld2

struct JLD2Writer <: HillFunctions.AbstractSweepWriter
    path::String
    nsteps::Int
end

"""
    open_jld2_writer(path; nsteps, meta=NamedTuple())

Create/initialize a JLD2 sweep file. Stores metadata and allocates a q index vector.
"""
function HillFunctions.open_jld2_writer(path::AbstractString; nsteps::Int, meta::NamedTuple=NamedTuple())
    jldopen(path, "w") do f
        for (k, v) in pairs(meta)
            f["meta/$(k)"] = v
        end
        f["meta/bigfloat_precision_bits_at_write"] = precision(BigFloat)
        f["meta/nsteps"] = nsteps
    end
    return JLD2Writer(String(path), nsteps)
end

function HillFunctions.write_step!(w::JLD2Writer, iq, q, λ, V)
    jldopen(w.path, "a") do f
        g = "step/$(lpad(iq, 6, '0'))"
        f["$g/q"]     = q
        f["$g/evals"] = λ
        f["$g/evecs"] = V
    end
    return nothing
end

function HillFunctions.load_step_jld2(path::AbstractString, iq::Int)
    jldopen(path, "r") do f
        g = "step/$(lpad(iq, 6, '0'))"
        q = f["$g/q"]
        λ = f["$g/evals"]
        V = f["$g/evecs"]
        return q, λ, V
    end
end
end # module