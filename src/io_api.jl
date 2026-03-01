abstract type AbstractSweepWriter end

export open_jld2_writer, load_step_jld2

function open_jld2_writer(args...; kwargs...)
    throw(
        ArgumentError("JLD2 backend not loaded. Install JLD2 and `using JLD2` to enable."),
    )
end

function load_step_jld2(args...; kwargs...)
    throw(
        ArgumentError("JLD2 backend not loaded. Install JLD2 and `using JLD2` to enable."),
    )
end

function write_step!(::AbstractSweepWriter, iq, q, λ, V)
    throw(
        ArgumentError(
            "No I/O backend loaded. Install and `using JLD2` to enable JLD2 writers.",
        ),
    )
end

close_writer(::AbstractSweepWriter) = nothing
