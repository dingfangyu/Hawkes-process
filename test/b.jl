#=
b:
- Julia version: 1.2.0
- Author: ding
- Date: 2019-09-24
=#

# println(2)

include("a.jl")
include("a.jl")

function b(x)
    2x
end

println(methods(a))