"""
kernel function
- exp kernel
"""
const lambda = 0.04
g(t::Float64) = lambda * exp(-lambda * t)
G(t::Float64) = 1 - exp(-lambda * t)
