"""
kernel function
- exp kernel
"""
g(t::Float64, model::Hawkes) = model.beta * exp(-model.beta * t)
G(t::Float64, model::Hawkes) = 1 - exp(-model.beta * t)
