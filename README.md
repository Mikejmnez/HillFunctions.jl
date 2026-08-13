# HillFunctions

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Mikejmnez.github.io/HillFunctions.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Mikejmnez.github.io/HillFunctions.jl/dev/)
[![Build Status](https://github.com/Mikejmnez/HillFunctions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Mikejmnez/HillFunctions.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Mikejmnez/HillFunctions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Mikejmnez/HillFunctions.jl)


Computes the eigenvalues and eigenfunctions of a type of Hill's equation studied by Strutt M (1948). The even eigenvalue problem is determined by

$$\frac{d^2\phi_{2n}}{dy^2} + \left[a_{2n} - qU(2y)\right]\phi_{2n} = 0$$

subject to periodic boundary conditions

$$\left.\frac{d\phi_{2n}}{dy}\right|_{y=0} = \left.\frac{\phi_{2n}}{dy}\right|_{y=\pi} = 0$$

and the odd eigenvalue problem is determined by

$$\frac{d^2\phi_{2n+2}}{dy^2} + \left[b_{2n+2} - qU(2y)\right]\phi_{2n+2}$$
subjet to the boundary conditions

$$\left.\phi_{2n+2}\right|_{y=0} = \left.\phi_{2n+2}\right|_{y=\pi} = 0$$

The function $U(2y)$ is an integrable and periodic, and $q$ is a purely imaginary parameter.


### Usage
```julia
using HillFunctions

Ny = 100
y = range(0, π; length=Ny+1)[1:end-1]
q = ComplexF64(0,2020)

Soln = solve(q, y, Neumann(), [0.5])

# get the even eigefunctions and eigenvalues
an, Φe = Soln.an, Soln.Φe

```