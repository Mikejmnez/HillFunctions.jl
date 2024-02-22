# HillFunctions

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Mikejmnez.github.io/HillFunctions.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Mikejmnez.github.io/HillFunctions.jl/dev/)
[![Build Status](https://github.com/Mikejmnez/HillFunctions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Mikejmnez/HillFunctions.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Mikejmnez/HillFunctions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Mikejmnez/HillFunctions.jl)


Computes the eigenvalues and eigenfunctions of a type of Hill's equation studied by Strutt M (1948). The even eigenvalue problem is determined by

```math
\frac{d^2\phi_{2n}}{dy^2} + \left[a_{2n} - qU(2y)\right]\phi_{2n} = 0
```
subject to periodic boundary conditions
```math
\left.\frac{d\phi_{2n}}{dy}\right|_{y=0} = \left.\frac{\phi_{2n}}{dy}\right|_{y=\pi} = 0
```
and the odd eigenvalue problem is determined by

```math
\frac{d^2\phi_{2n+2}}{dy^2} + \left[b_{2n+2} - qU(2y)\right]\phi_{2n+2}
```
subjet to the boundary conditions
```math
\left.\phi_{2n+2}\right|_{y=0} = \left.\phi_{2n+2}\right|_{y=\pi} = 0
```

The function $U(2y)$ is an integrable and periodic.

