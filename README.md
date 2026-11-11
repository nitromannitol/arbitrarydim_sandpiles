# arbitrarydim_sandpiles

Julia code that stabilizes the Abelian sandpile on a hypercube in any dimension. It accompanies

> Ahmed Bou-Rabee,
> *Dynamic dimensional reduction in the Abelian sandpile*,
> Communications in Mathematical Physics 390 (2022), 933–958;
> [arXiv:2009.05968](https://arxiv.org/abs/2009.05968).

Given a dimension `d`, a size `M` of the hypercube (called side length or radius in the code
comments) and an initial number of chips at each site, the code computes the odometer of the
parallel toppling, the number of topplings at each site, together with its discrete Laplacian
`-2d·v(x) + Σ_{y∼x} v(y)`, which is the change in the number of chips. The computation runs on
the sorted fundamental domain `{M ≥ x₁ ≥ ⋯ ≥ xd ≥ 1}` of the hypercube, so that the number of
sites stored is the `M`-th hypertetrahedral number of dimension `d`. A CPU version and a CUDA
GPU version are provided. The repository supersedes the deprecated
[sandpiles](https://github.com/nitromannitol/sandpiles).

A machine-checked Lean 4 formalization of the paper is in
[Dynamic-Dimensional-Reduction](https://github.com/nitromannitol/Dynamic-Dimensional-Reduction).

## Running

The original instructions ask for [Julia](https://julialang.org/downloads/) 1.5.0 or newer.
The code dates from August 2020 and has not been re-tested since, neither on later Julia
releases nor on later releases of CUDA.jl. There is no `Project.toml`; the packages are:

| Needed for | Packages |
|---|---|
| CPU code ([`sandpile_cpu_multidim_init.jl`](sandpile_cpu_multidim_init.jl)) | none (only `Base.Cartesian`) |
| GPU code ([`sandpile_gpu_multidim_init.jl`](sandpile_gpu_multidim_init.jl)) | `CUDA.jl` and a CUDA-capable GPU |
| 3D visualization (sample 2 of `run_*.jl`) | `Makie.jl`, which must be loaded before the sample is run |
| PNG output (sample 3 of `run_*.jl`) | `FileIO`, `Colors`, `IndirectArrays`, and a PNG backend for `FileIO` |
| writing the output to a text file | `DelimitedFiles` (standard library) |

From the repository directory, start Julia and load the CPU code

```julia
include("sandpile_cpu_multidim_init.jl")
```

or, with `CUDA.jl` installed, the GPU code

```julia
include("sandpile_gpu_multidim_init.jl")
```

Then stabilize, for example, the sandpile with `2d` chips at every site of the hypercube of
size `3` in dimension `6`:

```julia
d = 6
M = 3
init_condition = 2*d
@time v, index, num_iters = getOdometer_cpu(init_condition, M, d)
s = getLaplacian_cpu(v, index, M, d)
```

Here `v` is the odometer on the fundamental domain, `index` the corresponding Cartesian
indices, `num_iters` the number of parallel toppling steps, and `s` the Laplacian of `v`.
The initial condition may also be an array with one entry per site of the fundamental domain.
`fillOut(s, index, M, k)` embeds the result in the full cube and returns its `k`-dimensional
central slice, for `k` equal to 2 or 3. To write an output to a file:

```julia
using DelimitedFiles
writedlm("file_name", s)
```

[`run_cpu.jl`](run_cpu.jl) and [`run_gpu.jl`](run_gpu.jl) contain three samples each: the
constant initial condition in 3D, a single source in 3D with a 3D visualization using Makie.jl
(recommended for small `M` only; for large `M` the comments suggest writing a VTK file and
using ParaView), and the central 2D slices of the stabilization of `2d + k` chips per site for
`d = 2, …, 6` and `k = 0, …, 4`, with the lines that write the PNG files commented out.
According to its comments, the third sample produces a low-resolution version of Table 1 of
the paper.

## Repository layout

| Path | Contents |
|---|---|
| [`sandpile_cpu_multidim_init.jl`](sandpile_cpu_multidim_init.jl) | CPU odometer and Laplacian |
| [`sandpile_gpu_multidim_init.jl`](sandpile_gpu_multidim_init.jl) | the same on a CUDA GPU, with 32-bit integers |
| [`multidim_helpers.jl`](multidim_helpers.jl) | indexing of the fundamental domain, neighbour sums, `fillOut` |
| [`run_cpu.jl`](run_cpu.jl), [`run_gpu.jl`](run_gpu.jl) | sample runs |
| [`Figures/`](Figures/) | 90 precomputed PNG slices named `k<k>_d<d>_kk<kk>_.png`: `2d + k` chips per site in dimension `d` on the hypercube of size `2^kk` |

## Authors and citation

The code is by **Ahmed Bou-Rabee**. To cite it, use [`CITATION.cff`](CITATION.cff), which also
lists the paper above.

## License

The code is licensed under the Apache License 2.0; see [`LICENSE`](LICENSE).
