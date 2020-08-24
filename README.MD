Instructions

1. install Julia 1.5.0+
	
	https://julialang.org/downloads/

2. 	start Julia and run 

		(for CPU code with no dependencies)

		>include("PATHTOFILE/sandpile_cpu_multidim_init.jl")


		(for GPU code install CUDA.jl and run)

		>include("PATHTOFILE/sandpile_gpu_multidim_init.jl")
		


3.  compute arbitrary dimensional sandpiles on the hypercube by running, for example

		>d = 6
		>M= 3
		>init_condition = 2*d
		>@time v, index, num_iters = getOdometer_cpu(init_condition,M,d)
		>s = getLaplacian_cpu(v,index,M,d)

--- 
	- see run_gpu.jl  and run_cpu.jl for some sample code

	- to write the outputs to a file run

		> using DelimitedFiles
		> writedlm("file_name", s) 

	- for 3D viz in Julia we recommend installing Makie.jl 
	and following the recipe in sample 2 in run_*.jl

	- for writing 2D sandpiles to PNG files we recommend
	installing FileIO,Colors, IndirectArrays and then following 
	the recipe in sample 3 in run_*.jl 


