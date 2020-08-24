include("sandpile_gpu_multidim_init.jl")

#############
## sample 1
## -------
## stabilize the all 7s sandpile in 3D 
## on a hypercube of side length 2^5
d = 3
M= 2^6
init_condition = Int32(6) #GPUs run faster with 32-bit integers
@time v, index, num_iters = getOdometer_gpu(init_condition,M,d)
s = Array(getLaplacian_gpu(v,index,M,d))
S_2D = fillOut(s,Array(index),M,2)# extract the 2D center slice of the Laplacian
S_3D = fillOut(s,Array(index),M,3)# extract the 3D central slice of the Laplacian




#############
## sample 2
## -------
## stabilize the single source sandpile in 3D 
## with ~2^20 number of chips at the center
d = 3
M= 2^6
N = getHyperTetrahedral(d,M);
init_condition = CUDA.fill(Int32(0),N)
init_condition[1] = Int32(d*N); 
@time v, index, num_iters = getOdometer_gpu(init_condition,M,d)
s = Array(getLaplacian_gpu(v,index,M,d))
s = s + Array(init_condition)
S = fillOut(s,Array(index),M,3)
T = fillOut(Array(v), Array(index),M,3)



## visualize the output using Makie
## if you do not already have Makie run 
## > ]  > add Makie


## WARNING  ## 
## This method of visualization only works well 
## for relatively small M (M <= 2^6)
## for large M we suggest writing to a VTK file 
## and using paraview

## extract the positions of grains
positions0 = Array{Point{3,Int64},1}()
positions1 = Array{Point{3,Int64},1}()
positions2 = Array{Point{3,Int64},1}()
positions3 = Array{Point{3,Int64},1}()
positions4 = Array{Point{3,Int64},1}()
positions5 = Array{Point{3,Int64},1}()
for x in 1:size(S,1)
	for y in 1:size(S,2)
		for z in 1:size(S,3)
			if(T[x,y,z] == 0) continue; end
			s = S[x,y,z]
			points = Point3(x,y,z)
			if(s == 0)
				push!(positions0,points)
			elseif(s==1)
				push!(positions1,points)
			elseif(s == 2)
				push!(positions2,points)
			elseif(s ==3)
				push!(positions3,points)

			elseif(s == 4)
				push!(positions4,points)

			elseif(s == 5)
				push!(positions5,points)
			end
		end
	end
end		
meshscatter(positions0,markersize=0.5,color=:red)
meshscatter!(positions1,markersize=0.5,color=:blue)
meshscatter!(positions2,markersize=0.5,color=:black)
meshscatter!(positions3,markersize=0.5,color=:yellow)
meshscatter!(positions4,markersize=0.5,color=:white)
meshscatter!(positions5,markersize=0.5,color=:orange)







#############
## sample 3
## -------
## extract the center slice of the stabilization
## of 2d + k for  d = 2 ... 6
## and k = 0 ... 4
## this produces Table 1 in "Dynamic Dimensional Reduction ..."

## load the packages for writing the sandpile 
## to a png file 
## if you do no thave them run in the REPL
## > ] add FileIO
## > ] add Colors
## > ] add IndirectArrays

using FileIO,Colors, IndirectArrays
c0 = colorant"lightblue"
c1 = colorant"red"
c2 = colorant"green"
c3=  colorant"black"
c4=  colorant"darkblue"
c5=  colorant"white"
c6=  colorant"gray"
c7 = colorant"orange"
c8 = colorant"darkred"
c9 = colorant"yellow"
c10 = colorant"pink"
c11 = colorant"black"
c_p = [c0, c1, c2, c3, c4,c5,c6,c7,c8,c9,c10,c11]

kk = 4
M = 2^kk

for d in 2:6
	for k in 0:4
		println("Dimension $d, k: $k")
		slice_dimension = 2
		init_condition = Int32(2*d+k)
		v,index,n = getOdometer_gpu(init_condition,M,d)
		s = Array(getLaplacian_gpu(v,index,M,d))
		S = fillOut(s,index,M,slice_dimension)
		S = abs.(S)
		c_p = c_p
		# comment out and replace str 
		# with your desired output directory
		# to save the file to png
		#str = "Figures/k$(k)_d$(d)_kk$(kk)_.png";
		#A_c = IndirectArray(S, c_p)
		#save(str, A_c)
	end
end
