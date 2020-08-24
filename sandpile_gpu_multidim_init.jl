using Base.Cartesian, CUDA
include("multidim_helpers.jl")

const numthreads = 2^8; ## change depending on what your GPU capability is


#######################
## Description 
##		computes the stabilization of the d-dimensional
##		sandpile on the hypercube using the GPU
## Inputs
## 		init_condition: an array of integers specifying the initial condition  
##		d: dimension of hypercube 
#		M: radius of the hypercube
## Output
##      returns odometer, cartesian indices of the vector, 
##		and number of iterations it took to stabilize  

function getOdometer_gpu(init_condition::CuArray{Int32,1},M, d)
	if(all(init_condition .< 2*d)) error("Input sandpile is stable"); end
	N = getHyperTetrahedral(d,M);
	v_prev = CUDA.fill(Int32(0),N);
	v = Int32(1)*(init_condition.>=2*(d));
	index_vec = Array{CartesianIndex{Int64(d)},1}(undef, N);
	initializeIndexVec!(index_vec,M);
	index_vec = CuArray(index_vec)
	v, num_iters = stabilize_gpu!(v,v_prev,init_condition,index_vec,M,d)
	return v, index_vec, num_iters;
end





#######################
## Description 
##		stabilize a sandpile by toppling in order of 
##		of given linear index 
##		cc + e_{ii} - getPositive
## 		cc - e_{ii} - getNegative
## Inputs
##		v,v_prev:  CUDA initial current and previous odometers, vectors of integers
## 		init_condition: initial sandpile, vector of integers  
##		index_vector:  array of cartesian indices gotten from initializeIndexVec!(index_vec,M);
##		M: radius of the hypercube
##		d: dimension of the hypercube
## Output
##      returns the terminal odometer and number of iterations it took to get there 

function stabilize_gpu!(v,v_prev,init_condition,index_vec,M, d)
	num_iters = 0;
	N = length(v)
	numblocks = ceil(Int, N/numthreads)
	while(v!=v_prev)
		copyto!(v_prev, v)
		CUDA.@sync @cuda threads=numthreads blocks = numblocks topple_gpu!(v, v_prev, init_condition, index_vec,M,Int32(d))
		num_iters+=1;
	end
	return v, num_iters

end

## helper kernel function for the above  
function topple_gpu!(v,v_prev,init_condition,index_vec,M, d)
	N = size(v,1);
	#compute offset to beginnign of block
	#and increment by thread ID
	index_x = (blockIdx().x-1)*blockDim().x + threadIdx().x
	stride_x = blockDim().x*gridDim().x;
	
	for curr_ind in index_x:stride_x:N
		@inbounds cc = index_vec[curr_ind]; 
		z =  neighbor_sum(curr_ind, cc, v_prev, M,d);
		@inbounds @fastmath z = div(init_condition[curr_ind] + z,2*d)
		@inbounds v[curr_ind] = z;
	end
	return nothing; 
end



################################################################################
################################################################################
#### SAME FUNCTIONALITY BUT TYPED FOR A CONSTANT INITIAL CONDITION SO AS TO HAVE LESS STORAGE
################################################################################
################################################################################

function topple_gpu!(v,v_prev,init_condition::Int32,index_vec,M, d)
	N = size(v,1);
	#compute offset to beginnign of block
	#and increment by thread ID
	index_x = (blockIdx().x-1)*blockDim().x + threadIdx().x
	stride_x = blockDim().x*gridDim().x;
	
	for curr_ind in index_x:stride_x:N
		@inbounds cc = index_vec[curr_ind]; 
		z =  neighbor_sum(curr_ind, cc, v_prev, M,d);
		@inbounds @fastmath z = div(init_condition + z,2*d)
		@inbounds v[curr_ind] = z;
	end
	return nothing; 
end


function stabilize_gpu!(v,v_prev,init_condition::Int32,index_vec,M, d)
	num_iters = 0;
	N = length(v)
	numblocks = ceil(Int, N/numthreads)
	while(v!=v_prev)
		copyto!(v_prev, v)
		CUDA.@sync @cuda threads=numthreads blocks = numblocks topple_gpu!(v, v_prev, init_condition, index_vec,M,Int32(d))
		num_iters+=1;
	end
	return v, num_iters

end

function getOdometer_gpu(init_condition::Int32,M, d)
	N = getHyperTetrahedral(d,M);
	v_prev = CUDA.fill(Int32(0),N);
	if(init_condition < 2*d) error("Input sandpile is stable"); end
	v = CUDA.fill(Int32(1),N)
	index_vec = Array{CartesianIndex{Int64(d)},1}(undef, N);
	initializeIndexVec!(index_vec,M)
	index_vec = CuArray(index_vec)
	v, num_iters = stabilize_gpu!(v,v_prev,init_condition,index_vec,M,d)
	return v, index_vec, num_iters;
end

################################################################################
################################################################################






#######################
## Description 
##		computes the Laplacian of a function v
##		on a fundamental domain of the hypercube
## Inputs
##		v:  CUDA vector of integers 
##		index_vector:  CUDA vector of Cartesian Indices gotten from: 
##					index_vec = Array{CartesianIndex{Int64(d)},1}(undef, N);
##					initializeIndexVec!(index_vec,M)
##					index_vec = CuArray(index_vec)
##
##		M: radius of the hypercube
##		d: dimension of the hypercube
## Output
##      returns the computed Laplacian in a CUDA vector 

function getLaplacian_gpu(v,index_vec,M,d)
	s = CUDA.fill(Int32(0),length(v))
	N = length(v);
	numblocks = ceil(Int, N/numthreads)
	CUDA.@sync @cuda threads=numthreads blocks = numblocks laplace_gpu!(s,v,index_vec,M,d)
	return s
end

## helper function for the above
function laplace_gpu!(s,v,index_vec,M,d)
	N = size(v,1);
	#compute offset to beginnign of block
	#and increment by thread ID
	index_x = (blockIdx().x-1)*blockDim().x + threadIdx().x
	stride_x = blockDim().x*gridDim().x;
	
	for curr_ind in index_x:stride_x:N
		#@cuprintln("Index in array: ", curr_ind);
		@inbounds cc = index_vec[curr_ind]; 
		z = neighbor_sum(curr_ind, cc, v, M,d);
		@inbounds s[curr_ind] = -2*d*v[curr_ind]+z;
	end
	return nothing; 
end