using Base.Cartesian
include("multidim_helpers.jl")

#######################
## Description 
##		computes the stabilization of the d-dimensional
##		sandpile on the hypercube
## Inputs
## 		init_condition: an array of integers specifying the initial condition  
##		d: dimension of hypercube 
#		M: radius of the hypercube
## Output
##      returns odometer, cartesian indices of the vector, 
##		and number of iterations it took to stabilize  


function getOdometer_cpu(init_condition::Array{Int64,1},M::Int64, d::Int64)
	N = getHyperTetrahedral(d,M);
	v_prev = CUDA.fill(Int64(0),N);
	v = Int64(1)*(init_condition.>=2*(d));
	index_vec = Array{CartesianIndex{Int64(d)},1}(undef, N);
	initializeIndexVec!(index_vec,M);
	v, num_iters = stabilize_cpu!(v,v_prev,init_condition,index_vec,M,d)
	return v, index_vec, num_iters;
end



#######################
## Description 
##		stabilize a sandpile by toppling in order of 
##		of given linear index 
##		cc + e_{ii} - getPositive
## 		cc - e_{ii} - getNegative
## Inputs
##		v,v_prev:  initial current and previous odometers, vectors of integers
## 		init_condition: initial sandpile, vector of integers  
##		index_vector:  array of cartesian indices gotten from initializeIndexVec!(index_vec,M);
##		M: radius of the hypercube
##		d: dimension of the hypercube
## Output
##      returns the terminal odometer and number of iterations it took to get there 

function stabilize_cpu!(v,v_prev,init_condition::Array{Int64,1},index_vec,M::Int64, d::Int64)
	num_iters = 0;
	N = length(v)
	while(v!=v_prev)
		copyto!(v_prev, v)
		topple_cpu!(v, v_prev, init_condition, index_vec,M,Int64(d))
		num_iters+=1;
	end
	return v, num_iters
end


#######################
## Description 
##		topples a sandpile exactly once on the hypercube
##		of given linear index 
##		cc + e_{ii} - getPositive
## 		cc - e_{ii} - getNegative
## Inputs
##		v,v_prev:  initial current and previous odometers, vectors of integers
## 		init_condition: initial sandpile, vector of integers  
##		index_vector:  array of cartesian indices gotten from initializeIndexVec!(index_vec,M);
##		M: radius of the hypercube
##		d: dimension of the hypercube
## Output
##      increments the odometer in place 

function topple_cpu!(v,v_prev,init_condition::Array{Int64,1},index_vec,M::Int64, d::Int64)
	N = size(v,1);
	for curr_ind in 1:N
		@inbounds cc = index_vec[curr_ind]; 
		z =  neighbor_sum(curr_ind, cc, v_prev, M,d);
		@inbounds @fastmath z = div(init_condition[curr_ind] + z,2*d)
		@inbounds v[curr_ind] = z;
	end
end


################################################################################
################################################################################
#### SAME FUNCTIONALITY BUT TYPED FOR A CONSTANT INITIAL CONDITION SO AS TO HAVE LESS STORAGE
################################################################################
################################################################################

function topple_cpu!(v,v_prev,init_condition::Int64,index_vec,M::Int64, d::Int64)
	N = size(v,1);
	for curr_ind in 1:N
		@inbounds cc = index_vec[curr_ind]; 
		z =  neighbor_sum(curr_ind, cc, v_prev, M,d);
		@inbounds @fastmath z = div(init_condition + z,2*d)
		@inbounds v[curr_ind] = z;
	end
end



function stabilize_cpu!(v,v_prev,init_condition::Int64,index_vec,M::Int64, d::Int64)
	num_iters = 0;
	N = length(v)
	while(v!=v_prev)
		copyto!(v_prev, v)
		topple_cpu!(v, v_prev, init_condition, index_vec,M,Int64(d))
		num_iters+=1;
	end
	return v, num_iters
end

function getOdometer_cpu(init_condition::Int64,M::Int64, d)
	N = getHyperTetrahedral(d,M);
	v_prev = fill(Int64(0),N);
	if(init_condition < 2*d) error("Input sandpile is stable") end
	v = fill(Int64(1),N)
	index_vec = Array{CartesianIndex{Int64(d)},1}(undef, N);
	initializeIndexVec!(index_vec,M)
	v, num_iters = stabilize_cpu!(v,v_prev,init_condition,index_vec,M,d)
	return v, index_vec, num_iters;
end
################################################################################
################################################################################




#######################
## Description 
##		computes the Laplacian of a function v
##		on a fundamental domain of the hypercube
## Inputs
##		v:  vector of integers 
##		index_vector:  array of cartesian indices gotten from initializeIndexVec!(index_vec,M);
##		M: radius of the hypercube
##		d: dimension of the hypercube
## Output
##      returns the computed Laplacian in a vector 

function getLaplacian_cpu(v,index_vec,M,d)
	s = zeros(length(v))
	N = length(v);
	laplace_cpu!(s,v,index_vec,M,d)
	return s
end

## helper function for the above
function laplace_cpu!(s,v,index_vec,M,d)
	N = size(v,1);
	for curr_ind in 1:N
		@inbounds cc = index_vec[curr_ind]; 
		z = neighbor_sum(curr_ind, cc, v, M,d);
		@inbounds s[curr_ind] = -2*d*v[curr_ind]+z;
	end
end





