#######################
## Description
##		iteratively computes the d-dimensional simplicial polytope number
## Inputs
## 		d: hypercube dimension 
##		M: hypercube side length
## Output
##      d-dimensional Mth hypertetrahedral (simplicial polytope) number
function getHyperTetrahedral(d,M)
	if(d == 0)
		return 1;
	else
		num = 1;
		for i in reverse(1:d)
			dd = d-i+1
			num = div(num*(M+i-1),dd)
		end
		return num
	end
end


#######################
## Description
##		dynamically writes the cartesian indices of the d-dimensional simplex into an array
## Inputs
## 		index_vec: empty vector of CartesianIndex
##		M: side length of hypercube
## Output
##      initializes in place the cartesian indices of the d-dimensional simplex
@generated function initializeIndexVec!(index_vec::Array{CartesianIndex{d}}, M::Int64) where d
	quote
		@nexprs $d j->(x_{j+1} = M);
		@nloops $d x j->(1:x_{j+1}) (cc = CartesianIndex(@ntuple $d x); index_vec[getLinear(cc,M)] = cc)
	end
end


#######################
## Description
##		computes the linear index of a given cartesian index from the simplex
##		using hypertetrahedral numbers
## Inputs
## 		cc: a CartesianIndex in the simplex
##		M: side length of hypercube
## Output
##      returns linear index of cc in the simplex of side length M
function getLinear(cc,M)
	x_s = Tuple(cc);
	d = length(x_s)
	s = getHyperTetrahedral(d,M);
	for i in 1:length(x_s)
		s = s - getHyperTetrahedral(i,M-x_s[d-i+1]);
	end
	return s
end


## if ind = Linear(x_1, .., x_j, .., x_d)
## computes getLinear(x+e_j) - getLinear(x)
function getPosInd(j, x_j,M,dd)
	d = dd-j;
	return getHyperTetrahedral(d,M-x_j)
end

## same as above except
## computes getLinear(x-e_j) - getLinear(x)
function getNegInd(j, x_j,M,dd)
	d = dd-j;
	return -getHyperTetrahedral(d, M-x_j+1);
end

## compute the appropriate reflection + rotation of 
#T(cc + e_ii)
#ind is the linear index of CartesianIndex cc 
#in the linear array v
#which is the hypercube of side length M 
#in dimension d
@generated function addPositive(ind, ii, cc::CartesianIndex{d}, v, M)  where d
	quote
		curr_val = cc[ii];
		bool = (cc[ii] == M); 
		if(cc[ii] == M)
			return Int32(0); #topple off the boundary
		else
			@nif $d j->(cc[$d-j+1] == cc[ii]) j->(jj=$d-j+1;  ind_ = ind+getPosInd(jj, cc[jj],M,$d);return v[ind_];)
		end
	end
end

## same as above except T[cc - e_ii]
@generated function addNegative(ind, ii, cc::CartesianIndex{d}, v, M) where d
	quote
		if(cc[ii] == 1)
			#ind_ = ind+getPosInd(jj, cc[jj],M,$d)
			ind_ = ind;
			return v[ind_]
		else
			## we are going backwards in this iteration, so no need to reverse the indicies
			@nif $d j->(cc[j] == cc[ii]) j->(ind_ = ind + getNegInd(j,cc[j],M,$d); return v[ind_];)
		end
	end
end



## initalizes vector of cartersian indicies
## corresponding to the fundamental domain of the hypercube
@generated function initializeIndexVec!(index_vec::Array{CartesianIndex{d}}, M::Int64) where d
	quote
		@nexprs $d j->(x_{j+1} = M);
		@nloops $d x j->(1:x_{j+1}) (cc = CartesianIndex(@ntuple $d x); index_vec[getLinear(cc,M)] = cc)
	end
end




## COMPUTES THE MTH HYPERTETRAHEDRAL NUMBER IN DIMENSION D
## ITERATIVELY
function getHyperTetrahedral(d,M)
	if(d == 0)
		return 1;
	else
		num = 1;
		for i in reverse(1:d)
			dd = d-i+1
			num = div(num*(M+i-1),dd)
		end
		return num
	end
end



### WE ASSUME  1 <= x_1 <= ... <= x_d <= M
### so at certain points in the code we need to
### reverse iteration indicies
### this gets the linear index in the fundamental 
### domain associated with hypercube
function getLinear(cc,M)
	x_s = Tuple(cc);
	d = length(x_s)
	s = getHyperTetrahedral(d,M);
	for i in 1:length(x_s)
		s = s - getHyperTetrahedral(i,M-x_s[d-i+1]);
	end
	return s
end


## if ind = Linear(x_1, .., x_j, .., x_d)
## computes getLinear(x+e_j) - getLinear(x)
function getPosInd(j, x_j,M,dd)
	d = dd-j;
	return getHyperTetrahedral(d,M-x_j)
end

## same as above except
## computes getLinear(x-e_j) - getLinear(x)
function getNegInd(j, x_j,M,dd)
	d = dd-j;
	return -getHyperTetrahedral(d, M-x_j+1);
end


## get sum_{y \sim x}(v(y))
### ind is the linear index of cc in index_vec (defined below)
### cc is the cartesian inde associated to the point v[ind]
### v is an array (odometer in practice)
### M is the side length of the hypercube
### d is the dimension
function neighbor_sum(ind, cc, v, M, d)
	val = 0;
	for ii in 1:d
		val = val + addPositive(ind, ii,cc,v,M)
		val = val + addNegative(ind,ii,cc,v,M)
	end
	return val;
end



#######################
## Description of getPositive / getNegative
##		dynamically compute the (appropriately reflected/rotated) nearest neighbor indices 
##		of given linear index 
##		cc + e_{ii} - getPositive
## 		cc - e_{ii} - getNegative
## Inputs
## 		ind: linear index  
##		cc: Cartesian Index
##		ii : direction of neighbor 
##		M: side length of hypercube
## Output
##      linear index of toSimplex(c+(-)e_{ii})

@generated function getPositive(ind, ii, cc::CartesianIndex{d}, M)  where d
	quote
		if(cc[ii] == M)
			return -1; # no neighbor
		else
			@nif $d j->(cc[$d-j+1] == cc[ii]) j->(jj=$d-j+1;  ind_ = ind+getPosInd(jj, cc[jj],M,$d); return ind_;)
		end
	end
end

## helper function for above function
## if ind = getLinear(x_1, .., x_j, .., x_d)
## computes getLinear(x+e_j) - getLinear(x)
function getPosInd(j, x_j,M,dd)
	d = dd-j;
	return getHyperTetrahedral(d,M-x_j)
end


@generated function getNegative(ind, ii, cc::CartesianIndex{d}, M) where d
	quote
		if(cc[ii] == 1)
			return ind; #reflecting boundary condition
		else
			## we are going backwards in this iteration, so no need to reverse the indicies
			@nif $d j->(cc[j] == cc[ii]) j->(ind_ = ind + getNegInd(j,cc[j],M,$d); return ind_;)
		end
	end
end

#helper function for above
## computes getLinear(x-e_j) - getLinear(x)
function getNegInd(j, x_j,M,dd)
	d = dd-j;
	return -getHyperTetrahedral(d, M-x_j+1);
end



#######################
## Description 
##		computes the full symmetrized version of an input 
##		function, v, on a fundamental domain of the hypercube
##		in two or three dimensions
## Inputs
##		v:  vector of integers 
##		index_vector:  array of cartesian indices gotten from initializeIndexVec!(index_vec,M);
##		M: radius of the hypercube
##		d: dimension of the slice (cannot be higher than input)
## Output
##      returns the symmetrized function in a matrix
function fillOut(v,index,M,d)
	if(d > 3) error("Filling out higher dimensions is not currently supported"); end
	A = convertToMat(v, index,M,d)	
	symmetrize!(A)
	if(d == 2)
		A = getFull(A)
	elseif(d == 3)
		A = getFull3D(A)
	end
	symmetrize!(A)
	return A
end



#######################
## Description 
##		embeds the input function onto the hypercube without symmetrizing  
## Inputs
##		s:  vector of integers 
##		index_vector:  array of cartesian indices gotten from initializeIndexVec!(index_vec,M);
##		M: radius of the hypercube
##		d: dimension of the slice
## Output
##      returns the function in a matrix
##		entries which are not filled out are filled by 2*d

function convertToMat(s, index_vec,M,d,k=1)
	inds = Int.(M*ones(d)); 
	A = fill(2*d,inds...)
	for i in 1:length(s)
		cc = Tuple(index_vec[i])
		if(all(cc[1:end-d].==1))
			A[CartesianIndex(cc[end-(d-1):end])] = s[i]; 
		else
			break
		end
	end
	return A;
end



############################################
## helper functions for converting from vector to matrix 
## and symmetrizing on the hypercube 


function symmetrize!(A)
	for xx in CartesianIndices(A)
		xx2 = CartesianIndex(Tuple(sort(collect(Tuple(xx)))))
		A[xx] = A[xx2]
	end
end
## fills out the fundamental domain 
## in 2D assuming you've already run symmetrize
function getFull(A)
  M = size(A,1)
  AA = fill(0, 2*M,2*M)
  AA[M+1:2*M,M+1:2*M] = copy(A);
  AA[M+1:2*M,1:M] = rotl90(A,3)
  AA[1:M, M+1:2*M] = rotl90(A,1)
  AA[1:M,1:M] = rotl90(A,2)
  return AA;
end

# same but in 3D 
function getFull3D(A)
  M = size(A,1)
  AA = fill(0, 2*M, 2*M,2*M)
  for i in M+1:2*M
  	AA[i,:,:] = getFull(A[i-M,:,:])
  	ii = i-M; 
  	AA[ii,:,:] = getFull(A[M-ii+1,:,:])
  end
  return AA;
end





##########################################
##########################################


