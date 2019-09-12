package linalg

import "math/rand"

// NewNumericMatrix returns a new matrix with
// the provided dimensions. Note that all values
// are intialised to the Numeric zero.
func NewNumericMatrix(dimx, dimy int) *NumericMatrix {
	mtx := NumericMatrix{}
	for i := 0; i < dimx; i++ {
		mtx.values = append(mtx.values, make([]Number, dimy))
	}
	mtx.dimx = dimx
	mtx.dimy = dimy
	return &mtx
}

func NewNumericMatrixFromVector(vec NumericVector) *NumericMatrix {
	mtx := NewNumericMatrix(vec.dim, 1)
	for i := 0; i < vec.dim; i++ {
		mtx.values[i][0] = vec.values[i]
	}
	return mtx
}

// NewIdentityMatrix returns an identity matrix with
// the provided dimension (these are square, diagonal
// contain zeros everywhere except the diagonal, which are unity).
func NewIdentityMatrix(dim int) *NumericMatrix {
	mtx := NewNumericMatrix(dim, dim)
	for i := 0; i < dim; i++ {
		mtx.values[i][i] = 1
	}
	return mtx
}

// NewRandomMatrix will return a matrix with the
// provided dimensions, whose elements are
// random numbers.
func NewRandomMatrix(dimx, dimy int) *NumericMatrix {
	mtx := NewNumericMatrix(dimx, dimy)
	for i := 0; i < dimx; i++ {
		for j := 0; j < dimy; j++ {
			mtx.values[i][j] = Number(rand.Float64()*2 - 1)
		}
	}
	return mtx
}
