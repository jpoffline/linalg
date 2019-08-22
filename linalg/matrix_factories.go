package linalg

import "math/rand"

func NewNumericMatrix(dimx, dimy int) *NumericMatrix {
	mtx := NumericMatrix{}
	for i := 0; i < dimx; i++ {
		mtx.values = append(mtx.values, make([]Number, dimy))
	}
	mtx.dimx = dimx
	mtx.dimy = dimy
	return &mtx
}

func NewIdentityMatrix(dim int) *NumericMatrix {
	mtx := NewNumericMatrix(dim, dim)
	for i := 0; i < dim; i++ {
		mtx.values[i][i] = 1
	}
	return mtx
}

func NewRandomMatrix(dimx, dimy int) *NumericMatrix {
	mtx := NewNumericMatrix(dimx, dimy)
	for i := 0; i < dimx; i++ {
		for j := 0; j < dimy; j++ {
			mtx.values[i][j] = Number(rand.Float64())
		}
	}
	return mtx
}
