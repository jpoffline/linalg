package linalg

import "fmt"

// Mul will multiply the provided matrices together.
// We check for dimensional-compatibility.
func (mtx *NumericMatrix) Mul(mtx1 *NumericMatrix) (*NumericMatrix, error) {
	if mtx.dimx != mtx1.dimy {
		return nil, fmt.Errorf("incompatible matrix multiplication attempted")
	}

	res := NewNumericMatrix(mtx.dimx, mtx1.dimy)

	for i := 0; i < res.dimx; i++ {
		for j := 0; j < res.dimy; j++ {
			for k := 0; k < mtx1.dimx; k++ {
				res.values[i][j] += mtx.values[i][k] * mtx1.values[k][j]
			}
		}
	}

	return res, nil
}

// Value will return the value of the matrix at the
// provided i,j values.
func (mtx *NumericMatrix) Value(i, j int) Number {
	return mtx.values[i][j]
}

// Set will set the value of the matrix at i,j to be
// the provided value.
func (mtx *NumericMatrix) Set(i, j int, val Number) {
	mtx.values[i][j] = val
}

// Print will crudely print out the matrix.
func (mtx *NumericMatrix) Print() {
	for _, rr := range mtx.values {
		fmt.Println(rr)
	}
}

// Len returns the dimx and dimy of the matrix.
func (mtx *NumericMatrix) Len() (int, int) {
	return mtx.dimx, mtx.dimy
}
