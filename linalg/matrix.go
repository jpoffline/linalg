package linalg

import "fmt"

// Mul will multiply the provided matrices together.
// We check for dimensional-compatibility.
func (mtx *NumericMatrix) Mul(mtx1 *NumericMatrix) (*NumericMatrix, error) {
	if mtx.dimy != mtx1.dimx {
		mtx.Print()
		mtx1.Print()
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

func (mtx *NumericMatrix) Add(mtx1 *NumericMatrix) (*NumericMatrix, error) {
	if mtx.dimx != mtx1.dimx {
		return nil, fmt.Errorf("incompatible matrix addition attempted: dimx")
	}

	if mtx.dimy != mtx1.dimy {
		mtx.Print()
		mtx1.Print()
		return nil, fmt.Errorf("incompatible matrix addition attempted: dimy")
	}

	res := NewNumericMatrix(mtx.dimx, mtx.dimy)
	for i := 0; i < mtx.dimx; i++ {
		for j := 0; j < mtx.dimy; j++ {
			res.values[i][j] = mtx.values[i][j] + mtx1.values[i][j]
		}
	}
	return res, nil
}

// Map will apply the provided function to every element of the matrix.
func (mtx *NumericMatrix) Map(cb func(Number) Number) {
	for i := 0; i < mtx.dimx; i++ {
		for j := 0; j < mtx.dimy; j++ {
			mtx.values[i][j] = cb(mtx.values[i][j])
		}
	}
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
	fmt.Printf("matrix has dim %v x %v\n", mtx.dimx, mtx.dimy)
	for _, rr := range mtx.values {
		fmt.Println(rr)
	}
}

// Len returns the dimx and dimy of the matrix.
func (mtx *NumericMatrix) Len() (int, int) {
	return mtx.dimx, mtx.dimy
}
