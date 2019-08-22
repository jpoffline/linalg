package linalg

import "fmt"

func (mtx1 *NumericMatrix) Mul(mtx2 *NumericMatrix) (*NumericMatrix, error) {
	if mtx1.dimx != mtx2.dimy {
		return nil, fmt.Errorf("incompatible matrix multiplication attempted")
	}

	res := NewNumericMatrix(mtx1.dimx, mtx2.dimy)

	for i := 0; i < res.dimx; i++ {
		for j := 0; j < res.dimy; j++ {
			for k := 0; k < mtx2.dimx; k++ {
				res.values[i][j] += mtx1.values[i][k] * mtx2.values[k][j]
			}
		}
	}

	return res, nil
}

func (mtx *NumericMatrix) Value(i, j int) Number {
	return mtx.values[i][j]
}

func (mtx *NumericMatrix) Set(i, j int, val Number) {
	mtx.values[i][j] = val
}

func (mtx *NumericMatrix) Print() {
	for _, rr := range mtx.values {
		fmt.Println(rr)
	}
}
