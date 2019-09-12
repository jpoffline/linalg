package linalg

import (
	"fmt"
	"math"
)

func Sigmoid(x Number) Number {
	return Number(1 / (1 + math.Exp(-float64(x))))
}

func dsigmoid(x Number) Number {
	sig := Sigmoid(x)
	return sig * (1 - sig)
}

func (mtx *NumericMatrix) Dot(vec *NumericVector) (*NumericVector, error) {
	if mtx.dimy != vec.dim {
		return nil, fmt.Errorf("incompatible matrix/vector multiplication attempted")
	}
	res := NewNumericVector(mtx.dimx)

	for r := 0; r < mtx.dimx; r++ {
		for c := 0; c < mtx.dimy; c++ {
			res.values[r] += mtx.values[r][c] * vec.values[c]
		}
	}
	return res, nil
}
