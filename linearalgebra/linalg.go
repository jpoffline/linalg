package linearalgebra

import (
	"fmt"
	"math"
)

// Sigmoid computes the sigmoid function at the provided value.
// sigmoid(x) = 1/(1+e^-x)
func Sigmoid(x Number) Number {
	return Number(1 / (1 + math.Exp(-float64(x))))
}

// Dsigmoid computes the derivative of the sigmoid function
// at the provided value.
func Dsigmoid(x Number) Number {
	sig := Sigmoid(x)
	return Dsigmoid2(sig)
}

// Dsigmoid2 computes the pseudo-derivative of the
// sigmoid function, where the provided value
// is assumed to already have been sigmoid-ed.
func Dsigmoid2(sig Number) Number {
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
