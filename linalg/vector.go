package linalg

import (
	"fmt"
)

func (vec *NumericVector) sigmoid() {
	for _, v := range vec.values {
		v = Sigmoid(v)
	}
}

func (v *NumericVector) Push(value Number) {
	v.values = append(v.values, value)
	v.dim++
}

func (v *NumericVector) Value(i int) Number {
	return v.values[i]
}

func (v *NumericVector) Set(i int, val Number) {
	v.values[i] = val
}

func (v *NumericVector) Len() int {
	return len(v.values)
}

func (v1 *NumericVector) Dot(v2 *NumericVector) (Number, error) {
	if v1.dim != v2.dim {
		return 0, fmt.Errorf("incompatible vector scalar product attempted")
	}
	var dot Number
	for i := range v1.values {
		dot += v1.values[i] * v2.values[i]
	}
	return dot, nil
}

func (v1 *NumericVector) Sum(v2 *NumericVector) (*NumericVector, error) {
	if v1.dim != v2.dim {
		return nil, fmt.Errorf("incompatible vector addition attempted")
	}
	res := NewNumericVector(v1.dim)
	for i := range v1.values {
		res.values[i] = v1.Value(i) + v2.Value(i)
	}
	return res, nil
}

func (vec *NumericVector) Print() {
	for _, rr := range vec.values {
		fmt.Println(rr)
	}
}

// Map will apply the provided function to every element of the vector.
func (vec *NumericVector) Map(cb func(Number) Number) *NumericVector {
	res := NewNumericVector(vec.dim)
	for i := range vec.values {
		res.values[i] = cb(vec.values[i])
	}
	return res
}
