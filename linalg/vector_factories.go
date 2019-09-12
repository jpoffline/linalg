package linalg

import "math/rand"

func NewNumericVector(cap int) *NumericVector {
	vec := NumericVector{}
	vec.values = make([]Number, cap)
	vec.dim = cap
	return &vec
}

func NewEmptyNumericVector() *NumericVector {
	vec := NumericVector{}
	return &vec
}

func RandomNumbers(n int) *NumericVector {
	nums := NewNumericVector(n)
	for i := 0; i < n; i++ {
		nums.values[i] = Number(rand.Float64()*2 - 1)
	}
	return (nums)
}

func NewRandomVector(cap int) *NumericVector {
	return RandomNumbers(cap)
}
