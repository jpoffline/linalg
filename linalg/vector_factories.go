package linalg

import "math/rand"

func NewNumericVector(cap int) *NumericVector {
	vec := NumericVector{}
	vec.values = make([]Number, cap)
	vec.dim = cap
	return &vec
}

func RandomNumbers(n int) *NumericVector {
	nums := NewNumericVector(n)
	for i := 0; i < n; i++ {
		nums.Push(Number(rand.Float64()))
	}
	return (nums)
}
