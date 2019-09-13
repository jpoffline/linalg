package main

import (
	"fmt"

	linalg "github.com/jpoffline/linalg/linearalgebra"
)

func main2() {
	vec := linalg.NewNumericVector(4)
	vec.Set(0, 3)

	bias := linalg.RandomNumbers(10)
	weights := linalg.RandomNumbers(10)
	z, err := bias.Dot(weights)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(z)

	idm := linalg.NewIdentityMatrix(2)

	twobytwo := linalg.NewNumericMatrix(2, 2)
	twobytwo.Set(0, 0, 2)
	twobytwo.Set(0, 1, 2)
	twobytwo.Set(1, 0, 3)
	twobytwo.Set(1, 1, 2)

	mtx := linalg.NewNumericMatrix(2, 3)
	vvv := linalg.NewNumericVector(3)

	vv, err := mtx.Dot(vvv)
	if err != nil {
		fmt.Println(err)
	}
	vv.Print()

	twobytwo.Print()
	idmsq, err := twobytwo.Mul(idm)
	if err != nil {
		fmt.Println(err)
	}
	idmsq.Print()

	rr := linalg.NewRandomMatrix(2, 2)
	rr.Print()
	rr.Map(func(val linalg.Number) linalg.Number { return val * val })
	rr.Print()

	vec2 := linalg.NewNumericVector(4)
	mtx2 := linalg.NewNumericMatrixFromVector(*vec2)
	mtx2.Print()

}
