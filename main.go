package main

import (
	linalg "github.com/jpoffline/linalg/linearalgebra"
	"github.com/jpoffline/linalg/neural"
)

func runnn() {
	neuralnet := neural.New(2, 2, 1)
	inputs := linalg.NewNumericVector(2)
	inputs.Set(0, 1)
	inputs.Set(1, 2)
	neuralnet.FeedForward(inputs)
}

func main() {
	runnn()
}
