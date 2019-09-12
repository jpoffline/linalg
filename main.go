package main

import (
	linalg "github.com/jpoffline/jpnn/linalg"
	"github.com/jpoffline/jpnn/neural"
)

func runnn() {
	neuralnet := neural.New(2, 2, 2)
	inputs := linalg.NewEmptyNumericVector()
	inputs.Push(1)
	inputs.Push(0)

	targets := linalg.NewEmptyNumericVector()
	targets.Push(1)
	targets.Push(0)
	out := neuralnet.FeedForward(inputs)
	out.Print()

	neuralnet.Train(inputs, targets)
}

func main() {
	runnn()
}
