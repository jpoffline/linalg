package main

import (
	"math/rand"

	linalg "github.com/jpoffline/linalg/linearalgebra"
	"github.com/jpoffline/linalg/neural"
)

type TrainingData struct {
	Inputs  []linalg.Number
	Targets []linalg.Number
}

func xordata() {
	neuralnet := neural.New(2, 2, 1)
	td := []TrainingData{
		TrainingData{Inputs: []linalg.Number{1, 1}, Targets: []linalg.Number{0}},
		TrainingData{Inputs: []linalg.Number{1, 0}, Targets: []linalg.Number{1}},
		TrainingData{Inputs: []linalg.Number{0, 1}, Targets: []linalg.Number{1}},
		TrainingData{Inputs: []linalg.Number{0, 0}, Targets: []linalg.Number{0}},
	}

	//rand.NewSource(42)
	for i := 0; i < 100000; i++ {
		data := td[rand.Intn(4)]

		neuralnet.Train(linalg.VectorFromSlice(data.Inputs), linalg.VectorFromSlice(data.Targets))
	}

	out := neuralnet.FeedForward(linalg.VectorFromSlice(td[0].Inputs))
	out.Print()
	out2 := neuralnet.FeedForward(linalg.VectorFromSlice(td[1].Inputs))
	out2.Print()
	out3 := neuralnet.FeedForward(linalg.VectorFromSlice(td[2].Inputs))
	out3.Print()
	out4 := neuralnet.FeedForward(linalg.VectorFromSlice(td[3].Inputs))
	out4.Print()

}

func runnn() {
	/*
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
	*/
	xordata()
}

func main() {
	runnn()
}
