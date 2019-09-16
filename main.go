package main

import (
	"fmt"

	linalg "github.com/jpoffline/linalg/linearalgebra"
	"github.com/jpoffline/linalg/neural"
)

func xorCheck(nn *neural.NeuralNet, i, t []linalg.Number) {
	out := nn.FeedForward(i)
	fmt.Printf("Input: %v, output: %v, expected: %v\n", i, out, t)
}

func nnXOR() {
	neuralnet := neural.New(2, 2, 1)
	td := []neural.TrainingData{
		neural.TrainingData{Inputs: []linalg.Number{1, 1}, Targets: []linalg.Number{0}},
		neural.TrainingData{Inputs: []linalg.Number{1, 0}, Targets: []linalg.Number{1}},
		neural.TrainingData{Inputs: []linalg.Number{0, 1}, Targets: []linalg.Number{1}},
		neural.TrainingData{Inputs: []linalg.Number{0, 0}, Targets: []linalg.Number{0}},
	}
	neuralnet.Train(td, 100000)

	neuralnet.Serialise("nn.json")

	for _, d := range td {
		xorCheck(neuralnet, d.Inputs, d.Targets)
	}

}

func runnn() {

	nnXOR()
}

func main() {
	runnn()
}
