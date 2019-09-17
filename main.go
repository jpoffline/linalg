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

func XORTrainingData() []neural.TrainingData {
	return []neural.TrainingData{
		neural.TrainingData{Inputs: []linalg.Number{1, 1}, Targets: []linalg.Number{0}},
		neural.TrainingData{Inputs: []linalg.Number{1, 0}, Targets: []linalg.Number{1}},
		neural.TrainingData{Inputs: []linalg.Number{0, 1}, Targets: []linalg.Number{1}},
		neural.TrainingData{Inputs: []linalg.Number{0, 0}, Targets: []linalg.Number{0}},
	}
}

func nnXOR() {

	netArch := neural.NetworkArchitecture(2, 4, 1)

	neuralnet := neural.New(netArch)
	neuralnet.SetOutputLoc("output")

	td := XORTrainingData()
	neuralnet.Train(td, 100000)

	neuralnet.Serialise("nn.json")

	for _, d := range td {
		xorCheck(neuralnet, d.Inputs, d.Targets)
	}

}

func main() {
	nnXOR()
}
