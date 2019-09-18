package main

import (
	"fmt"

	"github.com/jpoffline/linalg/neural"
)

func xorCheck(nn *neural.NeuralNet, i, t neural.Vector) {
	out := nn.Predict(i)
	fmt.Printf("Input: %v, output: %v, expected: %v\n", i, out, t)
}

func XORTrainingData() []neural.TrainingData {
	return []neural.TrainingData{
		neural.TrainingData{Inputs: neural.Vector{1, 1}, Targets: neural.Vector{0}},
		neural.TrainingData{Inputs: neural.Vector{1, 0}, Targets: neural.Vector{1}},
		neural.TrainingData{Inputs: neural.Vector{0, 1}, Targets: neural.Vector{1}},
		neural.TrainingData{Inputs: neural.Vector{0, 0}, Targets: neural.Vector{0}},
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
