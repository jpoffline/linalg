package neural

import (
	linalg "github.com/jpoffline/linalg/linearalgebra"
)

type Meta struct {
	numInputs, numHidden, numOutput int
	trainCount                      int
	learningRate                    linalg.Number
}

// NeuralNet is the main neural net struct, containing
// all the required data.
type NeuralNet struct {
	OutputData
	meta       Meta
	trainCount int
	layers     []neurallayer
}

type neurallayer struct {
	weights     Weights
	bias        Bias
	activations Activations
}

type OutputData struct {
	Loc string
}

type Number = linalg.Number

type Vector = []Number

// Weights is the type to hold the weights connecting neurons
type Weights = *linalg.NumericMatrix

// Bias is the type to hold the biases
type Bias = *linalg.NumericMatrix

// Inputs is the type to hold the inputs to neurons
type Inputs = *linalg.NumericMatrix

// Outputs is the type to hold the outputs of the neural net
type Outputs = *linalg.NumericMatrix

type Activations = *linalg.NumericMatrix

type TrainingData struct {
	Inputs  []linalg.Number
	Targets []linalg.Number
}
