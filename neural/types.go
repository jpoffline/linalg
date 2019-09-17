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
	weightsIH  Weights
	weightsHO  Weights
	biasIH     Bias
	biasHO     Bias
}

type OutputData struct {
	Loc string
}

// Weights is the type to hold the weights connecting neurons
type Weights = *linalg.NumericMatrix

// Bias is the type to hold the biases
type Bias = *linalg.NumericMatrix

// Inputs is the type to hold the inputs to neurons
type Inputs = *linalg.NumericMatrix

// Outputs is the type to hold the outputs of the neural net
type Outputs = *linalg.NumericMatrix

type TrainingData struct {
	Inputs  []linalg.Number
	Targets []linalg.Number
}
