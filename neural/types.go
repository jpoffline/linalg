package neural

import (
	linalg "github.com/jpoffline/linalg/linearalgebra"
)

// NeuralNet is the main neural net struct, containing
// all the required data.
type NeuralNet struct {
	numInputs, numHidden, numOutput int
	weightsIH                       Weights
	weightsHO                       Weights
	biasIH                          Bias
	biasHO                          Bias
	learningRate                    linalg.Number
}

// Weights is the type to hold the weights connecting neurons
type Weights = *linalg.NumericMatrix
type Bias = *linalg.NumericMatrix
type Inputs = *linalg.NumericMatrix
type Outputs = *linalg.NumericMatrix
