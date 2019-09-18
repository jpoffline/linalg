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

// Number is the neural network number type;
// here we type-alias the number-type
// from the linalg library.
type Number = linalg.Number

// Vector is the neural network vector type;
// here we make it a slice of neural-net-Numbers.
type Vector = []Number

// Matrix is the neural network numeric matrix type;
// here we type-alias the pointer to a NumericMatrix
// from the linalg library.
type Matrix = *linalg.NumericMatrix

// Weights is the type to hold the weights connecting neurons.
type Weights = Matrix

// Bias is the type to hold the biases.
type Bias = Matrix

// Inputs is the type to hold the inputs to neurons.
type Inputs = Matrix

// Outputs is the type to hold the outputs of the neural net.
type Outputs = Matrix

// Activations is the type to hold the activations of the neurons.
type Activations = Matrix

type TrainingData struct {
	Inputs  []linalg.Number
	Targets []linalg.Number
}
