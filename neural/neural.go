package neural

import (
	"fmt"

	linalg "github.com/jpoffline/linalg/linearalgebra"
)

// New will initialise a new neural network, with the provided
// number of inputs, hidden neurons, and outputs.
func New(numInput, numHidden, numOutput int) *NeuralNet {
	nn := NeuralNet{
		numInputs:    numInput,
		numHidden:    numHidden,
		numOutput:    numOutput,
		learningRate: 0.1,
	}
	nn.Info()

	nn.weightsIH = linalg.NewRandomMatrix(numHidden, numInput)
	nn.weightsHO = linalg.NewRandomMatrix(numOutput, numHidden)
	nn.biasIH = linalg.NewRandomMatrix(numHidden, 1)
	nn.biasHO = linalg.NewRandomMatrix(numOutput, 1)

	return &nn
}

// Info will print out information about the neural net.
func (nn *NeuralNet) Info() {
	fmt.Println("---------------------------------------------")
	fmt.Println("Neural net info")
	fmt.Printf("    => number of inputs: %v\n", nn.numInputs)
	fmt.Printf("    => number of hidden neurons: %v\n", nn.numHidden)
	fmt.Printf("    => number of outputs: %v\n", nn.numOutput)
	fmt.Println("---------------------------------------------")
}
