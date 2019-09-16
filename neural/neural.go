package neural

import (
	"fmt"

	linalg "github.com/jpoffline/linalg/linearalgebra"
)

// New will initialise a new neural network, with the provided
// number of inputs, hidden neurons, and outputs.
func New(numInput, numHidden, numOutput int) *NeuralNet {
	nn := NeuralNet{
		numInputs: numInput,
		numHidden: numHidden,
		numOutput: numOutput,
	}

	nn.build()
	nn.SetLearningRate(0.05)
	nn.Info()

	return &nn
}

func (nn *NeuralNet) build() {
	nn.weightsIH = linalg.NewRandomMatrix(nn.numHidden, nn.numInputs)
	nn.weightsHO = linalg.NewRandomMatrix(nn.numOutput, nn.numHidden)
	nn.biasIH = linalg.NewRandomMatrix(nn.numHidden, 1)
	nn.biasHO = linalg.NewRandomMatrix(nn.numOutput, 1)
}

// SetLearningRate will set the learning rate of the neural net.
func (nn *NeuralNet) SetLearningRate(lr linalg.Number) {
	nn.learningRate = lr
}

// Info will print out information about the neural net.
func (nn *NeuralNet) Info() {
	fmt.Println("---------------------------------------------")
	fmt.Println("Neural net info")
	fmt.Printf("    => number of inputs: %v\n", nn.numInputs)
	fmt.Printf("    => number of hidden neurons: %v\n", nn.numHidden)
	fmt.Printf("    => number of outputs: %v\n", nn.numOutput)
	fmt.Printf("  learning rate: %v\n", nn.learningRate)
	fmt.Println("---------------------------------------------")
}
