package neural

import (
	"fmt"

	linalg "github.com/jpoffline/linalg/linearalgebra"
)

// New will initialise a new neural network, with the provided
// number of inputs, hidden neurons, and outputs.
func New(numInput, numHidden, numOutput int) *NeuralNet {
	nn := NeuralNet{
		meta: Meta{numInputs: numInput,
			numHidden: numHidden,
			numOutput: numOutput,
		},
	}

	nn.build(nn.meta)
	nn.SetLearningRate(0.05)
	nn.Info(nn.meta)

	return &nn
}

func (nn *NeuralNet) build(meta Meta) {
	nn.weightsIH = linalg.NewRandomMatrix(meta.numHidden, meta.numInputs)
	nn.weightsHO = linalg.NewRandomMatrix(meta.numOutput, meta.numHidden)
	nn.biasIH = linalg.NewRandomMatrix(meta.numHidden, 1)
	nn.biasHO = linalg.NewRandomMatrix(meta.numOutput, 1)
}

// SetLearningRate will set the learning rate of the neural net.
func (nn *NeuralNet) SetLearningRate(lr linalg.Number) {
	nn.meta.learningRate = lr
}

// Info will print out information about the neural net.
func (nn *NeuralNet) Info(meta Meta) {
	fmt.Println("---------------------------------------------")
	fmt.Println("Neural net info")
	fmt.Printf("    => number of inputs: %v\n", meta.numInputs)
	fmt.Printf("    => number of hidden neurons: %v\n", meta.numHidden)
	fmt.Printf("    => number of outputs: %v\n", meta.numOutput)
	fmt.Printf("  learning rate: %v\n", meta.learningRate)
	fmt.Println("---------------------------------------------")
}
