package neural

import (
	"fmt"
	"os"

	linalg "github.com/jpoffline/linalg/linearalgebra"
)

func NetworkArchitecture(numInput, numHidden, numOutput int) Meta {
	return Meta{numInputs: numInput,
		numHidden: numHidden,
		numOutput: numOutput,
	}
}

// New will initialise a new neural network, with the provided
// number of inputs, hidden neurons, and outputs.
func New(m Meta) *NeuralNet {
	nn := NeuralNet{
		meta: m,
	}

	nn.build(nn.meta)
	nn.SetLearningRate(0.05)
	nn.Info(nn.meta)

	return &nn
}

// SetOutputLoc sets the output location for the neural net.
func (nn *NeuralNet) SetOutputLoc(loc string) {
	nn.OutputData.Loc = loc
	os.MkdirAll(nn.OutputData.Loc, os.ModePerm)
}

func (nn *NeuralNet) build(meta Meta) {

	lyr0 := neurallayer{
		weights: linalg.NewRandomMatrix(meta.numHidden, meta.numInputs),
		bias:    linalg.NewRandomMatrix(meta.numHidden, 1),
	}
	lyr1 := neurallayer{
		weights: linalg.NewRandomMatrix(meta.numOutput, meta.numHidden),
		bias:    linalg.NewRandomMatrix(meta.numOutput, 1),
	}
	nn.layers = append(nn.layers, lyr0)
	nn.layers = append(nn.layers, lyr1)
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
