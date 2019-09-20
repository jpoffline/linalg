package neural

import (
	"fmt"
	"os"

	"github.com/jpoffline/linalg/datastore"
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

	return &nn
}

// InitDataStore will initialise the data store for the neural net;
// note that the nn will be indexed with the provided name.
func (nn *NeuralNet) InitDataStore(name string) {
	nn.DataStore = datastore.New(name)
	nn.SetOutputLoc(nn.DataStore.Root())
}

// SetOutputLoc sets the output location for the neural net.
func (nn *NeuralNet) SetOutputLoc(loc string) {
	nn.OutputData.Loc = loc
	os.MkdirAll(nn.OutputData.Loc, os.ModePerm)
}

// Info will print out meta data associated to the neural net.
func (nn *NeuralNet) Info() {
	nn.info(nn.meta)
}

func (nn *NeuralNet) report(format string, a ...interface{}) {
	fmt.Printf("* "+format+"\n", a...)
}

// SetLearningRate will set the learning rate of the neural net.
func (nn *NeuralNet) SetLearningRate(lr linalg.Number) {
	nn.meta.learningRate = lr
}

// Info will print out information about the neural net.
func (nn *NeuralNet) info(meta Meta) {
	fmt.Println("---------------------------------------------")
	fmt.Println("Neural net info")
	fmt.Printf("    => number of inputs: %v\n", meta.numInputs)
	fmt.Printf("    => number of hidden layers: %v\n", 1)
	fmt.Printf("    => number of hidden neurons: %v\n", meta.numHidden)
	fmt.Printf("    => number of outputs: %v\n", meta.numOutput)
	fmt.Printf("  learning rate: %v\n", meta.learningRate)
	fmt.Println("---------------------------------------------")
}
