package neural

import (
	"fmt"

	"github.com/jpoffline/jpnn/linalg"
)

type NeuralNet struct {
	numInputs, numHidden, numOutput int
	weightsIH                       Weights
	weightsHO                       Weights
	biasIH                          Bias
	biasHO                          Bias
}

type Weights = *linalg.NumericMatrix
type Bias = *linalg.NumericMatrix
type Inputs = *linalg.NumericMatrix
type Outputs = *linalg.NumericMatrix

// New will initialise a new neural network, with the provided
// number of inputs, hidden neurons, and outputs.
func New(numInput, numHidden, numOutput int) *NeuralNet {
	nn := NeuralNet{
		numInputs: numInput,
		numHidden: numHidden,
		numOutput: numOutput,
	}
	nn.Info()

	nn.weightsIH = linalg.NewRandomMatrix(numHidden, numInput)
	nn.weightsHO = linalg.NewRandomMatrix(numOutput, numHidden)
	nn.biasIH = linalg.NewRandomMatrix(numHidden, 1)
	nn.biasHO = linalg.NewRandomMatrix(numOutput, 1)

	return &nn
}

func (nn *NeuralNet) Info() {
	fmt.Println("---------------------------------------------")
	fmt.Println("Neural net info")
	fmt.Printf("    => number of inputs: %v\n", nn.numInputs)
	fmt.Printf("    => number of hidden neurons: %v\n", nn.numHidden)
	fmt.Printf("    => number of outputs: %v\n", nn.numOutput)
	fmt.Println("---------------------------------------------")
}

// FeedForward is the place to pass the provided inputs to the neural network.
func (nn *NeuralNet) FeedForward(inputs *linalg.NumericVector) {

	iv := linalg.NewNumericMatrixFromVector(*inputs)
	hi := doLayerCalc(nn.weightsIH, iv, nn.biasIH)
	ho := doLayerCalc(nn.weightsHO, hi, nn.biasHO)
	ho.Print()
}

func doLayerCalc(weights Weights, inputs Inputs, bias Bias) Outputs {
	fmt.Println("doing layer calc")
	p1, err := weights.Mul(inputs)
	if err != nil {
		fmt.Println(err)
	}

	p2, err := p1.Add(bias)
	if err != nil {
		fmt.Println(err)
	}
	p2.Map(func(num linalg.Number) linalg.Number { return linalg.Sigmoid(num) })
	return p2

}
