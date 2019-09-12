package neural

import (
	"fmt"

	"github.com/jpoffline/jpnn/linalg"
)

type NeuralNet struct {
	weightsIH *linalg.NumericMatrix
	weightsHO *linalg.NumericMatrix
	biasIH    *linalg.NumericVector
	biasHO    *linalg.NumericVector
}

// New will initialise a new neural network, with the provided
// number of inputs, hidden neurons, and outputs.
func New(numInput, numHidden, numOutput int) *NeuralNet {
	nn := NeuralNet{}

	nn.weightsIH = linalg.NewRandomMatrix(numHidden, numInput)
	nn.weightsHO = linalg.NewRandomMatrix(numOutput, numHidden)
	nn.biasIH = linalg.NewRandomVector(numHidden)
	nn.biasHO = linalg.NewRandomVector(numOutput)

	return &nn
}

// FeedForward is the place to pass the provided inputs to the neural network.
func (nn *NeuralNet) FeedForward(inputs *linalg.NumericVector) {

	hi := doLayerCalc(nn.weightsIH, inputs, nn.biasIH)
	hi.Print()

	ho := doLayerCalc(nn.weightsHO, hi, nn.biasHO)
	ho.Print()
}

func doLayerCalc(weights *linalg.NumericMatrix, inputs *linalg.NumericVector, bias *linalg.NumericVector) *linalg.NumericVector {
	p1, err := weights.Dot(inputs)
	if err != nil {
		fmt.Println(err)
	}

	p2, err := p1.Sum(bias)
	if err != nil {
		fmt.Println(err)
	}

	return p2.Operate(func(num linalg.Number) linalg.Number { return linalg.Sigmoid(num) })

}
