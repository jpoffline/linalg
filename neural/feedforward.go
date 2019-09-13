package neural

import (
	linalg "github.com/jpoffline/linalg/linearalgebra"
)

// FeedForward is the place to pass the provided inputs to the neural network.
func (nn *NeuralNet) FeedForward(inputs *linalg.NumericVector) *linalg.NumericVector {

	iv := linalg.NewNumericMatrixFromVector(*inputs)
	hi := doLayerCalc(nn.weightsIH, iv, nn.biasIH)
	ho := doLayerCalc(nn.weightsHO, hi, nn.biasHO)
	ho.Print()
	return ho.ToVector()
}
