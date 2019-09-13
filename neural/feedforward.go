package neural

import (
	linalg "github.com/jpoffline/linalg/linearalgebra"
)

// FeedForward is the place to pass the provided inputs to the neural network.
func (nn *NeuralNet) FeedForward(inputs *linalg.NumericVector) *linalg.NumericVector {
	// prepare the input.
	im := linalg.NewNumericMatrixFromVector(*inputs)
	// generate the hidden outputs.
	hiddenOutputs := doLayerCalc(nn.weightsIH, im, nn.biasIH)
	// generate the final output.
	outputs := doLayerCalc(nn.weightsHO, hiddenOutputs, nn.biasHO)
	// send back to caller as a vector.
	return outputs.ToVector()
}

func doLayerCalc(weights Weights, inputs Inputs, bias Bias) Outputs {
	// multiply W x I
	p1 := weights.Mul(inputs)
	// now add in the bias
	p2 := p1.Add(bias)
	// apply activation function

	// return to caller.
	return p2.Map(func(num linalg.Number) linalg.Number { return linalg.Sigmoid(num) })

}
